"""浏览器会话状态机；HTTP线程只读缓存，硬件采集和RPC保持同一条控制链。"""

import copy
import json
import threading
import time
from pathlib import Path

import numpy as np

from piperx_expo.client.environment import PiperXEnv
from piperx_expo.client.recording import EpisodeRecorder
from piperx_expo.client.rpc import run_client
from piperx_expo.review import PROTOCOL, observation_state, resolve_review


class GUIController:
    def __init__(self, config, root, allow_motion=False):
        self.config = copy.deepcopy(config)
        self.config.setdefault("admission_mode", "manual")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.allow_motion = allow_motion
        self.condition = threading.Condition(threading.RLock())
        self.stop_event = threading.Event()
        self.phase = "idle"
        self.error = ""
        self.env = None
        self.env_id = "browser_review"
        self.recorder = None
        self.connection = None
        self.worker = None
        self.latest_obs = None
        self.pending_action = None
        self.ready = False
        self.decision = None
        self.end_requested = False
        self.demo = False
        self.last_heartbeat = time.monotonic()
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_watch, daemon=True)
        self.heartbeat_thread.start()

    def _heartbeat_watch(self):
        while not self.stop_event.wait(0.25):
            if (
                self.config["backend"] == "piperx"
                and self.phase == "running"
                and time.monotonic() - self.last_heartbeat > 5
            ):
                self.stop("浏览器失联超过5秒，已停止并锁定")
                return

    def state(self):
        with self.condition:
            self.last_heartbeat = time.monotonic()
            meta = self.recorder.snapshot() if self.recorder else None
            return {
                "phase": self.phase,
                "error": self.error,
                "backend": self.config["backend"],
                "demo": self.demo,
                "allow_motion": self.allow_motion,
                "settings": {
                    k: self.config[k]
                    for k in (
                        "episode_steps",
                        "server_host",
                        "server_port",
                        "control_hz",
                        "language_instruction",
                        "admission_mode",
                    )
                },
                "detector": self.config.get("success_detector_config"),
                "episode": meta,
                "connected": self.connection is not None,
                "state": observation_state(self.latest_obs).tolist() if self.latest_obs else None,
            }

    def settings(self, values):
        # 只允许修改与当前硬件安全边界无关的参数；本回合启动后使用快照。
        allowed = {"episode_steps", "server_host", "server_port", "admission_mode"}
        if set(values) - allowed:
            raise ValueError("参数不在允许列表")
        with self.condition:
            if values.get("admission_mode", "manual") not in {"manual", "auto"}:
                raise ValueError("收录模式只支持manual/auto")
            if "episode_steps" in values:
                value = values["episode_steps"]
                if type(value) is not int or not 1 <= value <= 2000:
                    raise ValueError("每轨迹最大步数必须为1到2000整数")
            if "server_port" in values:
                if type(values["server_port"]) is not int or not 1 <= values["server_port"] <= 65535:
                    raise ValueError("服务器端口无效")
            if "server_host" in values:
                host = values["server_host"]
                if (
                    not isinstance(host, str)
                    or not host
                    or len(host) > 253
                    or any(c in host for c in "/ :\\")
                ):
                    raise ValueError("请输入服务器IP或主机名")
            if self.worker and self.worker.is_alive() and set(values) - {"episode_steps", "admission_mode"}:
                raise ValueError("连接地址只能在会话启动前修改")
            self.config.update(values)

    def start(self, demo=False, motion_confirmed=False):
        with self.condition:
            if self.worker is not None:
                raise ValueError("此客户端已启动过会话；停止后请重启服务以完成现场检查")
            if demo and self.config["backend"] != "mock":
                raise ValueError("模拟演示只支持mock配置")
            if self.config["backend"] == "piperx" and not (self.allow_motion and motion_confirmed):
                raise ValueError("真机需要启动参数--allow-motion，并勾选现场运动确认")
            self.demo = demo
            self.phase = "connecting"
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()

    def _run(self):
        try:
            if self.demo:
                self.handle(
                    {
                        "operation": "create_env",
                        "task_spec": {
                            **self.config,
                            "review_protocol": PROTOCOL,
                            "action_mode": "absolute_pose_width_m",
                        },
                    }
                )
                while not self.stop_event.is_set():
                    self.handle({"operation": "reset", "env_id": self.env_id})
                    while not self.stop_event.is_set():
                        action = observation_state(self.latest_obs)
                        action[2] = min(0.30, action[2] + 0.002)
                        action[6] = 0.0366
                        self.handle({"operation": "step", "env_id": self.env_id, "action": action})
                        if self.stop_event.wait(1 / self.config["control_hz"]):
                            return
                        response = self.handle({"operation": "get_observation", "env_id": self.env_id})
                        if response["done"]:
                            break
                    self.handle(
                        {
                            "operation": "review_episode",
                            "env_id": self.env_id,
                            "steps": self.recorder.meta["steps"],
                            "digest": self.recorder.digest.hexdigest(),
                        }
                    )
                    self.recorder.update(status="demo_reviewed")
                    self.phase = "awaiting_reset"
            else:
                run_client(
                    self.config,
                    self.allow_motion,
                    dispatcher=self,
                    stop_event=self.stop_event,
                    on_connection=self._connected,
                )
        except BaseException as error:
            if not self.stop_event.is_set():
                self.stop(str(error))
        finally:
            self.close()

    def _connected(self, connection):
        self.connection = connection

    def _wait(self, predicate):
        with self.condition:
            while not predicate():
                if self.stop_event.is_set():
                    raise RuntimeError("会话已停止")
                self.condition.wait(0.25)

    def _reset_confirm(self):
        with self.condition:
            self.ready = False
            self.phase = "awaiting_reset"
        self._wait(lambda: self.ready)

    def handle(self, request):
        operation = request["operation"]
        if self.stop_event.is_set():
            raise RuntimeError("会话已停止")
        if operation == "create_env":
            spec = request.get("task_spec", {})
            if spec.get("review_protocol") != PROTOCOL:
                raise ValueError("浏览器需要GUI复核服务器配置；命令行仍使用原配置")
            for key in ("language_instruction", "control_hz"):
                if spec.get(key) != self.config[key]:
                    raise ValueError(f"客户端与服务器{key}不一致")
            if spec.get("action_mode") != "absolute_pose_width_m":
                raise ValueError("动作规范不符")
            if self.env is not None:
                raise RuntimeError("禁止在已建立会话中重建硬件")
            self.env = PiperXEnv(
                copy.deepcopy(self.config), self.allow_motion, reset_confirm=self._reset_confirm
            )
            return {
                "status": "success",
                "env_id": self.env_id,
                "review_protocol": PROTOCOL,
                "task_description": self.config["language_instruction"],
            }
        if self.env is None or request.get("env_id") != self.env_id:
            raise ValueError("环境ID无效")
        response = {"status": "success"}
        if operation == "reset":
            if self.recorder and self.recorder.meta["status"] not in {
                "accepted",
                "rejected",
                "demo_reviewed",
            }:
                raise RuntimeError("上条轨迹尚未收到服务器确认，禁止启动下一条")
            observation = self.env.reset()
            with self.condition:
                self.env.config["episode_steps"] = self.config["episode_steps"]
                self.env.config["admission_mode"] = self.config["admission_mode"]
                self.latest_obs = observation
                self.recorder = EpisodeRecorder(self.root, observation, copy.deepcopy(self.env.config))
                self.pending_action = None
                self.end_requested = False
                self.decision = None
                self.phase = "running"
            response.update(observation=observation, done=False)
        elif operation == "step":
            if self.phase != "running" or self.pending_action is not None:
                raise RuntimeError("当前状态不能执行新动作")
            self.pending_action = self.env.step(request["action"])
            response.update(action=self.pending_action, action_type="policy")
        elif operation == "get_observation":
            if self.phase != "running" or self.pending_action is None:
                raise RuntimeError("复核模式要求每个执行目标对应一次后继观测")
            observation = self.env.get_observation()
            done, success, reward, mask = self.env.get_info_for_step(observation)
            if self.end_requested or request.get("force_done", False):
                done = True
                with self.env.lock:
                    self.env.armed = False
                    self.env.terminal = True
                    self.env.last_info = (done, success, reward, mask)
            self.recorder.append(self.pending_action, observation, (reward, mask, done, success))
            self.pending_action = None
            self.latest_obs = observation
            if done:
                self.recorder.finish(
                    auto_success=bool(success),
                    terminal_reason=(
                        "auto_success" if success else "operator_end" if self.end_requested else "step_limit"
                    ),
                )
                self.phase = "review_pending"
                if self.env.config["admission_mode"] == "auto":
                    self.submit(None, True, "自动收录模式")
            response.update(
                observation=observation, done=bool(done), success=bool(success), reward=reward, mask=mask
            )
        elif operation == "review_episode":
            if self.phase not in {"review_pending", "submitted"}:
                raise RuntimeError("轨迹尚未结束")
            if (
                request["steps"] != self.recorder.meta["steps"]
                or request["digest"] != self.recorder.digest.hexdigest()
            ):
                raise RuntimeError("客户端与服务器轨迹长度或SHA256不一致，禁止入池")
            self._wait(lambda: self.decision is not None)
            response["review"] = copy.deepcopy(self.decision)
        elif operation == "session_complete":
            if self.recorder and self.recorder.meta["status"] not in {"accepted", "rejected"}:
                raise RuntimeError("仍有未确认回合，不能报告会话完成")
            self.phase = "completed"
            self.stop_event.set()
        elif operation == "episode_ack":
            if self.phase != "submitted" or request.get("episode_id") != self.recorder.id:
                raise RuntimeError("服务器确认与当前轨迹不匹配")
            expected = "accepted" if self.decision["admitted"] else "rejected"
            if request.get("outcome") != expected:
                raise RuntimeError("服务器准入结果与人工复核不一致")
            self.recorder.update(status=expected, server_ack=request.get("details", {}))
            self.phase = "waiting_server"
        else:
            raise ValueError(f"复核模式不支持操作{operation}")
        return response

    def submit(self, manual_success, admitted, note=""):
        with self.condition:
            if self.phase != "review_pending":
                raise ValueError("只有待复核的已完成轨迹可以提交")
            if not isinstance(note, str) or len(note) > 2000:
                raise ValueError("备注最多2000字")
            labels = resolve_review(self.recorder.meta["auto_success"], manual_success, admitted)
            self.decision = {
                **labels,
                "episode_id": self.recorder.id,
                "steps": self.recorder.meta["steps"],
                "digest": self.recorder.digest.hexdigest(),
                "note": note,
            }
            self.recorder.update(status="submitted", review=self.decision)
            self.phase = "submitted"
            self.condition.notify_all()

    def command(self, action, values):
        if action == "start":
            self.start(bool(values.get("demo", False)), values.get("motion_confirmed") is True)
        elif action == "ready":
            with self.condition:
                if self.phase != "awaiting_reset" or self.ready:
                    raise ValueError("当前未等待现场复位")
                self.ready = True
                self.condition.notify_all()
        elif action == "review":
            self.submit(values.get("manual_success"), values.get("admitted"), values.get("note", ""))
        elif action == "settings":
            self.settings(values)
        elif action == "end":
            if self.phase != "running":
                raise ValueError("当前没有正在执行的回合")
            self.end_requested = True
        elif action == "stop":
            self.stop()
        else:
            raise ValueError("未知命令")

    def stop(self, error=""):
        with self.condition:
            self.error = error
            self.phase = "fault" if error else "stopped"
            self.stop_event.set()
            self.condition.notify_all()
        # 先切断控制，再等IO和连接退出；复核等待不会占用env锁。
        if self.env is not None:
            self.env.close()
        if self.connection is not None:
            self.connection.close()

    def close(self):
        if self.env is not None and not self.env.closed.is_set():
            self.env.close()
        if self.recorder and self.recorder.meta["status"] == "recording":
            self.recorder.finish(
                status="aborted",
                error=self.error,
                pending_action=np.asarray(self.pending_action).tolist()
                if self.pending_action is not None
                else None,
            )
        self.connection = None

    def history(self):
        result = []
        for path in self.root.glob("*/metadata.json"):
            value = json.loads(path.read_text())
            value.pop("points", None)
            result.append(value)
        return sorted(result, key=lambda v: v["created_at"], reverse=True)[:200]
