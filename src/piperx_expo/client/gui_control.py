"""浏览器会话状态机；HTTP线程只读缓存，硬件采集和RPC保持同一条控制链。"""

import copy
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from piperx_expo.client.arm_hold import ArmHold
from piperx_expo.client.cameras import CameraStream
from piperx_expo.client.devices import PiperXDevices
from piperx_expo.client.environment import PiperXEnv
from piperx_expo.client.recording import EpisodeRecorder, write_json
from piperx_expo.client.rpc import run_client
from piperx_expo.client.remote_server import RemoteServer
from piperx_expo.client.safety import validate_client_config
from piperx_expo.review import PROTOCOL, observation_state, resolve_review
from piperx_expo.observation import IMAGE_KEYS, make_observation


class FinishRequested(Exception):
    """回合边界的正常停止信号；不作为设备故障处理。"""


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
        self.arm_hold = None
        self.env_id = "browser_review"
        self.recorder = None
        self.connection = None
        self.worker = None
        self.prepare_worker = None
        self.preparation_cancel = threading.Event()
        self.finish_requested = False
        self.training_progress = {}
        self.episode_start_step = 0
        self.latest_obs = None
        self.pending_action = None
        self.policy_status = {}
        self.ready = False
        self.decision = None
        self.end_requested = False
        self.demo = False
        self.batch_training = False
        self.session_mode = "train"
        self.batch_pending_ids = set()
        self.last_heartbeat = time.monotonic()
        self.remote_server = RemoteServer(self.config["remote_server"]) if self.config.get("remote_server") else None
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_watch, daemon=True)
        self.heartbeat_thread.start()
        self.cameras = None
        self.camera_error = ""
        self.camera_ready = threading.Event()
        self.shutdown_event = threading.Event()
        self.camera_thread = None
        if self.config["backend"] == "piperx":
            # 相机与GUI同寿命；不依赖学习器连接、归位或每条回合的录制状态。
            self.camera_thread = threading.Thread(target=self._start_cameras, daemon=True)
            self.camera_thread.start()

    def _start_cameras(self):
        try:
            self.cameras = CameraStream(self.config)
        except Exception as error:
            self.camera_error = str(error)
        finally:
            self.camera_ready.set()

    def _device(self, config, *, ignore_stop=False):
        validate_client_config(config)  # 与CLI一致，在创建CAN设备前检查配置。
        while not self.camera_ready.wait(0.05):
            if (not ignore_stop and self.stop_event.is_set()) or self.shutdown_event.is_set():
                raise InterruptedError("设备连接已取消")
        if self.cameras is None:
            raise RuntimeError("相机启动失败：" + self.camera_error)
        if (not ignore_stop and self.stop_event.is_set()) or self.shutdown_event.is_set():
            raise InterruptedError("设备连接已取消")
        self.cameras.read_images()
        return PiperXDevices(config, cameras=self.cameras)

    def frame(self, camera):
        if self.cameras is not None:
            # 图像预览只读常驻缓存，不能触发CAN、磁盘写入或重开相机。
            images = self.cameras.read_images()
            observation = make_observation(
                np.zeros(7), images, self.config["language_instruction"]
            )
            return observation[IMAGE_KEYS[camera]]
        if self.latest_obs is None:
            raise ValueError(self.camera_error or "暂无画面")
        return self.latest_obs[IMAGE_KEYS[camera]]

    def shutdown(self):
        self.shutdown_event.set()
        try:
            self.stop()
            if self.prepare_worker:
                self.prepare_worker.join(timeout=5)
            if self.worker:
                self.worker.join(timeout=5)
        finally:
            if self.camera_thread:
                self.camera_thread.join()
            if self.cameras:
                self.cameras.close()

    def _heartbeat_watch(self):
        while not self.stop_event.wait(0.25):
            if self.env is not None and getattr(self.env, "fault", False):
                self.stop(getattr(self.env, "fault_reason", "") or "设备控制故障，已停止")
                return
            if self.remote_server and self.phase == "connecting":
                remote = self.remote_server.snapshot()
                if (not remote.get("busy") and not remote.get("running")
                        and remote.get("phase") in {"stopped", "completed", "failed", "idle"}):
                    self.stop("学习器已停止或启动失败，连接等待已结束；请检查服务器错误后重新启动")
                    return
            if (
                self.config["backend"] == "piperx"
                and self.phase in {"running", "homing"}
                and time.monotonic() - self.last_heartbeat > 5
            ):
                self.stop("浏览器失联超过5秒，已停止并锁定")
                return

    def state(self):
        with self.condition:
            self.last_heartbeat = time.monotonic()
            meta = self.recorder.snapshot() if self.recorder else None
            camera_ok = False
            if self.cameras is not None:
                try:
                    self.cameras.read_images()
                    camera_ok = True
                    self.camera_error = ""
                except RuntimeError as error:
                    self.camera_error = str(error)
            return {
                "arm_hold": self.arm_hold.snapshot() if self.arm_hold else {"running": False, "phase": "idle"},
                "policy_status": copy.deepcopy(self.policy_status),
                "training_progress": copy.deepcopy(self.training_progress),
                "finish_requested": self.finish_requested,
                "batch_training": self.batch_training,
                "session_mode": self.session_mode,
                "camera_ready": camera_ok,
                "stop_status": getattr(self.env.device, "stop_result", "") if self.env else "",
                "camera_error": self.camera_error,
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
                "can_start": (self.worker is None or not self.worker.is_alive()) and
                             (self.prepare_worker is None or not self.prepare_worker.is_alive()),
                "state": observation_state(self.latest_obs).tolist() if self.latest_obs else None,
                "remote_server": self.remote_server.snapshot() if self.remote_server else {"enabled": False},
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

    def prepare(self, values):
        """合并启动和连接；只准备模型，归位/策略仍等待单独的ready。"""
        if not self.remote_server:
            raise ValueError("此配置未启用远程训练；模拟体验或高级连接入口仍可使用")
        if set(values) - {"resume_run", "total_steps", "mode"}:
            raise ValueError("准备训练仅接受历史实验和训练总步数")
        if values.get("mode", "train") not in {"train", "eval"}:
            raise ValueError("运行方式必须为继续训练或只推理")
        if values.get("mode") == "eval" and not values.get("resume_run"):
            raise ValueError("只推理需要先选择已有完整模型")
        if "total_steps" in values and (type(values["total_steps"]) is not int
                                        or not 1 <= values["total_steps"] <= 10_000_000):
            raise ValueError("训练总步数必须为1到10000000整数")
        with self.condition:
            if ((self.worker and self.worker.is_alive())
                    or (self.prepare_worker and self.prepare_worker.is_alive())):
                raise ValueError("当前会话尚未结束，请等待完成或停止")
            remote = self.remote_server.snapshot()
            if remote.get("stopping") or remote.get("phase") == "stopping":
                raise ValueError("上一训练仍在停止，请等待退出后再准备")
            if remote.get("running"):
                if remote.get("mode", "train") != values.get("mode", "train"):
                    raise ValueError("已有不同运行方式的会话，请先结束")
                if remote.get("training", {}).get("stage") == "completed":
                    raise ValueError("上一训练正在收尾，请等待退出")
                if values.get("resume_run") and values.get("mode", "train") != "eval":
                    raise ValueError("已有训练运行；连接当前实验前请清空历史实验选择")
                if values.get("mode") == "eval" and values.get("resume_run") != remote.get("evaluation_source", {}).get("run_id"):
                    raise ValueError("已有推理会话加载了另一模型，请先结束")
                if values.get("total_steps") not in (None, remote.get("total_steps")):
                    raise ValueError("已有训练总预算不同，请先结束当前实验再修改")
            else:
                self.remote_server.request("start", **values)
            self.preparation_cancel.clear()
            self.finish_requested = False
            self.training_progress = {}
            self.error = ""
            self.phase = "preparing"
            self.prepare_worker = threading.Thread(target=self._prepare_and_connect, daemon=True)
            self.prepare_worker.start()

    def _prepare_and_connect(self):
        try:
            deadline = time.monotonic() + self.config.get("prepare_timeout_s", 60)
            while not self.preparation_cancel.is_set():
                remote = self.remote_server.snapshot()
                if not remote.get("busy"):
                    if remote.get("error"):
                        raise RuntimeError(remote["error"])
                    if remote.get("running"):
                        with self.condition:
                            if self.preparation_cancel.is_set():
                                return
                            self.training_progress = {
                                "total_steps": remote.get("total_steps"),
                                "completed_steps": remote.get("completed_steps", remote.get("resume_step") or 0),
                            }
                            self.start(prepare_only=True)
                        return
                    if remote.get("phase") in {"failed", "stopped", "completed"}:
                        raise RuntimeError("学习器启动后已退出，请检查服务器日志")
                if time.monotonic() >= deadline:
                    raise TimeoutError("训练启动管理超时，请检查服务器连接与状态")
                self.preparation_cancel.wait(0.1)
        except Exception as error:
            if not self.preparation_cancel.is_set():
                self.stop(str(error))

    def start(self, demo=False, motion_confirmed=False, *, prepare_only=False):
        if self.stop_event.is_set():
            # 旧看门狗线程必须先退出，才能清除停止事件并创建全新硬件会话。
            self.heartbeat_thread.join(timeout=1)
        with self.condition:
            if self.remote_server:
                remote = self.remote_server.snapshot()
                if remote.get("stopping") or remote.get("phase") == "stopping":
                    raise ValueError("服务器仍在停止，请等待后再建立新会话")
                if not remote.get("running") and not remote.get("busy"):
                    raise ValueError("学习器尚未运行，请先启动服务器并等待模型初始化")
            if self.worker is not None and self.worker.is_alive():
                raise ValueError("上一会话尚未退出，请等待停止完成")
            if demo and self.config["backend"] != "mock":
                raise ValueError("模拟演示只支持mock配置")
            if self.config["backend"] == "piperx" and not (self.allow_motion and (motion_confirmed or prepare_only)):
                raise ValueError("真机需要启动参数--allow-motion，并勾选现场运动确认")
            if self.stop_event.is_set():
                if self.heartbeat_thread.is_alive():
                    raise ValueError("上一会话停止检查尚未结束")
                self.stop_event.clear()
                self.error = ""
                self.env = self.connection = self.recorder = self.latest_obs = self.pending_action = None
                self.ready = self.end_requested = self.finish_requested = False
                if not prepare_only:
                    self.training_progress = {}
                self.policy_status = {}
                self.decision = None
                self.batch_training = False
                self.session_mode = "train"
                self.batch_pending_ids.clear()
                self.last_heartbeat = time.monotonic()
                self.heartbeat_thread = threading.Thread(target=self._heartbeat_watch, daemon=True)
                self.heartbeat_thread.start()
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
                logging.exception("GUI控制链失败，停止会话")
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
        while True:
            with self.condition:
                self.ready = False
                self.phase = "awaiting_reset"
            self._wait(lambda: self.ready or self.finish_requested)
            if self.finish_requested:
                raise FinishRequested()
            if self.config["backend"] == "piperx":
                try:
                    # 起点不合法时只读拒绝，保留RPC等待；不让学习器退出并重新加载模型。
                    self.env.device.check_home_start()
                except ValueError as error:
                    with self.condition:
                        self.error = str(error) + "；现场处理后可再次点击确认归位。"
                    continue
                self.phase = "homing"
            self.error = ""
            return

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
            if self.arm_hold:
                self.arm_hold.close(release=self.arm_hold.snapshot()["phase"] in {"enable_sent", "enabled", "holding", "homed"})
                self.arm_hold = None  # 交接后显示当前控制链，旧手动故障保留在日志而非继续冒充当前状态。
            device = self._device(self.config) if self.config["backend"] == "piperx" else None
            with self.condition:
                if self.stop_event.is_set():
                    if device is not None:
                        device.close()
                    raise InterruptedError("设备连接已取消")
                self.env = PiperXEnv(
                    copy.deepcopy(self.config), self.allow_motion, reset_confirm=self._reset_confirm,
                    device=device
                )
            self.latest_obs = self.env.get_observation()
            return {
                "status": "success",
                "env_id": self.env_id,
                "review_protocol": PROTOCOL,
                "task_description": self.config["language_instruction"],
            }
        if self.env is None or request.get("env_id") != self.env_id:
            raise ValueError("环境ID无效")
        response = {"status": "success"}
        if operation == "prepare_policy":
            if self.recorder is not None or self.phase not in {"connecting", "warming"}:
                raise RuntimeError("只能在初始归位前预热模型")
            self.phase = "warming"
            self.session_mode = request.get("session_mode", "train")
            if self.session_mode not in {"train", "eval"}:
                raise ValueError("未知会话方式")
            self.batch_training = request.get("batch_training") is True
            self.env.config["session_mode"] = self.session_mode
            response.update(observation=self.env.get_observation(), watchdog_s=self.config["watchdog_s"],
                            batch_training=self.batch_training, session_mode=self.session_mode)
        elif operation == "policy_status":
            if self.phase != "warming":
                raise RuntimeError("当前未进行模型预热")
            self.policy_status = copy.deepcopy(request.get("details", {}))
        elif operation == "reset":
            if self.config["backend"] == "piperx" and self.policy_status.get("stage") != "ready":
                raise RuntimeError("模型尚未完成首次/RTC预热，禁止归位和采集")
            if self.recorder and self.recorder.meta["status"] not in {
                "accepted",
                "rejected",
                "demo_reviewed",
                "queued",
                "evaluated",
            }:
                raise RuntimeError("上条轨迹尚未收到服务器确认，禁止启动下一条")
            try:
                observation = self.env.reset()
            except FinishRequested:
                self.phase = "finishing"
                return {"status": "success", "stop_after_episode": True}
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
            self.training_progress["completed_steps"] = self.episode_start_step + self.env.steps
            if done:
                self.phase = "archiving"
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
        elif operation == "episode_context":
            start, total = request["start_step"], request["total_steps"]
            if (type(start) is not int or type(total) is not int or not 0 <= start <= total
                    or total < 1 or self.phase == "running"):
                raise ValueError("回合累计步数/总预算无效")
            self.episode_start_step = start
            self.training_progress.update(completed_steps=start, total_steps=total)
            if self.session_mode == "eval":
                self.training_progress.update(checkpoint_step=request.get("checkpoint_step"), session_mode="eval",
                                              update_count=0)
        elif operation in {"training_control", "episode_complete"}:
            if self.recorder and self.recorder.meta["status"] not in {"accepted", "rejected"}:
                raise RuntimeError("先完成本条审核和准入确认，再更新/保存")
            self.training_progress.update(request.get("progress", {}))
            response["stop_after_episode"] = self.finish_requested
            if operation == "episode_complete" and self.recorder:
                self.recorder.update(training=copy.deepcopy(self.training_progress))
        elif operation == "session_complete":
            if self.batch_pending_ids:
                raise RuntimeError("仍有未处理的整批轨迹，不能报告会话完成")
            if self.recorder and self.recorder.meta["status"] not in {"accepted", "rejected", "evaluated"}:
                raise RuntimeError("仍有未确认回合，不能报告会话完成")
            final_progress = request.get("progress", {})
            if self.session_mode != "eval" and self.finish_requested and (final_progress.get("checkpoint_step") is None
                    or final_progress.get("checkpoint_step") != final_progress.get("completed_steps")):
                raise RuntimeError("尚未确认本条checkpoint完整保存，不能报告正常收尾")
            self.training_progress.update(final_progress)
            self.phase = "completed"
            self.stop_event.set()
        elif operation == "episode_ack":
            if self.phase != "submitted" or request.get("episode_id") != self.recorder.id:
                raise RuntimeError("服务器确认与当前轨迹不匹配")
            expected = ("queued" if self.batch_training else "accepted") if self.decision["admitted"] else "rejected"
            if self.session_mode == "eval":
                expected = "evaluated"
            if request.get("outcome") != expected:
                raise RuntimeError("服务器准入结果与人工复核不一致")
            self.recorder.update(status=expected, server_ack=request.get("details", {}))
            if expected == "queued":
                self.batch_pending_ids.add(self.recorder.id)
            self.phase = "waiting_server"
        elif operation == "evaluation_complete":
            if self.session_mode != "eval" or self.recorder.meta["status"] != "evaluated":
                raise RuntimeError("当前不是已记录的推理回合")
            self.training_progress.update(request.get("progress", {}))
            response["stop_after_episode"] = self.finish_requested
        elif operation == "batch_status":
            if not self.batch_training or (self.recorder and self.recorder.meta["status"] not in {
                    "queued", "accepted", "rejected"}):
                raise RuntimeError("先完成本条审核与原始轨迹归档确认，再处理整批")
            stage = request["stage"]
            if stage not in {"episode_queued", "processing_batch", "updating", "training_skipped", "saving"}:
                raise ValueError("未知整批处理阶段")
            self.training_progress.update(request.get("progress", {}), stage=stage)
            self.phase = "waiting_server" if stage == "episode_queued" else "batch_training"
            response["stop_after_episode"] = self.finish_requested
        elif operation == "batch_episode_ack":
            episode_id, outcome = request.get("episode_id"), request.get("outcome")
            if (not self.batch_training or self.phase != "batch_training"
                    or not isinstance(episode_id, str) or len(episode_id) != 32
                    or any(c not in "0123456789abcdef" for c in episode_id)):
                raise RuntimeError("整批准入确认无效")
            if outcome == "accepted" and episode_id in self.batch_pending_ids:
                if self.recorder and self.recorder.id == episode_id:
                    self.recorder.update(status=outcome)
                else:
                    path = self.root / episode_id / "metadata.json"
                    meta = json.loads(path.read_text())
                    if meta["status"] != "queued":
                        raise RuntimeError("整批轨迹状态不一致")
                    write_json(path, {**meta, "status": outcome})
                self.batch_pending_ids.remove(episode_id)
            elif outcome != "rejected":
                raise RuntimeError("整批准入确认与待处理轨迹不一致")
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
            if self.session_mode == "eval":
                labels = resolve_review(self.recorder.meta["auto_success"], manual_success, False)
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
        if action == "prepare":
            self.prepare(values)
        elif action in {"server_start", "server_stop", "server_status"}:
            if not self.remote_server:
                raise ValueError("此配置未启用远程服务器管理")
            if values and (action != "server_start" or set(values) - {"resume_run", "additional_steps", "total_steps", "mode"}):
                raise ValueError("服务器管理不接受覆盖SSH或启动命令")
            if action == "server_stop":
                self.stop()  # 先停止本地运动，再发远程停止请求。
            else:
                self.remote_server.request(action.removeprefix("server_"), **values)
        elif action == "start":
            self.start(bool(values.get("demo", False)), values.get("motion_confirmed") is True)
        elif action in {"recover_enable", "hold_position", "home_arm"}:
            with self.condition:
                if self.config["backend"] != "piperx" or not self.allow_motion or values.get("motion_confirmed") is not True:
                    raise ValueError("机械臂手动操作需要真机启动许可及现场运动确认")
                if self.worker and self.worker.is_alive():
                    raise ValueError("请先停止当前学习器连接会话，再操作机械臂")
                if self.arm_hold and self.arm_hold.thread.is_alive():
                    if self.arm_hold.snapshot()["phase"] not in {"enable_sent", "enabled", "holding", "homed"}:
                        raise ValueError("上一机械臂操作尚未完成")
                    self.arm_hold.close(release=True)
                operation = {"recover_enable": "enable", "hold_position": "hold", "home_arm": "home"}[action]
                # 手动复位借用同一相机流；恢复/保持仍只连接CAN。
                self.arm_hold = ArmHold(copy.deepcopy(self.config), lambda: self.last_heartbeat,
                                        action=operation, device_factory=lambda cfg: self._device(cfg, ignore_stop=True))
        elif action == "cancel_hold":
            with self.condition:
                if self.arm_hold and self.arm_hold.action == "hold":
                    # 只结束本程序保持连接；不发送急停/失能，也不恢复旧目标。
                    self.arm_hold.close(release=True)
        elif action == "ready":
            with self.condition:
                if self.phase != "awaiting_reset" or self.ready or self.finish_requested:
                    raise ValueError("当前未等待现场复位，或已经申请保存后停止")
                if self.config["backend"] == "piperx" and values.get("motion_confirmed") is not True:
                    raise ValueError("请先勾选现场运动确认，再归位并开始本条")
                self.ready = True
                self.condition.notify_all()
        elif action == "review":
            self.submit(values.get("manual_success"), values.get("admitted"), values.get("note", ""))
        elif action == "settings":
            self.settings(values)
        elif action == "finish_training":
            with self.condition:
                if self.demo:
                    raise ValueError("模拟体验不保存模型；请使用结束本条或急停")
                if self.phase not in {"homing", "running", "archiving", "review_pending", "submitted",
                                      "waiting_server", "awaiting_reset", "finishing"}:
                    raise ValueError("请先完成训练准备；当前没有可正常收尾的回合")
                self.finish_requested = True
                self.condition.notify_all()
        elif action == "end":
            if self.phase != "running":
                raise ValueError("当前没有正在执行的回合")
            self.end_requested = True
        elif action == "stop":
            self.stop()
        else:
            raise ValueError("未知命令")

    def stop(self, error=""):
        self.preparation_cancel.set()
        with self.condition:
            error = error or (getattr(self.env, "fault_reason", "") if self.env else "")
            self.error = error
            self.phase = "fault" if error else "stopped"
            self.stop_event.set()
            self.condition.notify_all()
        if self.env is not None and not self.env.closed.is_set():
            self.env.emergency_stop()
        if self.arm_hold:
            self.arm_hold.close()
        # 先切断控制，再等IO和连接退出；复核等待不会占用env锁。
        if self.env is not None:
            self.env.close()
        if self.connection is not None:
            self.connection.close()
        if self.remote_server:
            self.remote_server.request("stop")

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
