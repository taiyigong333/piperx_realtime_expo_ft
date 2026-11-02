"""无浏览器的在线RL终端：人工确认、失败急停与按轨迹数整批传输。"""

import argparse
import copy
import json
import logging
import select
import signal
import sys
import threading
from datetime import datetime

from piperx_expo.client.environment import PiperXEnv
from piperx_expo.client.gui_control import GUIController
from piperx_expo.client.recording import EpisodeRecorder
from piperx_expo.config import load_config


# 数字仅作为原命令的别名，继续复用同一套状态检查和现场运动确认。
COMMAND_ALIASES = {"1": "READY", "2": "end", "3": "s", "4": "f", "5": "stop", "0": "quit"}
COMMAND_HELP = ("命令（输入后回车）：1/READY=现场确认/归位开始；2/end=结束本条；"
                "3/s=成功；4/f=失败；5/stop=急停并标失败；0/quit=退出任务")


class CLIController(GUIController):
    def __init__(self, config, root, allow_motion=False):
        config = copy.deepcopy(config)
        config.pop("remote_server", None)  # 两台终端各启动一个进程，不依赖GUI远程管理。
        config["admission_mode"] = "manual"
        self.episode_stop = threading.Event()
        self.episodes = {}
        self.batch_ids = []
        self.exported_ids = set()
        self.batch_size = None
        super().__init__(config, root, allow_motion)

    def _heartbeat_watch(self):
        # 命令行无浏览器心跳；硬件看门狗仍由PiperXEnv执行。
        while not self.stop_event.wait(.05):
            if self.env and self.env.fault and self.phase == "running":
                self.interrupt_episode(self.env.fault_reason or "设备已停止")

    def _new_recorder(self, observation):
        recorder = EpisodeRecorder(self.root, observation, copy.deepcopy(self.env.config), timestamp_prefix=True)
        self.episodes[recorder.id] = recorder.folder
        return recorder

    def _episode_folder(self, episode_id):
        return self.episodes[episode_id]

    def interrupt_episode(self, reason="操作员急停"):
        if self.episode_stop.is_set():
            return
        if self.phase == "review_pending":
            self.episode_stop.set()
            self.error = reason
            self.env.emergency_stop()
            self.recorder.update(terminal_reason="emergency_stop", error=reason)
            self.submit(False, self.recorder.meta["steps"] > 0, reason)
            return
        if self.phase not in {"running", "homing"}:
            return
        # 不获取RPC/归位锁，先切断硬件；停止事件只属于当前条，不能结束采集任务。
        self.error = reason
        self.episode_stop.set()
        if self.env:
            self.env.emergency_stop()

    def _finish_interrupted(self, *, missing_feedback=False):
        self.pending_action = None
        self.recorder.finish(auto_success=False, terminal_reason="emergency_stop", error=self.error,
                             incomplete_action=missing_feedback)
        self.phase = "review_pending"
        # 有未配对动作时保留失败档案，但禁止把不完整数据送进训练。
        self.submit(False, self.recorder.meta["steps"] > 0 and not missing_feedback, self.error)
        return {"status": "success", "episode_interrupted": True}

    def handle(self, request):
        operation = request["operation"]
        if operation == "create_env":
            if request.get("task_spec", {}).get("cli_batch") is not True:
                raise ValueError("请搭配命令行整批RL服务器配置")
        elif operation == "prepare_policy":
            self.batch_size = request.get("episodes_per_batch")
            if type(self.batch_size) is not int or self.batch_size < 1:
                raise ValueError("服务器未给出每批轨迹数量")
        elif self.env is None or request.get("env_id") != self.env_id or self.stop_event.is_set():
            raise ValueError("命令行会话无效")
        if operation == "reset":
            while True:
                if self.env.closed.is_set() or self.env.fault:
                    self.env.close()
                    device = self._device(self.config) if self.config["backend"] == "piperx" else None
                    self.env = PiperXEnv(copy.deepcopy(self.config), self.allow_motion, device=device,
                                         reset_confirm=self._reset_confirm)
                self.episode_stop.clear()
                try:
                    return super().handle(request)
                except Exception:
                    if not self.episode_stop.is_set():
                        raise
                    # 归位被打断时尚未开始录制，不虚构零动作轨迹；仍等待新一次READY。
                    logging.warning("归位被停止；处理现场后重新输入1或READY")
        if operation in {"step", "get_observation"} and self.episode_stop.is_set():
            return self._finish_interrupted(missing_feedback=self.pending_action is not None)
        if operation == "step":
            try:
                return super().handle(request)
            except Exception as error:
                self.interrupt_episode(str(error))
                return self._finish_interrupted(missing_feedback=True)
        if operation == "get_observation":
            try:
                response = super().handle(request)
            except Exception as error:
                self.interrupt_episode(str(error))
                return self._finish_interrupted(missing_feedback=True)
            if self.episode_stop.is_set() and self.phase == "review_pending":
                self.submit(False, True, self.error)
            return response
        if operation == "episode_ack":
            response = super().handle(request)
            self.batch_ids.append(request["episode_id"])
            return response
        if operation == "export_batch":
            if len(self.batch_ids) != self.batch_size:
                raise RuntimeError("只有采够指定轨迹数才能传输训练数据")
            return {"status": "success", "episodes": [
                json.loads((self.episodes[episode_id] / "metadata.json").read_text())["review"]
                for episode_id in self.batch_ids]}
        if operation == "export_episode":
            episode_id = request["episode_id"]
            if len(self.batch_ids) != self.batch_size or episode_id not in self.batch_ids:
                raise RuntimeError("未完成整批采集，或轨迹不属于当前批次")
            payload = (self.episodes[episode_id] / "episode.hdf5").read_bytes()
            self.exported_ids.add(episode_id)
            return {"status": "success", "hdf5": payload}
        if operation == "batch_received":
            if set(request["episode_ids"]) != set(self.batch_ids) or self.exported_ids != set(self.batch_ids):
                raise ValueError("整批传输确认不完整")
            self.batch_ids.clear()
            self.exported_ids.clear()
            return {"status": "success"}
        response = super().handle(request)
        if operation == "create_env":
            response["cli_batch"] = True
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/client/rotating_square.calibrated.yaml")
    parser.add_argument("--record-dir", default="data/cli_episodes")
    parser.add_argument("--allow-motion", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_config(args.config)
    if config["backend"] == "piperx" and not args.allow_motion:
        parser.error("真机需要--allow-motion和原采集预检")
    session = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f_cli")
    from pathlib import Path
    controller = CLIController(config, Path(args.record_dir) / session, args.allow_motion)
    print(COMMAND_HELP, flush=True)
    controller.start(prepare_only=True)
    previous = None
    previous_signal = signal.signal(signal.SIGINT, lambda *_: controller.interrupt_episode())
    try:
        while not controller.stop_event.is_set():
            state = (controller.phase, controller.error)
            if state != previous:
                print(f"状态：{state[0]} {state[1]}", flush=True)
                previous = state
            if not select.select([sys.stdin], [], [], .1)[0]:
                continue
            line = sys.stdin.readline()
            if not line:
                break
            command = line.strip()
            command = COMMAND_ALIASES.get(command, command)
            try:
                if command == "READY":
                    controller.command("ready", {"motion_confirmed": True})
                elif command == "stop":
                    controller.interrupt_episode()
                elif command == "end":
                    controller.command("end", {})
                elif command in {"s", "f"}:
                    controller.submit(command == "s", True)
                elif command == "quit":
                    break
                else:
                    print(COMMAND_HELP, flush=True)
            except ValueError as error:
                print(str(error), flush=True)
        if controller.error and controller.phase in {"fault", "stopped"}:
            raise RuntimeError(controller.error)
    finally:
        signal.signal(signal.SIGINT, previous_signal)
        controller.shutdown()


if __name__ == "__main__":
    main()
