"""兼容上游 EXPO EnvClient 的 WebSocket 客户端。"""

import argparse
import logging
import socket
import time

from openpi_client import msgpack_numpy
from websockets.sync.client import connect

from piperx_expo.client.environment import PiperXEnv
from piperx_expo.config import load_config

LOG = logging.getLogger(__name__)


class Dispatcher:
    def __init__(self, config, allow_motion=False):
        self.config = config
        self.allow_motion = allow_motion
        self.env = None
        self.env_id = None
        self.counter = 0
        self.needs_reset = True

    def handle(self, request):
        operation = request["operation"]
        if operation == "create_env":
            spec = request.get("task_spec")
            if spec and spec.get("review_protocol"):
                raise ValueError("复核服务器需要浏览器客户端；原命令行请使用原服务器配置")
            if spec is None and self.config["backend"] == "piperx":
                raise ValueError("真实客户端要求使用本项目服务器启动器完成任务协议核对")
            if spec is not None:
                for key in ("language_instruction", "control_hz", "episode_steps"):
                    if spec.get(key) != self.config[key]:
                        raise ValueError(f"客户端与服务器 {key} 不一致")
                if spec.get("action_mode") != "absolute_pose_width_m":
                    raise ValueError("服务器动作规范不匹配")
            self.close()
            self.env = PiperXEnv(self.config, self.allow_motion)
            self.counter += 1
            self.env_id = f"piperx_{self.counter}"
            self.needs_reset = True
            return {
                "status": "success",
                "env_id": self.env_id,
                "task_description": self.config["language_instruction"],
            }
        if self.env is None or request.get("env_id") != self.env_id:
            raise ValueError("环境 ID 无效，必须先 create_env")
        response = {"status": "success"}
        if operation == "reset":
            response.update(observation=self.env.reset(), done=False)
            self.needs_reset = False
        elif operation == "step":
            if self.needs_reset:
                raise RuntimeError("必须先 reset 再执行动作")
            # 上游 learner 按 control_hz 控制节拍，执行端不额外 sleep。
            response.update(action=self.env.step(request["action"]), action_type="policy")
        elif operation in {"get_observation", "get_info_for_step"}:
            observation = self.env.get_observation()
            done, success, reward, mask = self.env.get_info_for_step(observation)
            response.update(done=done, success=success, reward=reward, mask=mask)
            if operation == "get_observation":
                response["observation"] = observation
        elif operation == "render":
            response["frame"] = self.env.get_observation()["exterior_image_1_left"]
        else:
            raise ValueError(f"不支持的操作：{operation}")
        return response

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def run_client(config, allow_motion=False, max_requests=None, *, dispatcher=None, stop_event=None,
               on_connection=None):
    dispatcher = dispatcher or Dispatcher(config, allow_motion)
    requests = 0
    uri = f"ws://{config['server_host']}:{config['server_port']}"
    try:
        # 只在首次连接时重试。已连接后断线必须停止设备并退出，避免重发不确定动作。
        while True:
            if stop_event is not None and stop_event.is_set():
                return
            try:
                connection = connect(uri, compression=None, max_size=None, open_timeout=5)
                break
            except OSError:
                LOG.info("等待模型服务器 %s", uri)
                if stop_event is None:
                    time.sleep(2)
                else:
                    stop_event.wait(2)
        with connection:
            if on_connection is not None:
                on_connection(connection)
            connection.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            LOG.info("客户端已连接 %s，backend=%s", uri, config["backend"])
            while max_requests is None or requests < max_requests:
                raw = connection.recv(timeout=config["rpc_timeout_s"])
                try:
                    response = dispatcher.handle(msgpack_numpy.unpackb(raw))
                except Exception as error:
                    dispatcher.close()
                    connection.send(msgpack_numpy.packb({"status": "error", "message": str(error)}))
                    raise
                connection.send(msgpack_numpy.packb(response))
                requests += 1
    finally:
        dispatcher.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/client/mock.yaml")
    parser.add_argument("--allow-motion", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_client(load_config(args.config), args.allow_motion)


if __name__ == "__main__":
    main()
