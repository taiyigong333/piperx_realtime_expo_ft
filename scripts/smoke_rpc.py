"""在模拟设备上使用原 EXPO EnvClient，验证完整请求/响应闭环。"""

import json
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from piperx_expo.client.rpc import run_client
from piperx_expo.config import load_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/expo"))


def smoke():
    from expo_ft.env.env_client import EnvClient

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = load_config("configs/client/mock.yaml")
    config.update(server_port=port, episode_steps=3)
    rpc = EnvClient("127.0.0.1", port)
    rpc._ensure_server()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            # create/reset/3*(step+obs)/render/info = 10 requests。
            future = pool.submit(run_client, config, False, 10)
            env_id, prompt = rpc.create_env({"env_usage": "test"})
            observation, done = rpc.reset(env_id)
            assert prompt == config["language_instruction"] and not done
            for i in range(3):
                action = np.r_[observation["cartesian_position"], observation["gripper_position"]]
                action[0] += 0.001
                executed, kind = rpc.step(env_id, action)
                np.testing.assert_allclose(executed, action)
                assert kind == "policy"
                observation = rpc.get_observation(env_id)
                done, success, reward, mask = rpc.get_info_for_step(env_id)
                assert done == (i == 2) and not success and reward == 0 and mask == 1
            assert rpc.render(env_id).shape == (224, 224, 3)
            rpc.get_info_for_step(env_id)
            future.result(timeout=10)
    finally:
        rpc.close()
    return {"result": "pass", "steps": 3, "protocol": "upstream EnvClient + PiperX mock client"}


if __name__ == "__main__":
    print(json.dumps(smoke(), ensure_ascii=False))
