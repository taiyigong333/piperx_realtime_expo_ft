"""跨机器验证原 EXPO RPC 和图像传输，只使用 mock 设备。"""

import argparse
import json
import logging
import sys
import time

import numpy as np

from piperx_expo.client.rpc import run_client
from piperx_expo.config import ROOT, load_config, resolve_path


def test_config(args):
    config = load_config("configs/client/mock.yaml")
    config.update(server_host=args.host, server_port=args.port, episode_steps=args.steps)
    return config


def run_server(args, config):
    sys.path.insert(0, str(ROOT / "third_party/expo"))
    from expo_ft.env.env_client import EnvClient

    rpc = EnvClient(args.host, args.port)
    try:
        rpc._ensure_server()
        print(f"等待模拟客户端：{args.host}:{args.port}", flush=True)
        spec = {key: config[key] for key in ("language_instruction", "control_hz", "episode_steps")}
        spec["action_mode"] = "absolute_pose_width_m"
        env_id, prompt = rpc.create_env({"env_usage": "network_test", "task_spec": spec})
        peer = rpc._conn.socket.getpeername()[0]
        local = rpc._conn.socket.getsockname()[0]
        if args.expected_peer and peer != args.expected_peer:
            raise AssertionError(f"对端IP为 {peer}，预期 {args.expected_peer}")
        observation, done = rpc.reset(env_id)
        assert prompt == config["language_instruction"] and not done
        durations = []
        dispatches = []
        started = time.perf_counter()
        for step in range(args.steps):
            deadline = started + step / config["control_hz"]
            if deadline > time.perf_counter():
                time.sleep(deadline - time.perf_counter())
            tick = time.perf_counter()
            dispatches.append(tick)
            action = np.r_[observation["cartesian_position"], observation["gripper_position"]]
            action[0] += 0.0001
            executed, kind = rpc.step(env_id, action)
            np.testing.assert_allclose(executed, action)
            assert kind == "policy"
            observation = rpc.get_observation(env_id)
            done, success, reward, mask = rpc.get_info_for_step(env_id)
            assert done == (step == args.steps - 1) and not success and reward == 0 and mask == 1
            assert observation["wrist_image_left"].shape == (224, 224, 3)
            durations.append((time.perf_counter() - tick) * 1000)
        elapsed = time.perf_counter() - started
        assert rpc.render(env_id).shape == (224, 224, 3)
        return {
            "result": "pass",
            "role": "server",
            "local_ip": local,
            "peer_ip": peer,
            "steps": args.steps,
            "control_hz_target": config["control_hz"],
            "achieved_hz": round((args.steps - 1) / (dispatches[-1] - dispatches[0]), 2),
            "total_duration_s": round(elapsed, 3),
            "step_rpc_p50_ms": round(float(np.percentile(durations, 50)), 3),
            "step_rpc_p95_ms": round(float(np.percentile(durations, 95)), 3),
            "step_rpc_max_ms": round(float(max(durations)), 3),
            "scope": "真实跨机 EXPO RPC + 合成RGB/动作；不含模型或真实硬件延迟",
        }
    finally:
        rpc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["server", "client"])
    parser.add_argument("--host", required=True, help="服务器监听/连接地址")
    parser.add_argument("--port", type=int, default=8103)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--expected-peer")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("steps 至少为2")
    config = test_config(args)
    logging.basicConfig(level=logging.INFO)
    if args.role == "server":
        report = run_server(args, config)
    else:
        # 请求数固定，测试结束后正常退出；绝不构造真实硬件环境。
        run_client(config, allow_motion=False, max_requests=3 + 2 * args.steps)
        report = {
            "result": "pass",
            "role": "client",
            "backend": "mock",
            "server": f"{args.host}:{args.port}",
            "steps": args.steps,
        }
    output = resolve_path(args.output or f"runtime/validation/network_{args.role}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
