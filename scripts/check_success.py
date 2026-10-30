"""只读检查现场成功姿态的几何条件；不使能、不发送动作，也不构造控制环境。"""

import argparse
import json
import time

import numpy as np

from piperx_expo.client.success import GraspSuccessDetector
from piperx_expo.config import load_config, resolve_path


def main():
    from taiyi_piper_x_collect.config import load_config as load_collection
    from taiyi_piper_x_collect.devices.piper_x import PiperXRobot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    config = load_config(args.config)
    detector = GraspSuccessDetector(config["success_detector_config"])
    collection = load_collection(resolve_path(config["collection_config"]))
    if args.samples < detector.required or collection.session.pose_representation != "xyz_rxryrz":
        raise ValueError("采样数不足或位姿格式不符")
    robot = PiperXRobot(collection.robot, "xyz_rxryrz")
    rows = []
    try:
        robot.start()
        robot.wait_for_gripper_feedback()
        for _ in range(args.samples):
            state = robot.read()
            feedback = [robot._arm.get_flange_pose(), robot._require_gripper().get_gripper_status()]
            if any(f is None or not 0 <= time.time() - f.timestamp < config["feedback_timeout_s"]
                   for f in feedback):
                raise RuntimeError("只读检查收到过期反馈")
            rows.append([*state.tcp_pose.tolist(), robot.read_gripper_position()])
            time.sleep(1 / config["control_hz"])
    finally:
        # collector 的 stop 只断开CAN，不调用控制环境的 stop_hold/急停。
        robot.stop()
    values = np.asarray(rows)
    matches = (values[:, 2] > detector.height) & (values[:, 6] >= detector.width[0]) & (
        values[:, 6] <= detector.width[1])
    report = {"read_only": True, "samples": len(rows), "min": values.min(axis=0).tolist(),
              "max": values.max(axis=0).tolist(), "geometry_match_count": int(matches.sum()),
              "scope": "只验证高度/实际宽度；未验证动作闭合条件、空夹负例或真实RL"}
    output = resolve_path("runtime/validation/success_pose.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
