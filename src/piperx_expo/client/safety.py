"""本地动作边界检查，异常动作直接拒绝，不静默改变训练动作。"""

import numpy as np


def validate_client_config(config: dict):
    for key in (
        "control_hz",
        "watchdog_s",
        "max_translation_step_m",
        "max_rotation_step_rad",
        "max_gripper_step_m",
    ):
        value = float(config[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} 必须是有限正数")
    if not isinstance(config["episode_steps"], int) or config["episode_steps"] < 1:
        raise ValueError("episode_steps 必须是正整数")
    bounds = np.asarray(config["workspace_m"], dtype=float)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("workspace_m 必须是现场校准的3轴有效边界")
    gripper = np.asarray(config["gripper_range_m"], dtype=float)
    if gripper.shape != (2,) or not np.isfinite(gripper).all() or not 0 <= gripper[0] < gripper[1]:
        raise ValueError("gripper_range_m 不合法")
    if config["backend"] == "piperx":
        for key in ("feedback_timeout_s", "camera_timeout_s", "gripper_force_n"):
            if not np.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"{key} 必须是有限正数")
        if not 0 < config["speed_percent"] <= 100:
            raise ValueError("speed_percent 必须在 (0,100] 内")


def validate_action(action, current, config: dict) -> np.ndarray:
    target, state = np.asarray(action, dtype=float), np.asarray(current, dtype=float)
    if target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError("动作必须是7维有限数，拒绝 NaN/Inf 或错误维度")
    bounds = np.asarray(config["workspace_m"], dtype=float)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("workspace_m 必须是现场校准的3轴有效边界")
    if np.any(target[:3] < bounds[:, 0]) or np.any(target[:3] > bounds[:, 1]):
        raise ValueError("目标超出配置工作空间")
    if np.linalg.norm(target[:3] - state[:3]) > config["max_translation_step_m"]:
        raise ValueError("单步位移超过限制")
    delta_angle = (target[3:6] - state[3:6] + np.pi) % (2 * np.pi) - np.pi
    if np.max(np.abs(delta_angle)) > config["max_rotation_step_rad"]:
        raise ValueError("单步旋转超过限制")
    if not (
        -np.pi <= target[3] <= np.pi and -np.pi / 2 <= target[4] <= np.pi / 2 and -np.pi <= target[5] <= np.pi
    ):
        raise ValueError("欧拉角超过 SDK 的 RPY 范围")
    low, high = config["gripper_range_m"]
    if not low <= target[6] <= high:
        raise ValueError("夹爪宽度超限")
    if abs(target[6] - state[6]) > config["max_gripper_step_m"]:
        raise ValueError("夹爪单步变化超限")
    return target.astype(np.float32)
