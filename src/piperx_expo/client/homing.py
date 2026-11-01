"""固定关节归位；只在现场确认后调用，不向策略经验写入归位动作。"""

import logging
import time

import numpy as np

from piperx_expo.config import load_config

LOG = logging.getLogger(__name__)


def load_home(config):
    home = load_config(config["home_config"])
    target = np.asarray(home["joint_positions_rad"], dtype=float)
    limits = np.asarray(home["joint_limits_rad"], dtype=float)
    bounds = np.asarray(home["start_workspace_m"], dtype=float)
    if (target.shape != (6,) or limits.shape != (6, 2) or not np.isfinite(limits).all()
            or not np.isfinite(target).all() or np.any(limits[:, 0] >= limits[:, 1])
            or np.any(target < limits[:, 0]) or np.any(target > limits[:, 1])):
        raise ValueError("归位关节目标/限位无效")
    if (bounds.shape != (3, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("归位起点工作空间无效")
    for key in ("joint_tolerance_rad", "max_joint_delta_rad", "timeout_s", "log_interval_s",
                "max_distance_m", "max_rotation_rad", "control_ready_timeout_s", "poll_s"):
        if not np.isfinite(home[key]) or home[key] <= 0:
            raise ValueError(f"归位 {key} 必须为有限正数")
    if (not np.isfinite(home["initial_joint_tolerance_rad"]) or home["initial_joint_tolerance_rad"] < 0
            or not np.isfinite(home["speed_percent"]) or not 0 < home["speed_percent"] <= 100):
        raise ValueError("归位速度/起点容差无效")
    return home


def _rotation(rpy):
    # SDK 使用 Rz*Ry*Rx；直接计算 SO(3) 距离，避免欧拉角跨边界产生假大角度。
    x, y, z = rpy
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])


def fixed_joint_home(device, home):
    arm, cancel, config = device.arm, device.reset_cancel, device.config

    def check_cancel():
        if cancel.is_set():
            raise InterruptedError("归位已停止，禁止继续发送目标")

    def feedback(getter, shape):
        check_cancel()
        value = getter()
        if (value is None or not 0 <= time.time() - value.timestamp < config["feedback_timeout_s"]
                or value.hz <= 0):
            raise RuntimeError("归位反馈缺失或过期")
        array = np.asarray(value.msg, dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise RuntimeError("归位反馈格式无效")
        return array, value.timestamp

    def status():
        check_cancel()
        value = arm.get_arm_status()
        if (value is None or not 0 <= time.time() - value.timestamp < config["feedback_timeout_s"]
                or value.hz <= 0):
            raise RuntimeError("归位控制器状态缺失或过期")
        code = getattr(value.msg.arm_status, "value", value.msg.arm_status)
        if code != 0 or value.msg.err_code:
            raise RuntimeError("归位控制器故障；禁止自动清错")
        return value

    def activate(mode, speed):
        check_cancel()
        arm.set_speed_percent(speed)
        arm.set_motion_mode(mode)
        sent_at = time.time()
        deadline = time.monotonic() + home["control_ready_timeout_s"]
        expected = getattr(arm.ARM_STATUS.ModeFeedback, "MOVE_J" if mode == "j" else "MOVE_P")
        while True:
            value = status()
            enabled = list(arm.get_joints_enable_status_list())
            if (value.timestamp > sent_at and value.msg.ctrl_mode == arm.ARM_STATUS.CtrlMode.CAN_CTRL
                    and value.msg.mode_feedback == expected and len(enabled) == 6 and all(enabled)):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("归位控制模式未就绪，禁止发送目标")
            cancel.wait(home["poll_s"])

    target = np.asarray(home["joint_positions_rad"], dtype=float)
    limits = np.asarray(home["joint_limits_rad"], dtype=float)
    joints, _ = feedback(arm.get_joint_angles, (6,))
    current, _ = feedback(arm.get_flange_pose, (6,))
    tolerance = home["initial_joint_tolerance_rad"]
    if np.any(joints < limits[:, 0] - tolerance) or np.any(joints > limits[:, 1] + tolerance):
        raise ValueError("归位起点关节超限")
    if np.max(np.abs(joints - target)) > home["max_joint_delta_rad"]:
        raise ValueError("归位关节距离超限，需要现场调整")
    bounds = np.asarray(home["start_workspace_m"])
    if np.any(current[:3] < bounds[:, 0]) or np.any(current[:3] > bounds[:, 1]):
        raise ValueError("归位起点超出允许工作空间")
    pose = np.asarray(arm.fk(target.tolist()), dtype=float)
    bounds = np.asarray(config["workspace_m"])
    if (pose.shape != (6,) or not np.isfinite(pose).all()
            or np.any(pose[:3] < bounds[:, 0]) or np.any(pose[:3] > bounds[:, 1])):
        raise ValueError("归位目标正运动学结果无效或超出工作空间")
    angle = np.arccos(np.clip((np.trace(_rotation(current[3:]).T @ _rotation(pose[3:])) - 1) / 2, -1, 1))
    if np.linalg.norm(current[:3] - pose[:3]) > home["max_distance_m"] or angle > home["max_rotation_rad"]:
        raise ValueError("归位距离/旋转超限")
    status()
    device.observe()  # 人工等待期间相机也可能失效；运动前重新检查，不能到位后才发现。
    deadline = time.monotonic() + home["control_ready_timeout_s"]
    while True:
        check_cancel()
        if arm.enable():
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("归位使能超时")
        cancel.wait(home["poll_s"])
    activate("j", home["speed_percent"])
    check_cancel()
    LOG.info("归位目标(rad)=%s，起点(rad)=%s", target.tolist(), joints.tolist())
    sent_at = time.time()
    arm.move_j(target.tolist())  # 固件规划关节路径；不能改用无平滑 move_js 或猜测 TCP 起点。
    deadline, next_log = time.monotonic() + home["timeout_s"], 0.0
    while True:
        joints, stamp = feedback(arm.get_joint_angles, (6,))
        feedback(arm.get_flange_pose, (6,))
        status()
        error = float(np.max(np.abs(joints - target)))
        now = time.monotonic()
        if now >= next_log:
            LOG.info("归位最大关节误差=%.6frad", error)
            next_log = now + home["log_interval_s"]
        # 下发成功不等于到位；必须有目标下发后的真实反馈。
        if stamp > sent_at and error <= home["joint_tolerance_rad"]:
            activate("p", config["speed_percent"])
            check_cancel()
            device.observe()
            return
        if now >= deadline:
            raise TimeoutError(f"归位超时，最大关节误差={error:.6f}rad")
        cancel.wait(home["poll_s"])
