"""固定关节归位；只在现场确认后调用，不向策略经验写入归位动作。"""

from contextlib import nullcontext
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
    # 旧客户端的上限可能是0.08m，正式配置为0.10m；默认沿用各自已确认的硬件边界。
    width = home.setdefault("gripper_open_width_m", config["gripper_range_m"][1])
    if (not np.isfinite(width) or not config["gripper_range_m"][0] < width <= config["gripper_range_m"][1]):
        raise ValueError("复位夹爪张开目标超出配置的米制范围")
    for key, default in (("gripper_open_tolerance_m", .002), ("gripper_open_timeout_s", 5.0)):
        if not np.isfinite(home.get(key, default)) or home.get(key, default) <= 0:
            raise ValueError(f"复位 {key} 必须为有限正数")
    if home.get("gripper_open_tolerance_m", .002) >= width:
        raise ValueError("夹爪张开容差必须小于目标宽度，不能把闭合误报为张开")
    return home


def _rotation(rpy):
    # SDK 使用 Rz*Ry*Rx；直接计算 SO(3) 距离，避免欧拉角跨边界产生假大角度。
    x, y, z = rpy
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])


def _home_feedback(device, getter, shape):
    if device.reset_cancel.is_set():
        raise InterruptedError("归位已停止，禁止继续发送目标")
    value = getter()
    if (value is None or not 0 <= time.time() - value.timestamp < device.config["feedback_timeout_s"]
            or value.hz <= 0):
        raise RuntimeError("归位反馈缺失或过期")
    array = np.asarray(value.msg, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise RuntimeError("归位反馈格式无效")
    return array, value.timestamp


def check_home_start(device, home):
    """只读检查起点，允许GUI在人工调整后重新确认；不会使能或切换控制模式。"""
    arm = device.arm
    target = np.asarray(home["joint_positions_rad"], dtype=float)
    limits = np.asarray(home["joint_limits_rad"], dtype=float)
    joints, _ = _home_feedback(device, arm.get_joint_angles, (6,))
    current, _ = _home_feedback(device, arm.get_flange_pose, (6,))
    tolerance = home["initial_joint_tolerance_rad"]
    if np.any(joints < limits[:, 0] - tolerance) or np.any(joints > limits[:, 1] + tolerance):
        raise ValueError("归位起点关节超限")
    if np.max(np.abs(joints - target)) > home["max_joint_delta_rad"]:
        raise ValueError("归位关节距离超限，需要现场调整")
    bounds = np.asarray(home["start_workspace_m"])
    if np.any(current[:3] < bounds[:, 0]) or np.any(current[:3] > bounds[:, 1]):
        axes = "XYZ"
        details = "；".join(
            f"{axes[i]}={current[i]:.6f}m，允许[{bounds[i, 0]:.6f}, {bounds[i, 1]:.6f}]m"
            for i in range(3) if current[i] < bounds[i, 0] or current[i] > bounds[i, 1]
        )
        raise ValueError(f"归位起点超出允许工作空间：{details}。未使能、未发送归位目标；"
                         "请现场检查姿态/坐标系与home_config，不能跳过工作空间检查")
    return target, joints, current


def fixed_joint_home(device, home):
    arm, cancel, config = device.arm, device.reset_cancel, device.config

    def check_cancel():
        if cancel.is_set():
            raise InterruptedError("归位已停止，禁止继续发送目标")

    def send(method, *args):
        # 与急停共用发送锁，禁止“检查取消后、发目标前”被抢占而覆盖保持目标。
        with getattr(device, "motion_lock", nullcontext()):
            check_cancel()
            return method(*args)

    def feedback(getter, shape):
        return _home_feedback(device, getter, shape)

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
        send(arm.set_speed_percent, speed)
        send(arm.set_motion_mode, mode)
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

    target, joints, current = check_home_start(device, home)
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
        if send(arm.enable):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("归位使能超时")
        cancel.wait(home["poll_s"])
    activate("j", home["speed_percent"])
    check_cancel()
    LOG.info("归位目标(rad)=%s，起点(rad)=%s", target.tolist(), joints.tolist())
    sent_at = time.time()
    send(arm.move_j, target.tolist())  # 固件规划关节路径；不能改用无平滑 move_js 或猜测 TCP 起点。
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


def open_gripper_after_home(device, home):
    """归位后才张开，等待真实宽度到位；避免沿用上条夹持状态触发成功规则。"""
    width = float(home.get("gripper_open_width_m", device.config["gripper_range_m"][1]))
    tolerance = home.get("gripper_open_tolerance_m", .002)
    timeout = home.get("gripper_open_timeout_s", 5.0)

    def check_cancel():
        if device.reset_cancel.is_set():
            raise InterruptedError("夹爪张开已停止，禁止开始采集")

    with getattr(device, "motion_lock", nullcontext()):
        check_cancel()
        sent_at = time.time()
        device.gripper.move_gripper_m(width, float(device.config["gripper_force_n"]))
    LOG.info("复位后张开夹爪，目标宽度=%.6fm", width)
    deadline = time.monotonic() + timeout
    while True:
        check_cancel()
        packet = device.gripper.get_gripper_status()
        if (packet is None or not 0 <= time.time() - packet.timestamp < device.config["feedback_timeout_s"]
                or packet.hz <= 0 or packet.msg.mode != "width"):
            raise RuntimeError("夹爪张开反馈缺失、过期或不是米制宽度，禁止开始采集")
        actual = float(packet.msg.value)
        if not np.isfinite(actual):
            raise RuntimeError("夹爪张开宽度无效，禁止开始采集")
        if packet.timestamp > sent_at and abs(actual - width) <= tolerance:
            device.observe()  # 录制初始观测之前再次确认相机与CAN反馈。
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"夹爪张开超时：实际{actual:.6f}m，目标{width:.6f}m；禁止开始采集")
        device.reset_cancel.wait(home["poll_s"])
