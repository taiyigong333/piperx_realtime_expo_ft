"""人工恢复使能并保持当前关节；独立于学习器，不恢复旧策略动作。"""

import logging
import threading
import time
from types import SimpleNamespace

import numpy as np

from piperx_expo.client.homing import _home_feedback, load_home
from piperx_expo.config import resolve_path


def enable_current_hold(device, home):
    """只接受正常/软件急停状态；先覆盖旧目标，再复位和使能。"""
    arm, cancel = device.arm, device.reset_cancel

    def status():
        if cancel.is_set():
            raise InterruptedError("恢复使能已取消")
        packet = arm.get_arm_status()
        if (packet is None or packet.hz <= 0
                or not 0 <= time.time() - packet.timestamp < device.config["feedback_timeout_s"]):
            raise RuntimeError("恢复使能控制器反馈缺失或过期")
        code = getattr(packet.msg.arm_status, "value", packet.msg.arm_status)
        if code not in (0, 1) or packet.msg.err_code:
            raise RuntimeError("存在硬件故障，禁止自动清错/使能")
        return packet, code

    joints, _ = _home_feedback(device, arm.get_joint_angles, (6,))
    limits = np.asarray(home["joint_limits_rad"])
    # 保持不能沿用归位的起点测量容差：SDK会把越限目标裁剪，导致使能时移动。
    if np.any(joints < limits[:, 0]) or np.any(joints > limits[:, 1]):
        raise ValueError("当前关节超出SDK目标限位，保持目标会被裁剪；请先现场调整合法姿态")
    if not np.isfinite(home.get("hold_speed_percent", 1)) or not 0 < home.get("hold_speed_percent", 1) <= 10:
        raise ValueError("hold_speed_percent必须在(0,10]内")
    _, code = status()
    # 先在急停/未使能时覆盖旧目标；SDK仍执行自身目标限位检查，不绕过它。
    status()
    arm.set_speed_percent(home.get("hold_speed_percent", 1))
    status()
    arm.set_motion_mode("j")
    status()
    arm.move_j(joints.tolist())
    if code == 1:
        status()
        arm.reset()
    deadline = time.monotonic() + home["control_ready_timeout_s"]
    sent_at = time.time()
    while True:
        packet, code = status()
        actual, _ = _home_feedback(device, arm.get_joint_angles, (6,))
        if np.max(np.abs(actual - joints)) > home["joint_tolerance_rad"]:
            raise RuntimeError("恢复期间关节偏离保持起点")
        if code == 0:
            status()
            arm.enable()
            enabled = list(arm.get_joints_enable_status_list())
            if (packet.timestamp > sent_at and len(enabled) == 6 and all(enabled)
                    and packet.msg.ctrl_mode == arm.ARM_STATUS.CtrlMode.CAN_CTRL
                    and packet.msg.mode_feedback == arm.ARM_STATUS.ModeFeedback.MOVE_J):
                # 六个驱动使能标志也必须来自新鲜反馈，不能相信缓存的True。
                for index in range(1, 7):
                    driver = arm.get_driver_states(joint_index=index)
                    if (driver is None or driver.hz <= 0 or driver.timestamp <= sent_at
                            or not 0 <= time.time() - driver.timestamp < device.config["feedback_timeout_s"]):
                        break
                else:
                    return joints
        if time.monotonic() >= deadline:
            raise TimeoutError("恢复使能/保持模式确认超时")
        cancel.wait(home["poll_s"])


class ArmHold:
    def __init__(self, config, heartbeat):
        self.config = config
        self.heartbeat = heartbeat
        self.cancel = threading.Event()
        self.release_requested = False
        self.info = {"phase": "enabling", "running": True, "error": ""}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def snapshot(self):
        return dict(self.info)

    def _run(self):
        robot = arm = None
        try:
            from taiyi_piper_x_collect.config import load_config
            from taiyi_piper_x_collect.devices.piper_x import PiperXRobot

            collection = load_config(resolve_path(self.config["collection_config"]))
            robot = PiperXRobot(collection.robot, "xyz_rxryrz")
            robot.start()
            arm = robot._arm
            device = SimpleNamespace(arm=arm, config=self.config, reset_cancel=self.cancel)
            home = load_home(self.config)
            deadline = time.monotonic() + home["control_ready_timeout_s"]
            while arm.get_joint_angles() is None or arm.get_arm_status() is None:
                if self.cancel.wait(home["poll_s"]):
                    raise InterruptedError("恢复使能已取消")
                if time.monotonic() >= deadline:
                    raise TimeoutError("等待恢复使能反馈超时")
            target = enable_current_hold(device, home)
            self.info = {"phase": "holding", "running": True, "error": "", "joints_rad": target.tolist()}
            logging.info("机械臂已恢复使能，保持当前关节(rad)=%s", target.tolist())
            while not self.cancel.wait(home["poll_s"]):
                if time.monotonic() - self.heartbeat() > 5:
                    raise RuntimeError("浏览器失联，保持已停止并锁定")
                actual, _ = _home_feedback(device, arm.get_joint_angles, (6,))
                packet = arm.get_arm_status()
                if (packet is None or not 0 <= time.time() - packet.timestamp < self.config["feedback_timeout_s"]
                        or packet.hz <= 0 or packet.msg.arm_status != 0 or packet.msg.err_code
                        or not all(arm.get_joints_enable_status_list())
                        or packet.msg.ctrl_mode != arm.ARM_STATUS.CtrlMode.CAN_CTRL
                        or packet.msg.mode_feedback != arm.ARM_STATUS.ModeFeedback.MOVE_J):
                    raise RuntimeError("保持期间控制状态/使能异常")
                for index in range(1, 7):
                    driver = arm.get_driver_states(joint_index=index)
                    if (driver is None or driver.hz <= 0
                            or not 0 <= time.time() - driver.timestamp < self.config["feedback_timeout_s"]):
                        raise RuntimeError("保持期间驱动反馈过期")
                if np.max(np.abs(actual - target)) > home["joint_tolerance_rad"]:
                    raise RuntimeError("保持期间关节偏离目标")
            self.info = {"phase": "released" if self.release_requested else "stopped",
                         "running": False, "error": ""}
        except BaseException as error:
            logging.exception("恢复使能/保持失败，执行停止锁定")
            self.info = {"phase": "fault", "running": False, "error": str(error)}
        finally:
            try:
                # 仅正常交接给新环境时保留使能；故障、取消、失联仍急停锁定。
                if arm is not None and not (self.release_requested and self.info["phase"] == "released"):
                    arm.electronic_emergency_stop()
            finally:
                if robot is not None:
                    robot.stop()

    def close(self, release=False):
        self.release_requested = release
        self.cancel.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("恢复使能线程尚未退出，禁止建立新硬件会话")
        if release and self.info["phase"] != "released":
            raise RuntimeError("保持状态异常，请现场检查后重新恢复使能")
