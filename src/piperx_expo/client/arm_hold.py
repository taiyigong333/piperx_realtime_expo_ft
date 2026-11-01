"""独立的恢复使能、当前位置保持和固定复位；不恢复旧策略动作。"""

import logging
import threading
import time
from types import SimpleNamespace

import numpy as np

from piperx_expo.client.homing import _home_feedback, load_home
from piperx_expo.config import resolve_path


def restore_enable(device):
    """仅发送SDK使能指令，不关联归位、保持、模式切换或控制器复位。"""
    if device.reset_cancel.is_set():
        raise InterruptedError("恢复使能已取消")
    device.arm.enable()


def hold_current_position(device, home=None):
    """读取一次法兰位置并锁定；不把归位检查作为手动保持的前提。"""
    if device.reset_cancel.is_set():
        raise InterruptedError("保持位置已取消")
    # CAN刚连接时等待首帧；这只取得保持目标，不检查其时效/状态/工作空间。
    deadline = time.monotonic() + device.config.get("manual_feedback_wait_s", 2)
    packet = device.arm.get_flange_pose()
    while packet is None:
        if device.reset_cancel.wait(0.02):
            raise InterruptedError("保持位置已取消")
        if time.monotonic() >= deadline:
            raise RuntimeError("无法读取当前位置（CAN尚无位姿反馈），未发送保持目标")
        packet = device.arm.get_flange_pose()
    pose = np.asarray(packet.msg, dtype=float)
    if device.reset_cancel.is_set():
        raise InterruptedError("保持位置已取消")
    # 原样使用米/rad法兰位姿，避免SDK将负关节角裁剪成零而产生移动。
    # 不额外判断反馈年龄、关节限位、工作区、使能或控制器模式；SDK自身格式校验保留。
    device.arm.move_p(pose.tolist())
    return pose


def _enable_confirmation(arm, timeout):
    """使能后的只读显示，不拦截发送，也不因反馈未到触发急停。"""
    try:
        packets = [arm.get_driver_states(joint_index=i) for i in range(1, 7)]
        if any(p is None or not 0 <= time.time() - p.timestamp < timeout or p.hz <= 0 for p in packets):
            return False
        flags = list(arm.get_joints_enable_status_list())
        return len(flags) == 6 and all(flags)
    except Exception:
        return False


class ArmHold:
    def __init__(self, config, heartbeat, action="enable"):
        if action not in {"enable", "hold", "home"}:
            raise ValueError("未知机械臂手动操作")
        self.action = action
        self.config = config
        self.heartbeat = heartbeat
        self.cancel = threading.Event()
        self.release_requested = False
        self.info = {"phase": {"enable": "enabling", "hold": "locking", "home": "homing"}[action],
                     "action": action, "running": True, "error": ""}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def snapshot(self):
        return dict(self.info)

    def _run(self):
        robot = arm = full_device = None
        try:
            home = None
            poll_s = 0.05
            if self.action == "home":
                home = load_home(self.config)
                poll_s = home["poll_s"]
                from piperx_expo.client.devices import PiperXDevices

                # 固定复位沿用原设备/相机预检和全部归位检查，不单独绕过运动前检查。
                full_device = PiperXDevices(self.config)
                full_device.reset_cancel = self.cancel
                arm = full_device.arm
                device = full_device
                full_device.reset(confirmed=True)
                target = np.asarray(home["joint_positions_rad"], dtype=float)
                phase, expected_mode = "homed", arm.ARM_STATUS.ModeFeedback.MOVE_P
            else:
                from taiyi_piper_x_collect.config import load_config
                from taiyi_piper_x_collect.devices.piper_x import PiperXRobot

                collection = load_config(resolve_path(self.config["collection_config"]))
                robot = PiperXRobot(collection.robot, "xyz_rxryrz")
                robot.start()
                arm = robot._arm
                device = SimpleNamespace(arm=arm, config=self.config, reset_cancel=self.cancel)
                if self.action == "enable":
                    restore_enable(device)
                    phase, target = "enable_sent", None
                else:
                    target = hold_current_position(device)
                    phase = "holding"
            self.info = {"phase": phase, "action": self.action, "running": True, "error": ""}
            if target is not None:
                self.info["joints_rad" if self.action == "home" else "pose_m_rad"] = target.tolist()
            logging.info("机械臂手动操作%s指令已发送", self.action)
            while not self.cancel.wait(poll_s):
                if time.monotonic() - self.heartbeat() > 5:
                    raise RuntimeError("浏览器失联，手动操作已停止并锁定")
                if self.action == "enable":
                    confirmed = _enable_confirmation(arm, self.config["feedback_timeout_s"])
                    self.info = {"phase": "enabled" if confirmed else "enable_sent", "action": "enable",
                                 "running": True, "error": "", "enable_confirmed": confirmed}
                    continue
                if self.action == "hold":
                    # 目标只读取/发送一次；不再用归位反馈判定撤销用户的保持操作。
                    continue
                actual, _ = _home_feedback(device, arm.get_joint_angles, (6,))
                packet = arm.get_arm_status()
                enabled = list(arm.get_joints_enable_status_list())
                if (packet is None or not 0 <= time.time() - packet.timestamp < self.config["feedback_timeout_s"]
                        or packet.hz <= 0 or packet.msg.arm_status != 0 or packet.msg.err_code
                        or len(enabled) != 6 or not all(enabled)
                        or packet.msg.ctrl_mode != arm.ARM_STATUS.CtrlMode.CAN_CTRL
                        or packet.msg.mode_feedback != expected_mode):
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
            if self.release_requested and self.cancel.is_set() and isinstance(error, InterruptedError):
                self.info = {"phase": "released", "running": False, "error": ""}
            else:
                logging.exception("机械臂手动操作失败，执行停止锁定")
                self.info = {"phase": "fault", "running": False, "error": str(error)}
        finally:
            try:
                # 仅正常交接给新环境时保留使能；故障、取消、失联仍急停锁定。
                released = self.release_requested and self.info["phase"] == "released"
                if full_device is not None:
                    full_device.close(stop_motion=not released)
                elif arm is not None and not released:
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
