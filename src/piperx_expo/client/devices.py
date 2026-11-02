"""模拟设备和复用采集 SDK 的 Piper-X 设备。"""

import threading
import time

import numpy as np

from piperx_expo.client.cameras import CameraStream
from piperx_expo.config import resolve_path
from piperx_expo.observation import CAMERAS, make_observation


def freeze_position(arm, config, gripper=None):
    """锁存当前法兰/夹爪并保留使能；反馈或发送失败则退回阻尼急停。"""
    try:
        packet = arm.get_flange_pose()
        if packet is None or not 0 <= time.time() - packet.timestamp < config["feedback_timeout_s"]:
            raise RuntimeError("停止时法兰反馈缺失或过期")
        pose = np.asarray(packet.msg, dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise RuntimeError("停止时法兰反馈无效")
        # 用当前位置覆盖旧目标并保留使能；SDK会切到P模式。
        # 不用move_j，避免起点负角反馈被SDK关节限位裁剪后产生位移。
        arm.move_p(pose.tolist())
        result = "已锁存当前位置，后续动作已禁止；保持效果待现场确认"
        if gripper is not None:
            feedback = gripper.get_gripper_status()
            if (feedback is not None
                    and 0 <= time.time() - feedback.timestamp < config["feedback_timeout_s"]
                    and feedback.msg.mode == "width"):
                width = float(feedback.msg.value)
                if np.isfinite(width) and 0 <= width <= config["gripper_range_m"][1]:
                    gripper.move_gripper_m(width, float(config["gripper_force_n"]))
    except Exception:
        # 保持无法确认时使用SDK阻尼急停，绝不猜测保持目标或自动恢复使能。
        arm.electronic_emergency_stop()
        result = "保持失败，已请求阻尼急停；请现场检查"

    return result


class MockDevices:
    def __init__(self, config):
        self.config = config
        self.initial = np.array(config["mock_initial_state"], dtype=np.float32)
        self.state = self.initial.copy()
        self.stopped = False

    def observe(self):
        images = {name: np.full((224, 224, 3), 64 + i * 32, dtype=np.uint8) for i, name in enumerate(CAMERAS)}
        return make_observation(self.state, images, self.config["language_instruction"])

    def read_state(self):
        return self.state.copy()

    def reset(self, confirmed=False):
        if self.stopped:
            raise RuntimeError("设备已停止，需要重新创建会话")
        self.state = self.initial.copy()

    def send(self, action):
        if self.stopped:
            raise RuntimeError("设备已停止")
        self.state = np.asarray(action, dtype=np.float32).copy()

    def stop_hold(self):
        self.stopped = True

    def close(self):
        self.stop_hold()


class PiperXDevices:
    def __init__(self, config, cameras=None):
        from taiyi_piper_x_collect.config import load_config
        from taiyi_piper_x_collect.devices.piper_x import PiperXRobot

        self.config = config
        collection = load_config(resolve_path(config["collection_config"]))
        if collection.session.pose_representation != "xyz_rxryrz":
            raise ValueError("策略控制只支持 xyz_rxryrz，禁止把四元数当欧拉角")
        if any(collection.robot.tool_offset_m):
            raise ValueError("当前回放动作是法兰绝对位姿；需要零 tool_offset_m")
        if {c.name for c in collection.enabled_cameras} != set(CAMERAS):
            raise ValueError("必须配置前视与腕部两路 RGB 相机")
        self.robot = PiperXRobot(collection.robot, "xyz_rxryrz")
        self.camera_stream = cameras
        self.owns_cameras = cameras is None
        self.closed = threading.Event()
        self.reset_cancel = threading.Event()
        self.motion_stopped = threading.Event()
        self.motion_lock = threading.Lock()
        self.stop_sent = False
        self.stop_result = ""
        self.arm = None
        self.gripper = None
        try:
            self.robot.start()
            # 仅初始化反馈，不使能电机、不发送目标。
            self.arm = self.robot._arm
            self.gripper = self.robot._require_gripper()
            if self.camera_stream is None:
                self.camera_stream = CameraStream(config)
            self.camera_stream.read_images()
            # 复用相机时没有原先的启动等待；CAN新连接仍需等到首批新鲜反馈。
            deadline = time.monotonic() + config.get("feedback_start_timeout_s", 2)
            while True:
                try:
                    self.read_state()
                    packets = (self.arm.get_joint_angles(), self.arm.get_flange_pose(), self.arm.get_arm_status())
                    if any(packet is None or packet.hz <= 0 or not
                           0 <= time.time() - packet.timestamp < config["feedback_timeout_s"] for packet in packets):
                        raise RuntimeError("CAN首批关节/法兰/控制器反馈尚未稳定")
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
            self.observe()
        except BaseException:
            self.close()
            raise

    def read_state(self):
        for feedback in (
            self.arm.get_joint_angles(),
            self.arm.get_flange_pose(),
            self.gripper.get_gripper_status(),
        ):
            if (
                feedback is None
                or not 0 <= time.time() - feedback.timestamp < self.config["feedback_timeout_s"]
            ):
                raise RuntimeError("CAN 反馈缺失或过期，拒绝继续控制")
        state = self.robot.read()
        return np.r_[state.tcp_pose, self.robot.read_gripper_position()].astype(np.float32)

    def observe(self):
        return make_observation(
            self.read_state(), self.camera_stream.read_images(), self.config["language_instruction"]
        )

    def check_home_start(self):
        from piperx_expo.client.homing import check_home_start, load_home

        check_home_start(self, load_home(self.config))

    def reset(self, confirmed=False):
        from piperx_expo.client.homing import fixed_joint_home, load_home

        # 方块由现场复位；机械臂每回合都返回同一关节目标，不使用 DROID 抛放动作。
        answer = "READY" if confirmed else input("请复位方块并确认归位路径安全，输入 READY 执行固定关节归位：")
        if answer != "READY":
            raise RuntimeError("现场复位未确认")
        fixed_joint_home(self, load_home(self.config))

    def send(self, action):
        # 急停先置闩锁；与当前一次SDK发送串行，之后不允许任何策略目标接管。
        with self.motion_lock:
            if self.motion_stopped.is_set():
                raise RuntimeError("设备已急停，禁止继续发送动作")
            self.arm.move_p(action[:6].tolist())
            if not self.motion_stopped.is_set():
                self.gripper.move_gripper_m(float(action[6]), float(self.config["gripper_force_n"]))

    def stop_hold(self):
        self.motion_stopped.set()
        self.reset_cancel.set()
        with self.motion_lock:
            if self.arm is None or self.stop_sent:
                return
            self.stop_sent = True
            self.stop_result = freeze_position(self.arm, self.config, self.gripper)

    def close(self, stop_motion=True):
        if self.closed.is_set():
            return
        self.closed.set()
        try:
            if stop_motion:
                self.stop_hold()
        finally:
            # GUI借用常驻相机；回合/连接/CAN结束不能关闭它。
            try:
                if self.owns_cameras and self.camera_stream is not None:
                    self.camera_stream.close()
            finally:
                self.robot.stop()
