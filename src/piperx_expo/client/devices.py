"""模拟设备和复用采集 SDK 的 Piper-X 设备。"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from piperx_expo.config import resolve_path
from piperx_expo.observation import CAMERAS, make_observation


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
    def __init__(self, config):
        from taiyi_piper_x_collect.config import load_config
        from taiyi_piper_x_collect.devices.piper_x import PiperXRobot
        from taiyi_piper_x_collect.devices.realsense import RealSenseCamera

        self.config = config
        collection = load_config(resolve_path(config["collection_config"]))
        if collection.session.pose_representation != "xyz_rxryrz":
            raise ValueError("策略控制只支持 xyz_rxryrz，禁止把四元数当欧拉角")
        if any(collection.robot.tool_offset_m):
            raise ValueError("当前回放动作是法兰绝对位姿；需要零 tool_offset_m")
        if {c.name for c in collection.enabled_cameras} != set(CAMERAS):
            raise ValueError("必须配置前视与腕部两路 RGB 相机")
        self.robot = PiperXRobot(collection.robot, "xyz_rxryrz")
        self.cameras = [RealSenseCamera(c) for c in collection.enabled_cameras]
        self.lock = threading.Lock()
        self.frames = {}
        self.failure = None
        self.closed = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=len(self.cameras))
        self.arm = None
        self.gripper = None
        try:
            self.robot.start()
            # 仅初始化反馈，不使能电机、不发送目标。
            self.arm = self.robot._arm
            self.gripper = self.robot._require_gripper()
            for camera in self.cameras:
                camera.start(False)
                self.pool.submit(self._camera_worker, camera)
            deadline = time.monotonic() + 10
            while len(self.frames) < 2 and time.monotonic() < deadline and self.failure is None:
                time.sleep(0.02)
            self.observe()
        except BaseException:
            self.close()
            raise

    def _camera_worker(self, camera):
        try:
            while not self.closed.is_set():
                rgb = camera.read().color[..., ::-1].copy()  # 采集 SDK 的原始输出固定为 BGR。
                with self.lock:
                    self.frames[camera._config.name] = (time.monotonic(), rgb)
        except Exception as error:
            self.failure = error

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
        if self.failure is not None:
            raise RuntimeError("相机读取线程失败") from self.failure
        with self.lock:
            frames = dict(self.frames)
        if set(frames) != set(CAMERAS) or any(
            time.monotonic() - stamp > self.config["camera_timeout_s"] for stamp, _ in frames.values()
        ):
            raise RuntimeError("相机缺帧或过期，拒绝使用旧画面")
        return make_observation(
            self.read_state(), {k: v[1] for k, v in frames.items()}, self.config["language_instruction"]
        )

    def reset(self, confirmed=False):
        # 每个 episode 由现场人员复位，不复用 DROID 的自动抛放动作。
        answer = "READY" if confirmed else input("请人工复位方块/机械臂并确认周围安全，输入 READY 继续：")
        if answer != "READY":
            raise RuntimeError("现场复位未确认")
        deadline = time.monotonic() + 5
        while not self.arm.enable():
            if time.monotonic() > deadline:
                raise RuntimeError("使能超时；请现场检查，不自动调用 reset()")
            time.sleep(0.05)
        self.arm.set_speed_percent(self.config["speed_percent"])
        self.observe()

    def send(self, action):
        self.arm.move_p(action[:6].tolist())
        self.gripper.move_gripper_m(float(action[6]), float(self.config["gripper_force_n"]))

    def stop_hold(self):
        if self.arm is not None:
            self.arm.electronic_emergency_stop()

    def close(self):
        self.closed.set()
        try:
            self.stop_hold()
        finally:
            self.pool.shutdown(wait=True)
            for camera in self.cameras:
                camera.stop()
            self.robot.stop()
