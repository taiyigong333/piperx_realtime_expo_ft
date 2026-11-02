"""模拟相机/CAN，验证跨回合共享与急停优先级；不访问真实硬件。"""

import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from piperx_expo.client.cameras import CameraStream
from piperx_expo.client.devices import PiperXDevices
from piperx_expo.client.environment import PiperXEnv
from piperx_expo.client.gui_control import GUIController
from piperx_expo.config import load_config
from piperx_expo.observation import CAMERAS


def test_constant_stream_only_stops_when_owner_closes(monkeypatch):
    cameras = []

    class Camera:
        def __init__(self, config):
            self._config = config
            self.starts = self.stops = self.reads = 0
            cameras.append(self)

        def start(self, depth):
            self.starts += 1

        def read(self):
            time.sleep(0.005)
            self.reads += 1
            return SimpleNamespace(color=np.full((10, 10, 3), [1, 2, 3], dtype=np.uint8))

        def stop(self):
            self.stops += 1

    for name in ("taiyi_piper_x_collect", "taiyi_piper_x_collect.devices"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    config_module, camera_module = ModuleType("config"), ModuleType("realsense")
    config_module.load_config = lambda _: SimpleNamespace(
        enabled_cameras=[SimpleNamespace(name=name) for name in CAMERAS])
    camera_module.RealSenseCamera = Camera
    monkeypatch.setitem(sys.modules, "taiyi_piper_x_collect.config", config_module)
    monkeypatch.setitem(sys.modules, "taiyi_piper_x_collect.devices.realsense", camera_module)
    stream = CameraStream({"collection_config": "unused", "camera_timeout_s": 0.2})
    try:
        np.testing.assert_array_equal(stream.read_images()[CAMERAS[0]][0, 0], [3, 2, 1])
        # 多次设备关闭（手动归位/学习器重连）都不拥有相机寿命。
        for _ in range(2):
            device = object.__new__(PiperXDevices)
            device.closed = threading.Event()
            device.owns_cameras, device.camera_stream = False, stream
            device.robot = SimpleNamespace(stop=lambda: None)
            device.close(stop_motion=False)
        assert all(camera.starts == 1 and camera.stops == 0 for camera in cameras)
        assert set(stream.read_images()) == set(CAMERAS)
    finally:
        stream.close()
    stream.close()
    assert all(camera.stops == 1 for camera in cameras)


def test_emergency_does_not_wait_for_environment_lock():
    env = PiperXEnv(load_config("configs/client/mock.yaml"))
    try:
        with env.lock:
            worker = threading.Thread(target=env.emergency_stop)
            worker.start()
            worker.join(0.5)
            assert not worker.is_alive() and env.device.stopped and env.closed.is_set()
        with pytest.raises(RuntimeError):
            env.step(env.config["mock_initial_state"])
    finally:
        env.close()


def test_gui_stop_keeps_camera_and_preview_without_recording(tmp_path):
    gui = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    images = {name: np.zeros((20, 20, 3), dtype=np.uint8) for name in CAMERAS}
    gui.cameras = SimpleNamespace(read_images=lambda: images)
    try:
        gui.stop()
        assert gui.frame(0).shape == (224, 224, 3)
        assert gui.recorder is None and not list(tmp_path.iterdir())
    finally:
        gui.cameras = None
        gui.shutdown()


def test_stop_holds_fresh_pose_once_and_rejects_later_targets():
    device = object.__new__(PiperXDevices)
    pose = [0.3, 0, 0.2, 0.1, 0.2, -0.3]
    commands = []
    device.config = load_config("configs/client/rotating_square.calibrated.yaml")
    device.motion_lock = threading.Lock()
    device.motion_stopped = threading.Event()
    device.reset_cancel = threading.Event()
    device.stop_sent = False
    device.gripper = None
    device.arm = SimpleNamespace(
        get_flange_pose=lambda: SimpleNamespace(timestamp=time.time(), msg=pose),
        move_p=lambda target: commands.append(target),
        electronic_emergency_stop=lambda: commands.append("fallback"))
    device.stop_hold()
    device.stop_hold()
    assert commands == [pose]
    with pytest.raises(RuntimeError, match="急停"):
        device.send(pose + [0.04])
    assert commands == [pose]


def test_stop_with_stale_feedback_falls_back_without_guessing():
    from piperx_expo.client.devices import freeze_position

    commands = []
    arm = SimpleNamespace(
        get_flange_pose=lambda: SimpleNamespace(timestamp=time.time() - 1, msg=[0] * 6),
        move_p=lambda _: pytest.fail("禁止用过期反馈保持"),
        electronic_emergency_stop=lambda: commands.append("fallback"))
    assert "阻尼急停" in freeze_position(arm, {"feedback_timeout_s": 0.2})
    assert commands == ["fallback"]


def test_shared_camera_does_not_skip_first_can_feedback_wait(monkeypatch):
    for name in ("taiyi_piper_x_collect", "taiyi_piper_x_collect.devices"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    collection = SimpleNamespace(
        session=SimpleNamespace(pose_representation="xyz_rxryrz"),
        robot=SimpleNamespace(tool_offset_m=[0, 0, 0]),
        enabled_cameras=[SimpleNamespace(name=name) for name in CAMERAS])
    config_module, robot_module = ModuleType("config"), ModuleType("piper_x")
    config_module.load_config = lambda _: collection
    robot_module.PiperXRobot = lambda *args: SimpleNamespace(
        start=lambda: None, stop=lambda: None, _arm=object(), _require_gripper=lambda: None)
    monkeypatch.setitem(sys.modules, "taiyi_piper_x_collect.config", config_module)
    monkeypatch.setitem(sys.modules, "taiyi_piper_x_collect.devices.piper_x", robot_module)
    attempts = []

    def read_state(self):
        attempts.append(True)
        if len(attempts) < 3:
            raise RuntimeError("CAN首帧未到达")
        return np.zeros(7)

    monkeypatch.setattr(PiperXDevices, "read_state", read_state)
    monkeypatch.setattr(PiperXDevices, "observe", lambda _: None)
    device = PiperXDevices(load_config("configs/client/rotating_square.calibrated.yaml"),
                           cameras=SimpleNamespace(read_images=lambda: {}))
    try:
        assert len(attempts) == 3 and not device.motion_stopped.is_set()
    finally:
        device.close(stop_motion=False)
