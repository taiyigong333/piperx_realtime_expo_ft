import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from piperx_expo.client.arm_hold import enable_current_hold
from piperx_expo.client.gui_control import GUIController
from piperx_expo.config import load_config
from test_homing import device_and_home


def recovery_device(code=1):
    device, home = device_and_home()
    arm = device.arm
    arm.joints = np.asarray(home["joint_positions_rad"]).copy()
    arm.status_code = code
    arm.get_driver_states = lambda joint_index: arm.packet(True)
    arm.get_arm_status = lambda: arm.packet(SimpleNamespace(
        arm_status=arm.status_code, err_code=0, ctrl_mode=1, mode_feedback=1 if arm.mode == "j" else 2))

    def reset():
        arm.commands.append(("reset", None))
        arm.status_code = 0

    def enable():
        arm.commands.append(("enable", None))
        return True

    arm.reset, arm.enable = reset, enable
    return device, home


def test_recovery_overwrites_old_target_before_reset_enable():
    device, home = recovery_device()
    actual = device.arm.joints.copy()
    target = enable_current_hold(device, home)
    np.testing.assert_array_equal(target, actual)
    commands = [name for name, value in device.arm.commands]
    assert commands.index("move_j") < commands.index("reset") < commands.index("enable")
    sent = next(value for name, value in device.arm.commands if name == "move_j")
    np.testing.assert_array_equal(sent, actual)


@pytest.mark.parametrize("failure", ["hardware", "stale", "cancel", "clamping"])
def test_recovery_rejects_before_commands(failure):
    device, home = recovery_device()
    if failure == "hardware":
        device.arm.status_code = 7
    elif failure == "stale":
        device.arm.stale = True
    elif failure == "cancel":
        device.reset_cancel.set()
    else:
        device.arm.joints[1] = -0.0444
    with pytest.raises((ValueError, RuntimeError, InterruptedError)):
        enable_current_hold(device, home)
    assert not device.arm.commands


def test_stale_driver_flags_do_not_confirm_enabled():
    device, home = recovery_device(0)
    device.arm.get_driver_states = lambda joint_index: SimpleNamespace(timestamp=time.time()-1, hz=100)
    home.update(control_ready_timeout_s=0.01, poll_s=0.001)
    with pytest.raises(TimeoutError):
        enable_current_hold(device, home)


def test_recovery_requires_motion_authorization_and_no_active_session(tmp_path, monkeypatch):
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    try:
        with pytest.raises(ValueError, match="真机"):
            controller.command("recover_enable", {"motion_confirmed": True})
        controller.config["backend"] = "piperx"
        controller.allow_motion = True
        with pytest.raises(ValueError, match="现场"):
            controller.command("recover_enable", {})
        controller.worker = threading.current_thread()
        with pytest.raises(ValueError, match="先停止"):
            controller.command("recover_enable", {"motion_confirmed": True})
    finally:
        controller.worker = None
        controller.stop()


@pytest.mark.parametrize("ending", ["release", "stop", "lost_browser", "lost_enable"])
def test_hold_session_keeps_enable_until_handoff_or_stop(monkeypatch, ending):
    import sys
    import types
    from piperx_expo.client.arm_hold import ArmHold

    device, home = recovery_device()
    robot = SimpleNamespace(_arm=device.arm, disconnected=False)
    robot.start = lambda: None
    robot.stop = lambda: setattr(robot, "disconnected", True)
    device.arm.electronic_emergency_stop = lambda: device.arm.commands.append(("estop", None))
    config_module = types.ModuleType("taiyi_piper_x_collect.config")
    config_module.load_config = lambda path: SimpleNamespace(robot=object())
    devices_module = types.ModuleType("taiyi_piper_x_collect.devices.piper_x")
    devices_module.PiperXRobot = lambda *args: robot
    monkeypatch.setitem(sys.modules, "taiyi_piper_x_collect.config", config_module)
    monkeypatch.setitem(sys.modules, "taiyi_piper_x_collect.devices.piper_x", devices_module)
    heartbeat = [time.monotonic()]
    session = ArmHold(device.config, lambda: heartbeat[0])
    try:
        deadline = time.monotonic() + 1
        while session.snapshot()["phase"] == "enabling" and time.monotonic() < deadline:
            time.sleep(0.005)
        assert session.snapshot()["phase"] == "enabled", session.snapshot()
        assert not any(cmd == "estop" for cmd, _ in device.arm.commands)
        if ending in {"lost_browser", "lost_enable"}:
            if ending == "lost_browser":
                heartbeat[0] -= 10
            else:
                device.arm.get_joints_enable_status_list = lambda: []
            session.thread.join(1)
            assert session.snapshot()["phase"] == "fault"
        else:
            session.close(release=ending == "release")
        assert not session.thread.is_alive() and robot.disconnected
        assert any(cmd == "estop" for cmd, _ in device.arm.commands) == (ending != "release")
    finally:
        session.close()


def test_hold_only_never_enables_or_resets():
    from piperx_expo.client.arm_hold import hold_current_position

    device, home = recovery_device(0)
    actual = device.arm.joints.copy()
    np.testing.assert_array_equal(hold_current_position(device, home), actual)
    assert [name for name, _ in device.arm.commands] == ["speed", "mode", "move_j"]


@pytest.mark.parametrize("reason", ["disabled", "emergency"])
def test_hold_requires_enabled_robot_without_implicit_recovery(reason):
    from piperx_expo.client.arm_hold import hold_current_position

    device, home = recovery_device(1 if reason == "emergency" else 0)
    if reason == "disabled":
        device.arm.get_joints_enable_status_list = lambda: [False] * 6
    with pytest.raises(RuntimeError, match="需要先恢复使能"):
        hold_current_position(device, home)
    assert not device.arm.commands


def test_separate_home_uses_original_reset_and_preserves_enable_on_handoff(monkeypatch):
    from piperx_expo.client.arm_hold import ArmHold
    from piperx_expo.client.homing import fixed_joint_home

    device, home = recovery_device(0)
    calls = []
    device.reset = lambda confirmed: (calls.append(("reset", confirmed)), fixed_joint_home(device, home))
    device.close = lambda stop_motion=True: calls.append(("close", stop_motion))
    monkeypatch.setattr("piperx_expo.client.devices.PiperXDevices", lambda config: device)
    session = ArmHold(device.config, time.monotonic, action="home")
    try:
        deadline = time.monotonic() + 1
        while session.snapshot()["phase"] == "homing" and time.monotonic() < deadline:
            time.sleep(0.005)
        assert session.snapshot()["phase"] == "homed", session.snapshot()
        assert calls == [("reset", True)]
        targets = [value for name, value in device.arm.commands if name == "move_j"]
        assert targets == [home["joint_positions_rad"]]
        assert not any(name == "reset" for name, value in device.arm.commands)
        session.close(release=True)
        assert calls[-1] == ("close", False)
    finally:
        session.close()


def test_three_commands_dispatch_separate_operations(monkeypatch, tmp_path):
    calls = []

    class Session:
        def __init__(self, config, heartbeat, action):
            calls.append(action)
            self.thread = SimpleNamespace(is_alive=lambda: False)

        def close(self, release=False):
            pass

        def snapshot(self):
            return {"phase": "released", "running": False}

    monkeypatch.setattr("piperx_expo.client.gui_control.ArmHold", Session)
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path, allow_motion=True)
    controller.config["backend"] = "piperx"
    try:
        for action in ("recover_enable", "hold_position", "home_arm"):
            controller.command(action, {"motion_confirmed": True})
        assert calls == ["enable", "hold", "home"]
    finally:
        controller.stop()
