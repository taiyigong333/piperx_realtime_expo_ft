import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from piperx_expo.client.arm_hold import restore_enable, hold_current_position, _enable_confirmation
from piperx_expo.client.gui_control import GUIController
from piperx_expo.config import load_config
from test_homing import device_and_home


def recovery_device(code=1):
    device, home = device_and_home()
    arm = device.arm
    arm.joints = np.asarray(home["joint_positions_rad"]).copy()
    arm.status_code = code
    arm.get_driver_states = lambda joint_index: arm.packet(
        SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=True)))
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


def test_enable_only_ignores_all_home_feedback():
    device, home = recovery_device()
    device.arm.get_joint_angles = lambda: pytest.fail("恢复使能不能读关节")
    device.arm.get_flange_pose = lambda: pytest.fail("恢复使能不能读位姿")
    restore_enable(device)
    assert device.arm.commands == [("reset", None), ("speed", device.config["speed_percent"]),
                                   ("mode", "p"), ("enable", None)]


def test_cancelled_enable_does_not_send_command():
    device, home = recovery_device()
    device.reset_cancel.set()
    with pytest.raises(InterruptedError):
        restore_enable(device)
    assert not device.arm.commands


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


@pytest.mark.parametrize("ending", ["release", "stop", "lost_browser"])
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
        assert session.snapshot()["phase"] in {"enable_sent", "enabled"}, session.snapshot()
        assert not any(cmd == "estop" for cmd, _ in device.arm.commands)
        if ending == "lost_browser":
            heartbeat[0] -= 10
            session.thread.join(1)
            assert session.snapshot()["phase"] == "fault"
        else:
            session.close(release=ending == "release")
        assert not session.thread.is_alive() and robot.disconnected
        assert any(cmd == "estop" for cmd, _ in device.arm.commands) == (ending != "release")
    finally:
        session.close()


def test_hold_reads_one_pose_without_home_checks_or_implicit_commands():
    device, home = recovery_device(7)
    pose = [-0.08, 0, -0.01, -2.3, 0.1, -1.57]
    reads = []
    device.arm.get_flange_pose = lambda: (reads.append(True) or SimpleNamespace(msg=pose, timestamp=0, hz=0))
    device.arm.get_joint_angles = lambda: pytest.fail("保持不能读取/检查关节")
    device.arm.get_arm_status = lambda: pytest.fail("保持不能判断控制器状态")
    device.arm.move_p = lambda value: device.arm.commands.append(("move_p", value))
    np.testing.assert_array_equal(hold_current_position(device), pose)
    assert len(reads) == 1
    assert device.arm.commands == [("move_p", pose)]


def test_hold_missing_pose_cannot_invent_target():
    device, home = recovery_device()
    device.config["manual_feedback_wait_s"] = 0.001
    device.arm.get_flange_pose = lambda: None
    device.arm.move_p = lambda value: pytest.fail("无位姿不能发送目标")
    with pytest.raises(RuntimeError, match="无法读取当前位置"):
        hold_current_position(device)


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
        def __init__(self, config, heartbeat, action, device_factory=None):
            calls.append(action)
            self.thread = SimpleNamespace(is_alive=lambda: False)
            self.action = action
            self.release = None

        def close(self, release=False):
            self.release = release

        def snapshot(self):
            return {"phase": "released", "running": False}

    monkeypatch.setattr("piperx_expo.client.gui_control.ArmHold", Session)
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path, allow_motion=True)
    controller.config["backend"] = "piperx"
    try:
        for action in ("recover_enable", "hold_position", "home_arm"):
            controller.command(action, {"motion_confirmed": True})
        assert calls == ["enable", "hold", "home"]
        controller.command("hold_position", {"motion_confirmed": True})
        controller.command("cancel_hold", {})
        assert controller.arm_hold.release is True
    finally:
        controller.stop()


@pytest.mark.parametrize("action", ["enable", "hold"])
def test_simple_actions_do_not_load_home_or_keep_checking_positions(monkeypatch, action):
    import sys
    import types
    from piperx_expo.client.arm_hold import ArmHold

    device, home = recovery_device()
    calls = []
    device.arm.get_flange_pose = lambda: (calls.append("read") or device.arm.packet([0.1, 0, 0.2, 0, 0, 0]))
    device.arm.move_p = lambda value: calls.append(("move_p", value))
    device.arm.get_driver_states = lambda joint_index: None
    device.arm.electronic_emergency_stop = lambda: calls.append("estop")
    robot = SimpleNamespace(_arm=device.arm, start=lambda: None, stop=lambda: calls.append("disconnect"))
    config_module = types.ModuleType("taiyi_piper_x_collect.config")
    config_module.load_config = lambda path: SimpleNamespace(robot=object())
    devices_module = types.ModuleType("taiyi_piper_x_collect.devices.piper_x")
    devices_module.PiperXRobot = lambda *args: robot
    monkeypatch.setitem(sys.modules, "taiyi_piper_x_collect.config", config_module)
    monkeypatch.setitem(sys.modules, "taiyi_piper_x_collect.devices.piper_x", devices_module)
    monkeypatch.setattr("piperx_expo.client.arm_hold.load_home", lambda config: pytest.fail("不能加载归位配置"))
    device.config["enable_recovery_timeout_s"] = 0.15
    session = ArmHold(device.config, time.monotonic, action=action)
    try:
        deadline = time.monotonic()+1
        while session.snapshot()["phase"] in {"enabling", "locking"} and time.monotonic()<deadline:
            time.sleep(0.005)
        time.sleep(0.12)
        assert session.snapshot()["phase"] == ("enable_sent" if action == "enable" else "holding")
        assert not session.snapshot()["error"]
        assert calls.count("read") == (1 if action == "hold" else 0)
        if action == "enable":
            time.sleep(0.15)
            count = sum(name == "enable" for name, _ in device.arm.commands)
            time.sleep(0.15)
            assert sum(name == "enable" for name, _ in device.arm.commands) == count
            assert "恢复未完成" in session.snapshot()["status_message"]
        session.close(release=True)
        assert "estop" not in calls
        if action == "enable":
            names = [name for name, _ in device.arm.commands]
            assert names.count("reset") == 1 and names.count("enable") >= 1
            assert not any(name in {"move_j", "move_p"} for name in names)
        else:
            assert device.arm.commands == []
    finally:
        session.close()


def test_cancel_hold_waiting_for_first_pose_never_sends_stop_or_target(monkeypatch):
    import sys
    import types
    from piperx_expo.client.arm_hold import ArmHold

    calls = []
    arm = SimpleNamespace(get_flange_pose=lambda: None,
                          move_p=lambda pose: calls.append("move_p"),
                          electronic_emergency_stop=lambda: calls.append("estop"))
    robot = SimpleNamespace(_arm=arm, start=lambda: None, stop=lambda: calls.append("disconnect"))
    config_module = types.ModuleType("taiyi_piper_x_collect.config")
    config_module.load_config = lambda path: SimpleNamespace(robot=object())
    devices_module = types.ModuleType("taiyi_piper_x_collect.devices.piper_x")
    devices_module.PiperXRobot = lambda *args: robot
    monkeypatch.setitem(sys.modules, "taiyi_piper_x_collect.config", config_module)
    monkeypatch.setitem(sys.modules, "taiyi_piper_x_collect.devices.piper_x", devices_module)
    config = load_config("configs/client/rotating_square.calibrated.yaml")
    session = ArmHold(config, time.monotonic, action="hold")
    session.close(release=True)
    assert session.snapshot()["phase"] == "released"
    assert calls == ["disconnect"]


@pytest.mark.parametrize("controller_code,ctrl_mode", [(1, 1), (0, 0)])
def test_driver_flags_alone_cannot_report_recovery_success(controller_code, ctrl_mode):
    device, home = recovery_device(controller_code)
    device.arm.get_arm_status = lambda: device.arm.packet(SimpleNamespace(
        arm_status=controller_code, err_code=0, ctrl_mode=ctrl_mode))
    status = _enable_confirmation(device.arm, device.config["feedback_timeout_s"])
    assert status["motors_enabled"]
    assert not status["controller_ready"] and not status["enable_confirmed"]


def test_fresh_first_driver_packets_with_zero_hz_can_confirm_recovery():
    device, home = recovery_device(0)
    original = device.arm.get_driver_states

    def first_packet(joint_index):
        packet = original(joint_index)
        packet.hz = 0
        return packet

    device.arm.get_driver_states = first_packet
    status = _enable_confirmation(device.arm, device.config["feedback_timeout_s"])
    assert status["motors_enabled"] and status["controller_ready"] and status["enable_confirmed"]


def test_enable_recovers_late_software_estop_once_without_position_commands():
    device, home = recovery_device(1)
    getter = device.arm.get_arm_status
    device.arm.get_arm_status = lambda: None
    reset_sent = restore_enable(device)
    assert not reset_sent
    device.arm.get_arm_status = getter
    reset_sent = restore_enable(device, reset_sent=reset_sent)
    assert reset_sent
    restore_enable(device, reset_sent=reset_sent)
    names = [name for name, _ in device.arm.commands]
    assert names.count("reset") == 1
    assert names.count("enable") == 3
    assert "move_j" not in names and "move_p" not in names


def test_enable_does_not_reset_hardware_fault():
    device, home = recovery_device(7)
    restore_enable(device)
    assert not any(name == "reset" for name, _ in device.arm.commands)
    status = _enable_confirmation(device.arm, device.config["feedback_timeout_s"])
    assert not status["controller_ready"] and "未自动清错" in status["status_message"]
