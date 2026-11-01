import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from piperx_expo.client.homing import fixed_joint_home, load_home
from piperx_expo.config import load_config


class FakeArm:
    ARM_STATUS = SimpleNamespace(ModeFeedback=SimpleNamespace(MOVE_J=1, MOVE_P=2),
                                 CtrlMode=SimpleNamespace(CAN_CTRL=1))

    def __init__(self, target):
        self.joints = target + 0.1
        self.mode = "p"
        self.commands = []
        self.follow = True
        self.stale = False

    def packet(self, msg):
        return SimpleNamespace(msg=msg, timestamp=time.time() - int(self.stale), hz=100)

    def get_joint_angles(self):
        return self.packet(self.joints)

    def get_flange_pose(self):
        return self.packet([0.3, 0, 0.25, 0, 0, 0])

    def fk(self, target):
        return [0.3, 0, 0.25, 0, 0, 0]

    def get_arm_status(self):
        return self.packet(SimpleNamespace(arm_status=0, err_code=0, ctrl_mode=1,
                                          mode_feedback=1 if self.mode == "j" else 2))

    def get_joints_enable_status_list(self):
        return [True] * 6

    def enable(self):
        return True

    def set_speed_percent(self, speed):
        self.commands.append(("speed", speed))

    def set_motion_mode(self, mode):
        self.mode = mode
        self.commands.append(("mode", mode))

    def move_j(self, target):
        self.commands.append(("move_j", target))
        if self.follow:
            self.joints = np.asarray(target)


def device_and_home():
    config = load_config("configs/client/rotating_square.calibrated.yaml")
    home = load_home(config)
    device = SimpleNamespace(config=config, arm=FakeArm(np.array(home["joint_positions_rad"])),
                             reset_cancel=threading.Event(), observe=lambda: None)
    return device, home


def test_each_reset_uses_same_target_and_restores_policy_mode():
    device, home = device_and_home()
    for offset in (0.1, -0.1):
        device.arm.joints = np.asarray(home["joint_positions_rad"]) + offset
        fixed_joint_home(device, home)
    moves = [target for cmd, target in device.arm.commands if cmd == "move_j"]
    assert moves == [home["joint_positions_rad"]] * 2
    assert device.arm.commands[-1] == ("mode", "p")


@pytest.mark.parametrize("reason", ["stale", "distance", "cancel"])
def test_unsafe_home_never_sends_target(reason):
    device, home = device_and_home()
    if reason == "stale":
        device.arm.stale = True
    elif reason == "distance":
        home["max_joint_delta_rad"] = 0.01
    else:
        device.reset_cancel.set()
    with pytest.raises((RuntimeError, ValueError, InterruptedError)):
        fixed_joint_home(device, home)
    assert not device.arm.commands


def test_home_timeout_does_not_start_policy():
    device, home = device_and_home()
    device.arm.follow = False
    home.update(timeout_s=0.01, poll_s=0.001)
    with pytest.raises(TimeoutError, match="归位超时"):
        fixed_joint_home(device, home)
    assert device.arm.mode == "j"


def test_stop_interrupts_home_wait():
    device, home = device_and_home()
    device.arm.follow = False
    timer = threading.Timer(0.03, device.reset_cancel.set)
    timer.start()
    try:
        with pytest.raises(InterruptedError):
            fixed_joint_home(device, home)
        assert device.arm.mode == "j"
    finally:
        timer.join()


def test_environment_close_interrupts_reset_and_fault_latches():
    from piperx_expo.client.environment import PiperXEnv

    entered = threading.Event()

    class WaitingDevice:
        stopped = False

        def reset(self, confirmed=False):
            entered.set()
            assert self.reset_cancel.wait(1), "停止事件未传递到归位"
            raise InterruptedError("归位已停止")

        def stop_hold(self):
            self.stopped = True

        def close(self):
            self.stop_hold()

    device = WaitingDevice()
    env = PiperXEnv(load_config("configs/client/rotating_square.calibrated.yaml"),
                    allow_motion=True, device=device, reset_confirm=lambda: None)
    errors = []

    def reset():
        try:
            env.reset()
        except InterruptedError as error:
            errors.append(error)

    worker = threading.Thread(target=reset)
    worker.start()
    try:
        assert entered.wait(1)
        env.close()
        worker.join(1)
        assert not worker.is_alive() and errors and device.stopped and env.fault
    finally:
        env.close()
        worker.join(2)


def test_workspace_rejection_reports_measurement_and_never_enables():
    from piperx_expo.client.homing import check_home_start

    device, home = device_and_home()
    device.arm.get_flange_pose = lambda: device.arm.packet([-0.11, 0, -0.11, 0, 0, 0])
    with pytest.raises(ValueError, match=r"X=-0.110000m.*Z=-0.110000m.*未使能"):
        check_home_start(device, home)
    with pytest.raises(ValueError, match="起点超出"):
        fixed_joint_home(device, home)
    assert not device.arm.commands


def test_relaxed_home_start_keeps_policy_workspace_and_target_unchanged():
    device, home = device_and_home()
    assert device.config["workspace_m"][0][0] == -0.02
    assert device.config["workspace_m"][2][0] == 0
    device.arm.get_flange_pose = lambda: device.arm.packet([-0.08, 0, -0.01, 0, 0, 0])
    fixed_joint_home(device, home)
    moves = [target for cmd, target in device.arm.commands if cmd == "move_j"]
    assert moves == [home["joint_positions_rad"]]


def test_measured_joint_start_within_relaxed_tolerance_but_excess_still_rejected():
    device, home = device_and_home()
    device.arm.joints = np.asarray(home["joint_positions_rad"]).copy()
    device.arm.joints[1] = -0.04441863
    fixed_joint_home(device, home)
    device.arm.commands.clear()
    device.arm.joints[1] = -0.061
    with pytest.raises(ValueError, match="起点关节超限"):
        fixed_joint_home(device, home)
    assert not device.arm.commands
