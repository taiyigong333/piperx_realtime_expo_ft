import threading
import time

import numpy as np
import pytest
from openpi_client import msgpack_numpy

from piperx_expo.client.environment import PiperXEnv
from piperx_expo.client.rpc import Dispatcher
from piperx_expo.config import load_config


@pytest.fixture
def config():
    return load_config("configs/client/mock.yaml")


@pytest.mark.parametrize(
    "bad",
    [
        [float("nan"), 0, 0.2, 0, 0, 0, 0.04],
        [1, 0, 0.2, 0, 0, 0, 0.04],
        [0.4, 0, 0.2, 0, 0, 0, 0.04],
        [0.3, 0, 0.2, 0, 0, 0, 0.2],
        [0, 0],
    ],
)
def test_reject_unsafe_and_stop(config, bad):
    env = PiperXEnv(config)
    try:
        env.reset()
        with pytest.raises(ValueError):
            env.step(bad)
        assert env.fault and env.device.stopped
        with pytest.raises(RuntimeError):
            env.step(config["mock_initial_state"])
    finally:
        env.close()


def test_watchdog_latches_stop(config):
    config["watchdog_s"] = 0.05
    env = PiperXEnv(config)
    try:
        env.reset()
        env.step(config["mock_initial_state"])
        deadline = time.monotonic() + 1
        while not env.fault and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        assert env.fault and env.device.stopped
        with pytest.raises(RuntimeError):
            env.reset()
    finally:
        env.close()


def test_real_config_requires_explicit_hardware_configuration():
    config = load_config("configs/client/piperx.example.yaml")
    with pytest.raises(ValueError, match="allow-motion"):
        PiperXEnv(config)
    with pytest.raises(ValueError, match="工作空间"):
        PiperXEnv(config, allow_motion=True)


def test_dispatch_and_serialization(config):
    rpc = Dispatcher(config)
    try:
        response = rpc.handle({"operation": "create_env"})
        env_id = response["env_id"]
        with pytest.raises(RuntimeError):
            rpc.handle({"operation": "step", "env_id": env_id, "action": config["mock_initial_state"]})
        response = rpc.handle({"operation": "reset", "env_id": env_id})
        decoded = msgpack_numpy.unpackb(msgpack_numpy.packb(response))
        assert decoded["observation"]["wrist_image_left"].dtype == np.uint8
        assert decoded["observation"]["cartesian_position"].shape == (6,)
        with pytest.raises(ValueError):
            rpc.handle({"operation": "reset", "env_id": "old-id"})
    finally:
        rpc.close()


def test_no_extra_motion_after_terminal(config):
    config["episode_steps"] = 1
    env = PiperXEnv(config)
    try:
        env.reset()
        action = np.array(config["mock_initial_state"])
        env.step(action)
        assert env.get_info_for_step()[0]
        action[0] += 0.001
        executed = env.step(action)
        assert executed[0] == pytest.approx(config["mock_initial_state"][0])
        assert env.steps == 1
    finally:
        env.close()


def test_reject_mismatched_task_before_devices(config):
    rpc = Dispatcher(config)
    with pytest.raises(ValueError, match="control_hz"):
        rpc.handle(
            {
                "operation": "create_env",
                "task_spec": {
                    "language_instruction": config["language_instruction"],
                    "control_hz": 10,
                    "episode_steps": 200,
                    "action_mode": "absolute_pose_width_m",
                },
            }
        )
    assert rpc.env is None


def test_nan_safety_limit_is_rejected(config):
    config["max_translation_step_m"] = float("nan")
    with pytest.raises(ValueError, match="max_translation"):
        PiperXEnv(config)
