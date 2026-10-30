import numpy as np
import pytest

from piperx_expo.client.environment import PiperXEnv
from piperx_expo.config import load_config


@pytest.fixture
def grasp_config():
    config = load_config("configs/client/mock.yaml")
    config.update(success_detector="piperx_expo.client.success:GraspSuccessDetector",
                  success_detector_config={"height_m": 0.28, "gripper_width_m": [0.034, 0.039],
                                           "consecutive_steps": 3}, episode_steps=30)
    config["mock_initial_state"] = [0.3, 0, 0.29, 0, 0, 0, 0.0366]
    return config


def test_success_counts_actions_not_queries_and_reset(grasp_config):
    env = PiperXEnv(grasp_config)
    try:
        env.reset()
        assert not env.get_info_for_step()[1]
        for i in range(3):
            env.step(grasp_config["mock_initial_state"])
            first = env.get_info_for_step()
            for _ in range(5):
                assert env.get_info_for_step() == first
            assert first[1] == (i == 2)
        assert first == (True, True, 1.0, 0.0)
        env.step([0] * 7)
        assert env.steps == 3
        env.reset()
        assert env.get_info_for_step() == (False, False, 0.0, 1.0)
    finally:
        env.close()


@pytest.mark.parametrize("height,width,command", [(0.27, 0.0366, 0.0366),
                                                   (0.29, 0.0, 0.0), (0.29, 0.07, 0.07),
                                                   (0.29, 0.0366, 0.07)])
def test_negative_grasp_states(grasp_config, height, width, command):
    env = PiperXEnv(grasp_config)
    try:
        env.reset()
        obs = env.get_observation()
        obs["cartesian_position"][2] = height
        obs["gripper_position"][0] = width
        for step in range(1, 10):
            env.steps = step
            env.last_action = np.array([0, 0, height, 0, 0, 0, command])
            assert not env.detector(env, obs)
    finally:
        env.close()


def test_real_boot_placeholder_never_sends_motion(grasp_config):
    env = PiperXEnv(grasp_config)
    try:
        env.reset()
        env.real = True
        env.boot_placeholder_pending = True
        before = env.device.read_state()
        np.testing.assert_array_equal(env.step(np.zeros(7)), before)
        assert env.steps == 0 and env.last_action is None and not env.armed
        with pytest.raises(ValueError):
            env.step(np.zeros(7))
    finally:
        env.close()
