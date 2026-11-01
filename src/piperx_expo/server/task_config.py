"""上游配置加载入口。env_type=droid 仅用于选择兼容的回放读取器。"""

import os

import numpy as np
from ml_collections import ConfigDict

from piperx_expo.config import load_config


def get_config():
    experiment = load_config(os.environ["PIPERX_EXPERIMENT_CONFIG"])
    return ConfigDict(
        {
            "env_type": "droid",
            "env_name": "piperx_rotating_square",
            "language_instruction": experiment["language_instruction"],
            "action_space": "cartesian_position",
            "gripper_action_space": "position",
            "control_hz": experiment["control_hz"],
            "auto_reset_steps": experiment["episode_steps"],
            "example_action": np.zeros((1, 7), dtype=np.float32),
            # Dynamic Picking按论文只编辑XYZ/夹爪；旋转仍由基础VLA决定。
            "edit_action_xyzg": experiment.get("edit_action_xyzg", True),
            "critic_camera_keys": tuple(experiment.get("critic_camera_keys", ("base_0_rgb", "left_wrist_0_rgb"))),
        }
    )
