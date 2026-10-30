"""已训练 Piper-X checkpoint 专用张量布局；replay 仍保存米制绝对动作。"""

import dataclasses

from openpi import transforms
from openpi.policies.droid_policy import DroidInputs, DroidOutputs
from openpi.training import config


@dataclasses.dataclass(frozen=True)
class RightWristInputs(DroidInputs):
    def __call__(self, data):
        result = super().__call__(data)
        # 原checkpoint把真实腕部送入右腕槽位；RPC的历史字段名不代表相机左右。
        for key in ("image", "image_mask"):
            result[key]["left_wrist_0_rgb"], result[key]["right_wrist_0_rgb"] = (
                result[key]["right_wrist_0_rgb"], result[key]["left_wrist_0_rgb"],
            )
        return result


@dataclasses.dataclass(frozen=True)
class CheckpointReplayData(config.LeRobotDROIDDataConfig):
    def create(self, assets_dirs, model_config):
        data = super().create(assets_dirs, model_config)
        return dataclasses.replace(data, data_transforms=transforms.Group(
            inputs=[RightWristInputs(model_type=model_config.model_type)],
            outputs=[DroidOutputs(action_dim=7)],
        ))


def register_checkpoint(experiment):
    from piperx_expo.server.openpi_config import CONFIG_NAME

    base = config.get_config("expo_pi05_droid_lora_finetune_sft_cartesian_state")
    model = dataclasses.replace(base.model, action_horizon=experiment["action_horizon"])
    data = CheckpointReplayData(repo_id=experiment["asset_id"], use_cartesian_state=True,
                                output_action_dim=7, base_config=config.DataConfig(prompt_from_task=True))
    result = dataclasses.replace(base, name=CONFIG_NAME, model=model, data=data,
                                 freeze_filter=model.get_freeze_filter())
    config._CONFIGS_DICT[CONFIG_NAME] = result
    return result
