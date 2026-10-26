"""同步单卡 Real-Time EXPO-FT；RTC-SFT 与在线阶段共享动作和资产配置。"""

import os

from piperx_expo.config import load_config, resolve_path
from piperx_expo.server.openpi_config import CONFIG_NAME


def get_config():
    stage = os.environ["PIPERX_EXPO_STAGE"]
    if stage == "sft":
        from configs.model.rtc_pi_config import get_config as upstream
    else:
        from configs.model.realtime_expo_ft_pi_config import get_config as upstream
    config = upstream()
    experiment = load_config(os.environ["PIPERX_EXPERIMENT_CONFIG"])
    config.pi05_config_name = CONFIG_NAME
    config.pi05_weight_loader_path = str(
        resolve_path(experiment["base_weights" if stage == "sft" else "rtc_weights"])
    )
    config.pi05_assets_dir = str(resolve_path(experiment["assets_dir"]))
    config.pi05_asset_id = experiment["asset_id"]
    config.p1_max_delay = experiment["replan_steps"]
    config.valids_keep_terminal_windows = True
    if stage != "sft":
        config.N = 8
        config.n_edit_samples = 8
        config.filter_N = 8
        config.filter_n_edit = 1
        config.edit_scale = 0.1
    return config
