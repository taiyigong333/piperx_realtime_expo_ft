"""注册 Piper-X 独立配置，复用上游纯数据适配，不加载 DROID 机器人 SDK。"""

import dataclasses


CONFIG_NAME = "piperx_pi05_lora_absolute_rtc"


def register():
    from openpi.training import config

    if CONFIG_NAME not in config._CONFIGS_DICT:
        base = config.get_config("expo_pi05_droid_lora_finetune_sft_cartesian_state")
        # 此数据变换只拼接位姿和夹爪、处理RGB、截取7维输出，没有速度/增量换算。
        # 因此可复用张量布局，但归一化资产和权重必须来自 Piper-X。
        piper = dataclasses.replace(
            base, name=CONFIG_NAME, data=dataclasses.replace(base.data, repo_id="piperx/rotating_square")
        )
        config._CONFIGS_DICT[CONFIG_NAME] = piper
    return config._CONFIGS_DICT[CONFIG_NAME]
