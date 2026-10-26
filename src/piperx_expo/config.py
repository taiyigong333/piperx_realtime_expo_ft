"""统一解析相对项目根路径，避免切换目录后配置指向其他数据。"""

import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("PIPERX_EXPO_ROOT", Path(__file__).resolve().parents[2])).resolve()


def resolve_path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path: str) -> dict:
    with resolve_path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("配置必须是 YAML 对象")
    return config
