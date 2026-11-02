"""GUI训练阶段的轻量状态文件；不依赖JAX、模型或机器人。"""

import json
import logging
import os
import time
from pathlib import Path


class TrainingProgress:
    def __init__(self, path=None):
        self.path = path or os.environ.get("PIPERX_GUI_STATUS_PATH")
        self.value = {}

    def update(self, stage, **details):
        self.value.update(stage=stage, updated_at=time.time(), **details)
        logging.info("GUI阶段=%s %s", stage, details)
        if self.path:
            path = Path(self.path)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(self.value, ensure_ascii=False, allow_nan=False))
            temporary.replace(path)
