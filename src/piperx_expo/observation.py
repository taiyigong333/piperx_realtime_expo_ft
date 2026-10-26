"""采集转换与在线观测共用的米制坐标、RGB 和 padding 约定。"""

import numpy as np
from PIL import Image

CAMERAS = ("camera_front", "camera_wrist_right")
IMAGE_KEYS = ("exterior_image_1_left", "wrist_image_left")
ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper_width_m")


def resize_rgb(image: np.ndarray, size: int = 224) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("图像必须为 HWC uint8 RGB")
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized = Image.fromarray(image).resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BOX
    )
    canvas = Image.new("RGB", (size, size))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.asarray(canvas).copy()


def make_observation(state: np.ndarray, images: dict, prompt: str) -> dict:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("状态必须是有限的7维绝对位姿与夹爪宽度")
    front, wrist = (resize_rgb(images[name]) for name in CAMERAS)
    # 第二外部视图仅为上游 repack 必需字段；模型第三路掩码为 False。
    return {
        "cartesian_position": state[:6],
        "gripper_position": state[6:7],
        "exterior_image_1_left": front,
        "exterior_image_2_left": front.copy(),
        "wrist_image_left": wrist,
        "prompt": prompt,
    }
