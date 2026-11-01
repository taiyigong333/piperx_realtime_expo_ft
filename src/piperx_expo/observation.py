"""采集转换与在线观测共用的米制坐标、RGB 和 padding 约定。"""

import numpy as np
import cv2

CAMERAS = ("camera_front", "camera_wrist_right")
IMAGE_KEYS = ("exterior_image_1_left", "wrist_image_left")
ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper_width_m")


def resize_rgb(image: np.ndarray, size: int = 224) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("图像必须为 HWC uint8 RGB")
    height, width = image.shape[:2]
    scale = size / max(height, width)
    # 与 data-transfer 的 piperx_to_openpi_vla 分支逐像素一致，不能用 Pillow BOX 替代。
    nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = resized
    return canvas


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
