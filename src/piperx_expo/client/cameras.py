"""独立的常驻双相机；只缓存最新RGB，不连接CAN或创建轨迹文件。"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from piperx_expo.config import resolve_path
from piperx_expo.observation import CAMERAS


class CameraStream:
    def __init__(self, config):
        from taiyi_piper_x_collect.config import load_config
        from taiyi_piper_x_collect.devices.realsense import RealSenseCamera

        self.config = config
        collection = load_config(resolve_path(config["collection_config"]))
        if {c.name for c in collection.enabled_cameras} != set(CAMERAS):
            raise ValueError("必须配置前视与腕部两路 RGB 相机")
        self.cameras = [RealSenseCamera(c) for c in collection.enabled_cameras]
        self.lock = threading.Lock()
        self.frames = {}
        self.failure = None
        self.closed = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=len(self.cameras))
        self.started = []
        try:
            for camera in self.cameras:
                camera.start(False)
                self.started.append(camera)
                self.pool.submit(self._worker, camera)
            deadline = time.monotonic() + 10
            while len(self.frames) < len(CAMERAS) and time.monotonic() < deadline and self.failure is None:
                time.sleep(0.02)
            self.read_images()
        except BaseException:
            self.close()
            raise

    def _worker(self, camera):
        try:
            while not self.closed.is_set():
                rgb = camera.read().color[..., ::-1].copy()  # 采集SDK输出BGR，模型使用RGB。
                with self.lock:
                    # 每路只保留一帧；空闲/复核期间不积累历史或写盘。
                    self.frames[camera._config.name] = (time.monotonic(), rgb)
        except Exception as error:
            self.failure = error

    def read_images(self):
        if self.closed.is_set():
            raise RuntimeError("相机已关闭")
        if self.failure is not None:
            raise RuntimeError("相机读取线程失败") from self.failure
        with self.lock:
            frames = dict(self.frames)
        if set(frames) != set(CAMERAS) or any(
            time.monotonic() - stamp > self.config["camera_timeout_s"] for stamp, _ in frames.values()
        ):
            raise RuntimeError("相机缺帧或过期，拒绝使用旧画面")
        return {name: frame for name, (_, frame) in frames.items()}

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.pool.shutdown(wait=True)
        for camera in self.started:
            camera.stop()
