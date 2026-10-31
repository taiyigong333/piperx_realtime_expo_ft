"""逐条轨迹异步落盘；T个执行目标对应T+1个实测观测，禁止丢帧后继续训练。"""

import copy
import json
import queue
import threading
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
from piperx_expo.observation import IMAGE_KEYS
from piperx_expo.review import EpisodeDigest, observation_state


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class EpisodeRecorder:
    def __init__(self, root, observation, settings):
        self.lock = threading.RLock()
        self.id = uuid.uuid4().hex
        self.folder = Path(root) / self.id
        self.folder.mkdir(parents=True)
        self.meta = {
            "id": self.id,
            "created_at": time.time(),
            "status": "recording",
            "steps": 0,
            "settings": settings,
            "schema": "piperx_review_v1",
        }
        self.digest = EpisodeDigest()
        self.digest.observation(observation)
        self.points = [observation_state(observation).tolist()]
        self.queue = queue.Queue(maxsize=64)
        self.error = None
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()
        self._put((None, observation, None, time.monotonic()))
        write_json(self.folder / "metadata.json", self.meta)

    def _put(self, item):
        if self.error:
            raise RuntimeError("轨迹落盘失败") from self.error
        try:
            self.queue.put_nowait(item)
        except queue.Full as error:
            raise RuntimeError("轨迹写入跟不上控制速率，停止采集以免丢帧") from error

    def append(self, action, obs, info):
        # SDK与模型可能复用数组，入队前复制以固定本次传输内容。
        obs = {
            key: np.array(obs[key], copy=True)
            for key in (*IMAGE_KEYS, "cartesian_position", "gripper_position")
        }
        action = np.asarray(action, dtype=np.float32).copy()
        self._put((action, obs, info, time.monotonic()))
        self.digest.transition(action, obs)
        with self.lock:
            self.points.append(observation_state(obs).tolist())
            self.meta["steps"] += 1

    def _writer(self):
        try:
            with h5py.File(self.folder / "episode.partial.hdf5", "w") as file:
                file.attrs["schema"] = "piperx_review_v1"

                def dataset(name, shape, dtype):
                    return file.create_dataset(
                        name, shape=(0, *shape), maxshape=(None, *shape), chunks=(1, *shape), dtype=dtype
                    )

                states = dataset("observations/state", (7,), "f4")
                images = [dataset(f"observations/{key}", (224, 224, 3), "u1") for key in IMAGE_KEYS]
                actions = dataset("actions", (7,), "f4")
                infos = dataset("auto_reward_mask_done_success", (4,), "f4")
                stamps = dataset("observation_monotonic_s", (), "f8")
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    action, obs, info, acquired = item
                    for ds, value in [
                        (states, observation_state(obs)),
                        (stamps, acquired),
                        *zip(images, (obs[key] for key in IMAGE_KEYS)),
                    ]:
                        ds.resize(ds.shape[0] + 1, axis=0)
                        ds[-1] = value
                    if action is not None:
                        for ds, value in ((actions, action), (infos, info)):
                            ds.resize(ds.shape[0] + 1, axis=0)
                            ds[-1] = value
                file.flush()
        except BaseException as error:
            self.error = error

    def finish(self, status="review_pending", **extra):
        # 控制已停止才等待IO；控制期间只做有界的非阻塞入队。
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=30)
        if self.thread.is_alive() or self.error:
            raise RuntimeError("轨迹归档未完成") from self.error
        (self.folder / "episode.partial.hdf5").replace(self.folder / "episode.hdf5")
        self.update(status=status, digest=self.digest.hexdigest(), points=self.points, **extra)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy({**self.meta, "points": self.points})

    def update(self, **extra):
        with self.lock:
            self.meta.update(extra)
            write_json(self.folder / "metadata.json", self.meta)
