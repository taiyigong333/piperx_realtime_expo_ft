"""导入已人工确认成功的 LeRobot v2.1 示教，保留已训练 checkpoint 的观测约定。"""

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np

from piperx_expo.data.norm import compute_norm

LOG = logging.getLogger(__name__)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_dataset(source, output, checkpoint_norm, assets, prompt):
    import cv2
    import pandas as pd

    source, output, checkpoint_norm, assets = map(Path, (source, output, checkpoint_norm, assets))
    if output.exists() or assets.exists():
        raise FileExistsError("回放/资产输出已存在，请使用新版本路径")
    info = json.loads((source / "meta/info.json").read_text())
    tasks = [json.loads(line) for line in (source / "meta/tasks.jsonl").read_text().splitlines()]
    if info["fps"] != 30 or len(tasks) != 1 or tasks[0]["task"] != prompt:
        raise ValueError("只接受同一30Hz任务的示教")
    # 先核验源资产，转换失败时不会留下看似完整的 manifest。
    original_norm = json.loads(checkpoint_norm.read_text())
    if not {"state", "actions"} <= original_norm["norm_stats"].keys():
        raise ValueError("checkpoint 缺少 state/actions 归一化")
    episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines()]
    output.mkdir(parents=True)
    records = []
    for ep in episodes:
        index, n = ep["episode_index"], ep["length"]
        chunk = index // info["chunks_size"]
        parquet = source / info["data_path"].format(episode_chunk=chunk, episode_index=index)
        table = pd.read_parquet(parquet)
        state = np.stack(table["observation.state"]).astype(np.float32)
        action = np.stack(table["action"]).astype(np.float32)
        times = table["timestamp"].to_numpy()
        if n < 2 or state.shape != (n, 7) or action.shape != state.shape:
            raise ValueError(f"episode {index} 维度不符")
        if not np.isfinite(np.r_[state, action]).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"episode {index} 非有限数据/时间倒退")
        if abs(1 / np.median(np.diff(times)) - 30) > 3:
            raise ValueError(f"episode {index} 频率不是30Hz")
        folder = output / str(index)
        folder.mkdir()
        with h5py.File(folder / "traj.hdf5.partial", "w") as file:
            file.attrs.update(action_mode="absolute_pose_width_m", confirmed_success=True, fps=30)
            file["saved_observation/cartesian_position"] = state[:, :6]
            file["saved_observation/gripper_position"] = state[:, 6:7]
            file.create_dataset("saved_observation/prompt", data=[prompt] * n, dtype=h5py.string_dtype())
            file["action/cartesian_position"] = action[:, :6]
            file["action/gripper_position"] = action[:, 6]
            file["timestamp"] = times
            for camera, key in (("camera_front", "exterior_image_1_left"),
                                ("camera_wrist_right", "wrist_image_left")):
                video = source / info["video_path"].format(
                    episode_chunk=chunk, episode_index=index,
                    video_key=f"observation.images.{camera}",
                )
                dataset = file.create_dataset(f"saved_observation/{key}", (n, 224, 224, 3),
                                              dtype="u1", compression="lzf")
                capture = cv2.VideoCapture(str(video))
                try:
                    for t in range(n):
                        ok, frame = capture.read()
                        if not ok or frame.shape != (224, 224, 3):
                            raise ValueError(f"视频帧数/形状不符：{video}, {t}")
                        dataset[t] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if capture.read()[0]:
                        raise ValueError(f"视频多于元数据帧数：{video}")
                finally:
                    capture.release()
            # 上游 reader 需要此字段；硬链接避免重复保存前视图。
            file["saved_observation/exterior_image_2_left"] = file["saved_observation/exterior_image_1_left"]
        (folder / "traj.hdf5.partial").rename(folder / "traj.hdf5")
        records.append({"source": str(parquet), "source_sha256": sha256(parquet),
                        "episode": str(folder), "frames": n})
        LOG.info("已转换 %d/%d episode=%d frames=%d", len(records), len(episodes), index, n)
    if len(records) != info["total_episodes"] or sum(r["frames"] for r in records) != info["total_frames"]:
        raise ValueError("源元数据的总轨迹数/总帧数不符")
    manifest = {"action_mode": "absolute_pose_width_m", "language_instruction": prompt, "fps": 30,
                "confirmed_success": True, "success_confirmation": "用户明确确认全部示教成功",
                "source_format": "lerobot_v2.1", "source_dataset": str(source), "episodes": records}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    compute_norm(output, assets)
    norm = json.loads((assets / "norm_stats.json").read_text())
    # critic/replay 使用绝对动作；VLA 的 state 统计和 delta 动作统计保留原 checkpoint。
    norm["norm_stats"]["state"] = original_norm["norm_stats"]["state"]
    (assets / "norm_stats.json").write_text(json.dumps(norm, indent=2))
    shutil.copyfile(checkpoint_norm, assets / "vla_norm_stats.json")
    provenance = json.loads((assets / "provenance.json").read_text())
    provenance.update(checkpoint_norm_source=str(checkpoint_norm), checkpoint_norm_sha256=sha256(checkpoint_norm),
                      action_adapter="absolute_replay_delta_vla_v1", wrist_slot="right_wrist_0_rgb")
    (assets / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    return {"episodes": len(records), "frames": info["total_frames"], "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "checkpoint-norm", "assets"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--prompt", default="grab the rotating red square")
    parser.add_argument("--confirm-successful-demos", action="store_true")
    args = parser.parse_args()
    if not args.confirm_successful_demos:
        parser.error("必须先人工确认全部示教成功，并传 --confirm-successful-demos")
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(import_dataset(args.source, args.output, args.checkpoint_norm, args.assets, args.prompt)))


if __name__ == "__main__":
    main()
