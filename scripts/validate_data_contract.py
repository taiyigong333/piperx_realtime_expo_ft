"""与data-transfer固定OpenPI分支直接比较；仅合成数据，不连接机器人。"""

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np

from piperx_expo.data.convert import read_state
from piperx_expo.observation import resize_rgb


def verify_dataset(dataset, assets):
    import pandas as pd

    manifest = json.loads((dataset / "manifest.json").read_text())
    source = Path(manifest["source_dataset"])
    info = json.loads((source / "meta/info.json").read_text())
    episodes = manifest["episodes"]
    frames, video_frames = 0, 0
    for index, record in enumerate(episodes):
        path = Path(record["source"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["source_sha256"]
        table = pd.read_parquet(path)
        with h5py.File(Path(record["episode"]) / "traj.hdf5") as file:
            state = np.c_[file["saved_observation/cartesian_position"][:],
                          file["saved_observation/gripper_position"][:]]
            actions = np.c_[file["action/cartesian_position"][:], file["action/gripper_position"][:]]
            np.testing.assert_array_equal(state, np.stack(table["observation.state"]))
            np.testing.assert_array_equal(actions, np.stack(table["action"]))
            np.testing.assert_array_equal(file["timestamp"][:], table["timestamp"].to_numpy())
            assert len(state) == record["frames"]
            frames += len(state)
            # 全量数值与来源哈希；视频只抽首/中/末三条，避免宣称全量图像验收。
            if index in {0, len(episodes) // 2, len(episodes) - 1}:
                episode_index = int(table["episode_index"].iloc[0])
                for camera, key in (("camera_front", "exterior_image_1_left"),
                                    ("camera_wrist_right", "wrist_image_left")):
                    video = source / info["video_path"].format(
                        episode_chunk=episode_index // info["chunks_size"], episode_index=episode_index,
                        video_key=f"observation.images.{camera}")
                    capture = cv2.VideoCapture(str(video))
                    try:
                        for t in range(len(state)):
                            ok, frame = capture.read()
                            assert ok, str(video)
                            if t in {0, len(state) // 2, len(state) - 1}:
                                np.testing.assert_array_equal(file[f"saved_observation/{key}"][t],
                                                              cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                video_frames += 1
                    finally:
                        capture.release()
    assert len(episodes) == info["total_episodes"] and frames == info["total_frames"]
    result = {"episodes": len(episodes), "frames": frames, "sampled_video_frames": video_frames}
    if assets:
        provenance = json.loads((assets / "provenance.json").read_text())
        original = Path(provenance["checkpoint_norm_source"]).read_bytes()
        assert hashlib.sha256(original).hexdigest() == provenance["checkpoint_norm_sha256"]
        assert (assets / "vla_norm_stats.json").read_bytes() == original
        norm = json.loads((assets / "norm_stats.json").read_text())["norm_stats"]
        assert norm["state"] == json.loads(original)["norm_stats"]["state"]
        result["original_state_and_vla_norm"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=Path("third_party/converter/piperx_to_openpi_vla/convert.py"))
    parser.add_argument("--dataset", type=Path, help="可选：核对实际导入集与原LeRobot全量数值及抽样视频")
    parser.add_argument("--assets", type=Path, help="可选：核对原checkpoint归一化资产")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("reference_converter", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rng = np.random.default_rng(42)
    for height, width in ((480, 640), (335, 517), (224, 224), (80, 50)):
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        np.testing.assert_array_equal(resize_rgb(image), reference.resize_with_padding(image))
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        assert ok
        decoded = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(resize_rgb(decoded),
                                      reference.resize_with_padding(reference._decode_image(encoded)))
    with tempfile.TemporaryDirectory() as folder:
        for aligned in (True, False):
            with h5py.File(Path(folder) / f"{aligned}.hdf5", "w") as file:
                times = np.arange(4) / 30
                pose = rng.normal(size=(4, 6))
                pose[:, 3] = [3.1, 3.2, -3.1, -3.2]  # 明确覆盖旧适配层会折回/unwrap的数值。
                for name, values in (("end_effector_single_pose", pose),
                                     ("end_effector_single_position", np.full((4, 1), 0.04))):
                    group = file.create_group(f"puppet/{name}_{'align' if aligned else 'raw'}")
                    group["data"], group["timestamp"] = values, times
                query = times if aligned else (times[:-1] + times[1:]) / 2
                state = read_state(file, query)
                expected = reference._state_from_hdf5(file, query)
                np.testing.assert_array_equal(state, expected)
                np.testing.assert_array_equal(np.vstack([state[1:], state[-1:]]),
                                              np.vstack([expected[1:], expected[-1:]]))
    dataset_report = verify_dataset(args.dataset, args.assets) if args.dataset else None
    print(json.dumps({"result": "pass", "reference": str(args.reference), "dataset": dataset_report,
                      "image_cases": 4, "jpeg_cases": 4, "state_cases": 2,
                      "scope": "逐像素/逐数值对比；不验证视频压缩一致或真机同步"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
