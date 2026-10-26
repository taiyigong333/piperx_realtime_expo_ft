import io
import json

import h5py
import numpy as np
import pytest
from PIL import Image

from piperx_expo.data.convert import convert_replay, discover, read_state
from piperx_expo.data.norm import compute_norm
from piperx_expo.observation import CAMERAS


def make_raw(path, frames=4):
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as file:
        file["metadata/pose_representation"] = "xyz_rxryrz"
        file["metadata/language_instruction"] = "grab the rotating red square"
        file["metadata/trajectory_length"] = frames
        file["camera_observations/timestamp"] = np.arange(frames) / 30
        for name, values in (
            ("end_effector_single_pose", np.zeros((frames, 6))),
            ("end_effector_single_position", np.full((frames, 1), 0.04)),
        ):
            if values.shape[1] == 6:
                values[:, 0] = 0.3 + np.arange(frames) * 0.001
            file[f"puppet/{name}_align/data"] = values
            file[f"puppet/{name}_align/timestamp"] = np.arange(frames) / 30
        buffer = io.BytesIO()
        Image.new("RGB", (640, 480), (220, 30, 10)).save(buffer, format="JPEG")
        for name in CAMERAS:
            dataset = file.create_dataset(
                f"camera_observations/color_images/{name}",
                shape=(frames,),
                dtype=h5py.vlen_dtype(np.dtype("uint8")),
            )
            for i in range(frames):
                dataset[i] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def test_conversion_norm_and_no_overwrite(tmp_path):
    raw = tmp_path / "raw/batch_4/episode/forward/trajectory.hdf5"
    make_raw(raw)
    make_raw(tmp_path / "raw/batch_4/episode/reverse/trajectory.hdf5")
    assert discover(tmp_path / "raw", [4]) == [raw]
    output = tmp_path / "expo"
    result = convert_replay(tmp_path / "raw", output, "grab the rotating red square", [4])
    assert len(result["episodes"]) == 1
    with h5py.File(output / "0/traj.hdf5") as file:
        image = file["saved_observation/exterior_image_1_left"][0]
        assert image.shape == (224, 224, 3) and np.all(image[0] == 0)
        assert image[112, 112, 0] > image[112, 112, 2]  # 保持RGB，不能交换红蓝。
        assert file["action/cartesian_position"].shape == (4, 6)
        assert file["action/cartesian_position"][0, 0] == pytest.approx(0.301)
        assert file["action/gripper_position"][0] == pytest.approx(0.04)
    stats = compute_norm(output, tmp_path / "assets/piperx/rotating_square")
    assert len(stats["actions"]["mean"]) == 7
    assert np.all(np.subtract(stats["state"]["q99"], stats["state"]["q01"]) > 0)
    assert json.loads((output / "manifest.json").read_text())["confirmed_success"]
    with pytest.raises(FileExistsError):
        convert_replay(tmp_path / "raw", output, "grab the rotating red square")


def test_reject_different_task_and_quaternion(tmp_path):
    raw = tmp_path / "raw/forward/trajectory.hdf5"
    make_raw(raw)
    with pytest.raises(ValueError, match="指令"):
        convert_replay(tmp_path / "raw", tmp_path / "out", "wrong task")
    with h5py.File(raw, "a") as file:
        file["metadata/pose_representation"][()] = "xyz_xyzw"
    with pytest.raises(ValueError, match="xyz_xyzw"):
        convert_replay(tmp_path / "raw", tmp_path / "out2", "grab the rotating red square")


def test_raw_feedback_does_not_extrapolate(tmp_path):
    path = tmp_path / "raw.hdf5"
    with h5py.File(path, "w") as file:
        for name, width in (("end_effector_single_pose", 6), ("end_effector_single_position", 1)):
            file[f"puppet/{name}_raw/data"] = np.zeros((2, width))
            file[f"puppet/{name}_raw/timestamp"] = [1, 2]
        with pytest.raises(ValueError, match="外推"):
            read_state(file, np.array([0, 1, 2]))
