"""把已确认成功的 Piper-X 正向轨迹写成 EXPO 回放，或调用原 LeRobot 转换器。"""

import argparse
import json
import logging
import os
from pathlib import Path

import h5py
import cv2
import numpy as np

from piperx_expo.observation import ACTION_NAMES, CAMERAS, make_observation

LOG = logging.getLogger(__name__)


def discover(root: Path, batches=None):
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(root.rglob("trajectory.hdf5"))
    wanted = {f"batch_{n}" for n in batches} if batches else None
    paths = [
        p
        for p in paths
        if p.parent.name == "forward"
        and ".staging" not in p.parts
        and (wanted is None or wanted.intersection(p.parts))
    ]
    if not paths:
        raise ValueError(f"没有找到正式 forward/trajectory.hdf5：{root}")
    return paths


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def read_state(file, times, max_alignment_age_s=0.1):
    puppet = file["puppet"]
    names = ("end_effector_single_pose", "end_effector_single_position")
    # 原转换器把两列对齐反馈作为一组选择；不能混用一列align、一列raw。
    use_aligned = all(f"{name}_align" in puppet for name in names)
    if use_aligned:
        pose_group = puppet[f"{names[0]}_align"]
        use_aligned = len(pose_group["data"]) == len(times) and len(pose_group["timestamp"]) == len(times)
    components = []
    raw_times = None
    for name, width in zip(names, (6, 1)):
        group = puppet[f"{name}_{'align' if use_aligned else 'raw'}"]
        stamps, data = np.asarray(group["timestamp"]), np.asarray(group["data"])
        data = data.reshape(-1, width)
        if len(stamps) != len(data) or not np.isfinite(stamps).all():
            raise ValueError(f"{name} 时间戳或长度无效")
        if use_aligned:
            if len(stamps) != len(times) or np.max(np.abs(stamps - times)) > max_alignment_age_s:
                raise ValueError(f"{name} 对齐时间不一致；拒绝静默更换训练输入")
            values = data
        else:
            # 原分支使用pose的时间轴插值两列；不同时间轴时拒绝，不能猜测同步关系。
            if raw_times is not None and not np.array_equal(raw_times, stamps):
                raise ValueError("原始位姿与夹爪时间轴不同，不能按原OpenPI规则转换")
            raw_times = stamps
            stamps, indices = np.unique(stamps, return_index=True)
            data = data[indices]
            if len(stamps) < 2 or times[0] < stamps[0] or times[-1] > stamps[-1]:
                raise ValueError(f"{name} 原始反馈不能覆盖相机时间；禁止边界外推")
            nearest = np.searchsorted(stamps, times).clip(1, len(stamps) - 1)
            distance = np.minimum(abs(times - stamps[nearest]), abs(times - stamps[nearest - 1]))
            if np.max(distance) > max_alignment_age_s:
                raise ValueError(f"{name} 反馈间隙过大")
            # 与原分支一致：逐列线性插值，不unwrap、不将已有RPY折回[-pi,pi]。
            values = np.column_stack([np.interp(times, stamps, data[:, c]) for c in range(width)])
        if values.shape != (len(times), width):
            raise ValueError(f"{name} 维度错误：{values.shape}")
        components.append(values)
    state = np.column_stack(components).astype(np.float32)
    if not np.isfinite(state).all() or np.any(state[:, 6] < 0):
        raise ValueError("轨迹状态存在无效值")
    return state


def convert_episode(source, target, prompt, fps=30):
    with h5py.File(source, "r") as file:
        representation = _text(file["metadata/pose_representation"][()])
        if representation != "xyz_rxryrz":
            raise ValueError(f"{source} 是 {representation}；需米制 XYZ+RPY")
        actual_prompt = _text(file["metadata/language_instruction"][()]).strip()
        if actual_prompt != prompt.strip():
            raise ValueError(f"任务指令不一致：{actual_prompt!r}，配置为 {prompt!r}")
        times = np.asarray(file["camera_observations/timestamp"], dtype=float)
        n = int(file["metadata/trajectory_length"][()])
        if n < 2 or len(times) != n or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("轨迹长度或相机时间戳不合法")
        effective_fps = 1 / np.median(np.diff(times))
        if abs(effective_fps - fps) > fps * 0.1 or np.max(np.diff(times)) > 3 / fps:
            raise ValueError(f"轨迹频率/丢帧不满足 {fps}Hz：实测中位数 {effective_fps:.2f}Hz")
        state = read_state(file, times)
        actions = np.vstack([state[1:], state[-1:]])
        encoded = file["camera_observations/color_images"]
        if any(len(encoded[name]) != n for name in CAMERAS):
            raise ValueError("RGB 帧数不一致")
        target.mkdir(parents=True, exist_ok=False)
        temporary = target / "traj.hdf5.partial"
        with h5py.File(temporary, "w") as output:
            output.attrs.update(
                source=str(source.resolve()),
                action_mode="absolute_pose_width_m",
                fps=fps,
                confirmed_success=True,
            )
            obs = output.create_group("saved_observation")
            obs.create_dataset("cartesian_position", data=state[:, :6])
            obs.create_dataset("gripper_position", data=state[:, 6:7])
            obs.create_dataset("prompt", data=[prompt] * n, dtype=h5py.string_dtype())
            for key in ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"):
                obs.create_dataset(key, shape=(n, 224, 224, 3), dtype="u1", compression="lzf")
            for i in range(n):
                images = {}
                for name in CAMERAS:
                    # JPEG 解码也使用原转换器的 OpenCV 路径，避免解码器差异。
                    image = cv2.imdecode(np.asarray(encoded[name][i], dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("无法解码相机 JPEG 图像")
                    images[name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                observation = make_observation(state[i], images, prompt)
                for key in ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"):
                    obs[key][i] = observation[key]
            output.create_dataset("action/cartesian_position", data=actions[:, :6])
            output.create_dataset("action/gripper_position", data=actions[:, 6])
            output.create_dataset("timestamp", data=times - times[0])
        os.replace(temporary, target / "traj.hdf5")
    return {
        "source": str(source.resolve()),
        "episode": str(target),
        "frames": n,
        "effective_fps": float(effective_fps),
    }


def convert_replay(input_root, output_root, prompt, batches=None, fps=30):
    paths = discover(Path(input_root), batches)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"输出已存在：{output_root}；请指定新版本目录")
    output_root.mkdir(parents=True)
    records = []
    for index, source in enumerate(paths):
        LOG.info("转换 episode=%d source=%s", index, source)
        records.append(convert_episode(source, output_root / str(index), prompt, fps))
    manifest = {
        "action_names": ACTION_NAMES,
        "action_mode": "absolute_pose_width_m",
        "language_instruction": prompt,
        "fps": fps,
        "episodes": records,
        "confirmed_success": True,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=["expo", "lerobot"], default="expo")
    parser.add_argument("--batches", nargs="+", type=int)
    parser.add_argument("--prompt", default="grab the rotating red square")
    parser.add_argument("--confirm-successful-demos", action="store_true")
    args = parser.parse_args()
    if not args.confirm_successful_demos:
        parser.error("EXPO 会将示教末帧标为成功；只使用已核验成功示教并显式传 --confirm-successful-demos")
    logging.basicConfig(level=logging.INFO)
    if args.format == "expo":
        result = convert_replay(args.input, args.output, args.prompt, args.batches)
        print(json.dumps({"episodes": len(result["episodes"]), "output": str(args.output)}))
    else:
        from piperx_to_openpi_vla.convert import convert_dataset

        # 复用原转换器；先用 EXPO 的严格检查保证指令/动作/时间轴一致。
        for source in discover(args.input, args.batches):
            with h5py.File(source) as file:
                if _text(file["metadata/pose_representation"][()]) != "xyz_rxryrz":
                    raise ValueError("LeRobot 转换只接受 XYZ+RPY")
        print(
            convert_dataset(args.input, args.output, batches=args.batches, language_instruction=args.prompt)
        )


if __name__ == "__main__":
    main()
