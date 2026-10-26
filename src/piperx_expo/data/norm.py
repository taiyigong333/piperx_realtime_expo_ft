"""从实际 RTC 回放计算 OpenPI 归一化，避免混用 DROID 速度统计。"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def compute_norm(dataset: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"归一化资产已存在，请新建版本目录：{output}")
    states, actions = [], []
    paths = sorted(dataset.glob("[0-9]*/traj.hdf5"))
    if not paths or not (dataset / "manifest.json").is_file():
        raise ValueError("需要由 piperx-convert 产生的完整 EXPO 回放")
    for path in paths:
        with h5py.File(path) as file:
            states.append(
                np.column_stack(
                    [
                        file["saved_observation/cartesian_position"][:],
                        file["saved_observation/gripper_position"][:],
                    ]
                )
            )
            actions.append(
                np.column_stack([file["action/cartesian_position"][:], file["action/gripper_position"][:]])
            )
    result = {}
    for key, values in (("state", states), ("actions", actions)):
        matrix = np.concatenate(values)
        if matrix.shape[1] != 7 or not np.isfinite(matrix).all():
            raise ValueError("归一化输入必须为有限7维数据")
        q01, q99 = np.quantile(matrix, [0.01, 0.99], axis=0)
        # 固定维度给出非零分母；量纲仍保留为米/弧度，不人为设成 DROID 速度。
        constant = q99 - q01 < 1e-5
        q01[constant] = matrix.mean(axis=0)[constant] - 0.5
        q99[constant] = matrix.mean(axis=0)[constant] + 0.5
        result[key] = {
            "mean": matrix.mean(axis=0).tolist(),
            "std": np.maximum(matrix.std(axis=0), 1e-6).tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }
    output.mkdir(parents=True)
    (output / "norm_stats.json").write_text(json.dumps({"norm_stats": result}, indent=2))
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset.resolve()),
                "episodes": len(paths),
                "manifest_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
                "constant_dimension_policy": "q01/q99=center+-0.5 when span<1e-5",
            },
            indent=2,
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="assets_dir/asset_id")
    args = parser.parse_args()
    compute_norm(args.dataset, args.output)
    print(args.output / "norm_stats.json")


if __name__ == "__main__":
    main()
