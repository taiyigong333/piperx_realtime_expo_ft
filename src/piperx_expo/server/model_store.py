"""保存点核验与命名模型归档；默认只读查看，登记使用同盘硬链接保留完整权重。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import yaml

from piperx_expo.config import load_config, resolve_path


def checkpoint_info(directory, step=None):
    root = Path(directory).resolve()
    steps = sorted(int(p.name) for p in root.glob("*") if p.is_dir() and p.name.isdigit())
    if not steps:
        raise ValueError("尚无数字命名的已保存模型")
    if step is None:
        step = steps[-1]
    if type(step) is not int or step < 0 or step not in steps:
        raise ValueError("所选模型保存步数不存在或无效")
    folder = root / str(step)
    if not (folder / "_CHECKPOINT_METADATA").is_file():
        raise ValueError("checkpoint缺少Orbax提交元数据，不能使用未完成保存点")
    for item in ("agent", "params"):
        if not (folder / item).is_dir() or not (folder / item / "_METADATA").is_file():
            raise ValueError(f"完整模型缺少{item}或其元数据；不能仅用VLA参数代替RL策略")
    state_path = root / f"review_state_{step}.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else None
    return {"checkpoint_dir": str(root), "step": step, "latest_step": steps[-1], "review_state": state}


def readonly_checkpoint_manager(directory):
    # 延迟导入，使目录查看、模型登记和GUI元数据检查不需要JAX/GPU依赖。
    import orbax.checkpoint as ocp

    return ocp.CheckpointManager(str(Path(directory).resolve()), item_handlers={
        "agent": ocp.PyTreeCheckpointHandler(), "params": ocp.PyTreeCheckpointHandler(),
    }, options=ocp.CheckpointManagerOptions(read_only=True, create=False))


def inference_config(config, directory, step, *, output_dir=None, run_name=None, max_steps=None):
    info = checkpoint_info(directory, step)
    value = dict(config)
    value.update(episode_review=True, inference_only=True,
                 inference_checkpoint_dir=info["checkpoint_dir"], inference_checkpoint_step=info["step"],
                 output_dir=str(output_dir or (resolve_path(config["output_dir"]).parent / "evaluations")),
                 run_name=run_name or config["run_name"], checkpoint_model=False)
    if max_steps is not None:
        if type(max_steps) is not int or not 1 <= max_steps <= 10_000_000:
            raise ValueError("推理最大动作步数必须为1到10000000整数")
        value["max_steps"] = max_steps
    return value


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register_model(config_path, directory, step, store, name, *, kind="rl"):
    if kind not in {"rl", "rtc_sft"}:
        raise ValueError("仅支持完整RL模型或RTC-SFT基线登记")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", name):
        raise ValueError("模型名只允许1到96位字母、数字、下划线或短横线")
    config = load_config(str(config_path))
    info = checkpoint_info(directory, step)
    source = Path(info["checkpoint_dir"]) / str(step)
    target = Path(store).resolve() / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("同名模型已登记，禁止覆盖；请使用新的版本名")
    staging = target.parent / f".pending_{name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        # 已提交Orbax分片不原地修改。同盘硬链接不复制大文件，原run清理也不会删除登记版本。
        # 跨盘时明确失败，不在用户不知情时复制几十GiB。
        shutil.copytree(source, staging / "checkpoints" / str(step), copy_function=os.link)
        assets = resolve_path(config["assets_dir"]) / config["asset_id"]
        shutil.copytree(assets, staging / "assets" / config["asset_id"])
        manifest = resolve_path(config["dataset_path"]) / "manifest.json"
        shutil.copy2(manifest, staging / "dataset_manifest.json")
        (staging / "source_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        pinned = dict(config, assets_dir=str(target / "assets"))
        if kind == "rl":
            usage = inference_config(pinned, directory, step, run_name=name)
            usage["inference_checkpoint_dir"] = str(target / "checkpoints")
            usage_file = "inference.yaml"
        else:
            # RTC-SFT只有VLA前缀训练，不冒充具有匹配critic/edit的完整RL策略。
            usage = dict(pinned, rtc_weights=str(target / "checkpoints" / str(step) / "params"))
            usage_file = "initialization.yaml"
        (staging / usage_file).write_text(yaml.safe_dump(usage, allow_unicode=True, sort_keys=False))
        record = {"schema": "piperx_model_v1", "name": name, "kind": kind,
                  "source_checkpoint_dir": info["checkpoint_dir"], "step": step,
                  "source_config": str(resolve_path(str(config_path))), "review_state": info["review_state"],
                  "dataset_manifest_sha256": _sha256(manifest),
                  "assets_sha256": {p.name: _sha256(p) for p in assets.glob("*.json")},
                  "checkpoint_metadata_sha256": _sha256(source / "_CHECKPOINT_METADATA"),
                  "component_metadata_sha256": {item: _sha256(source / item / "_METADATA")
                                                for item in ("agent", "params")},
                  "source_config_sha256": _sha256(staging / "source_config.yaml"),
                  "usage_config_file": usage_file, "usage_config_sha256": _sha256(staging / usage_file),
                  "storage": "same_filesystem_hardlinks", "validation": "structure_only",
                  "note": "未证明模型恢复、推理质量或真机成功率；不得原地编辑任一权重分片"}
        (staging / "model.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        verify_model(staging)
        staging.rename(target)
        return record
    except BaseException:
        # 只清理本次新建的临时目录，绝不删除来源权重或既有登记模型。
        shutil.rmtree(staging)
        raise


def verify_model(folder):
    folder = Path(folder).resolve()
    record = json.loads((folder / "model.json").read_text())
    if record.get("schema") != "piperx_model_v1" or record.get("kind") not in {"rl", "rtc_sft"}:
        raise ValueError("模型登记格式或类别无效")
    usage_file = "inference.yaml" if record["kind"] == "rl" else "initialization.yaml"
    for name, expected in (("source_config.yaml", record["source_config_sha256"]),
                           (usage_file, record["usage_config_sha256"])):
        if _sha256(folder / name) != expected:
            raise ValueError(f"模型配置哈希不符：{name}")
    config = load_config(str(folder / "source_config.yaml"))
    checkpoint_info(folder / "checkpoints", record["step"])
    if _sha256(folder / "checkpoints" / str(record["step"]) / "_CHECKPOINT_METADATA") != record["checkpoint_metadata_sha256"]:
        raise ValueError("模型提交元数据已变化")
    for item, expected in record["component_metadata_sha256"].items():
        if _sha256(folder / "checkpoints" / str(record["step"]) / item / "_METADATA") != expected:
            raise ValueError(f"模型组件元数据已变化：{item}")
    assets = folder / "assets" / config["asset_id"]
    for name, expected in record["assets_sha256"].items():
        if _sha256(assets / name) != expected:
            raise ValueError(f"模型归一化资产哈希不符：{name}")
    if _sha256(folder / "dataset_manifest.json") != record["dataset_manifest_sha256"]:
        raise ValueError("登记的数据manifest哈希不符")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("list", help="只读列出run下的完整/未完成保存点")
    scan.add_argument("--runs", required=True)
    register = commands.add_parser("register", help="保留固定版本及资产，不移动来源权重")
    register.add_argument("--config", required=True)
    register.add_argument("--checkpoint-dir", required=True)
    register.add_argument("--step", type=int, required=True)
    register.add_argument("--store", required=True)
    register.add_argument("--name", required=True)
    register.add_argument("--kind", choices=("rl", "rtc_sft"), default="rl")
    verify = commands.add_parser("verify", help="只读核验登记版本的结构与小型资产哈希")
    verify.add_argument("folder")
    args = parser.parse_args()
    if args.command == "register":
        result = register_model(args.config, args.checkpoint_dir, args.step, args.store, args.name, kind=args.kind)
    elif args.command == "verify":
        result = verify_model(args.folder)
    else:
        result = []
        for root in sorted(Path(args.runs).resolve().glob("*/checkpoints")):
            for path in sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
                if path.is_dir() and path.name.isdigit():
                    item = {"run": root.parent.name, "step": int(path.name)}
                    try:
                        item.update(checkpoint_info(root, int(path.name)), complete=True)
                    except (OSError, ValueError) as error:
                        item.update(complete=False, reason=str(error))
                    result.append(item)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
