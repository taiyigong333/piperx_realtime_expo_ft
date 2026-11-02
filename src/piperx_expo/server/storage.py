"""GUI可管理的保存点清单与删除边界；路径仅由本机实验配置推导。"""

import hashlib
import json
from pathlib import Path
import re

import yaml

from piperx_expo.server.model_store import checkpoint_info
from piperx_expo.trash import RecycleBin


class CheckpointStorage:
    def __init__(self, manager):
        self.manager = manager

    def sources(self):
        sources, configs, records = {}, [], []
        for path in self.manager.runtime.glob("*/process.json"):
            record = json.loads(path.read_text())
            if not record.get("mock"):
                records.append((record, path.parent))
        for record, folder in sorted(records, key=lambda v: v[0].get("started_at", 0), reverse=True):
            config = yaml.safe_load((folder / "experiment.yaml").read_text())
            if record.get("mode", "train") == "eval" or config.get("inference_only"):
                continue
            configs.append(config)
            if config.get("episode_review") is True:
                output = Path(config["output_dir"])
                if not output.is_absolute():
                    output = self.manager.root / output
                sources[record["id"]] = {"root": output / f"{config['run_name']}_rl" / "checkpoints",
                                          "kind": "gui_rl", "label": config["run_name"]}
        default = self.manager.root / "configs/server/rotating_square.gui.yaml"
        if default.is_file():
            configs.append(yaml.safe_load(default.read_text()))
        references = []
        for config in configs:
            for key in ("base_weights", "rtc_weights"):
                if config.get(key):
                    path = Path(config[key])
                    references.append((path if path.is_absolute() else self.manager.root / path).resolve())
            if config.get("rtc_weights"):
                path = Path(config["rtc_weights"])
                path = (path if path.is_absolute() else self.manager.root / path).resolve()
                if path.name == "params" and path.parent.name.isdigit():
                    root = path.parent.parent
                    key = "rtc_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
                    sources[key] = {"root": root, "kind": "rtc_sft", "label": root.parent.name}
            if config.get("output_dir"):
                output = Path(config["output_dir"])
                output = output if output.is_absolute() else self.manager.root / output
                root = output.resolve().parent / "models/releases"
                key = "releases_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
                sources[key] = {"root": root, "kind": "release", "label": "命名模型版本"}
        return sources, references, configs, records

    def ensure_idle(self):
        from piperx_expo.server.gui_service import listening, owned

        _, _, configs, records = self.sources()
        if any(owned(record) for record, _ in records):
            raise RuntimeError("请先结束当前训练或推理，再管理checkpoint；不会自动停止任务")
        # 独立CLI也可能占用模型端口；不能仅根据GUI的active.json判断无人使用。
        if any(listening(port) for port in {c["port"] for c in configs if c.get("port")}):
            raise RuntimeError("模型端口仍在使用，请先结束服务再管理checkpoint")

    def source(self, source_id):
        sources, references, _, _ = self.sources()
        if source_id not in sources:
            raise ValueError("checkpoint来源不在已登记实验和配置范围内")
        source = sources[source_id]
        pattern = r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}" if source["kind"] == "release" else r"(?:0|[1-9][0-9]*|review_state_[0-9]+\.json)"
        return source, references, RecycleBin(source["root"], pattern)

    @staticmethod
    def protection(path, references):
        if any(ref == path.resolve() or ref.is_relative_to(path.resolve()) for ref in references):
            return "仍被SFT/RTC初始化配置引用；先显式切换配置，不能直接删除"
        return ""

    def catalog(self):
        sources, references, _, _ = self.sources()
        try:
            self.ensure_idle()
            blocked = ""
        except RuntimeError as error:
            blocked = str(error)
        rows, trash, seen = [], [], set()
        for source_id, source in sources.items():
            root = source["root"].resolve()
            if root in seen:
                continue
            seen.add(root)
            _, _, recycle = self.source(source_id)
            for record in recycle.records():
                trash.append({**record, "source_id": source_id, "source_label": source["label"],
                              "blocked_reason": blocked})
            for path in sorted(root.iterdir()) if root.is_dir() else []:
                if path.is_symlink() or not path.is_dir() or path.name.startswith("."):
                    continue
                if source["kind"] != "release" and not re.fullmatch(r"0|[1-9][0-9]*", path.name):
                    continue
                if source["kind"] == "release" and not (path / "model.json").is_file():
                    continue
                row = {"source_id": source_id, "item_id": path.name, "source_label": source["label"],
                       "kind": source["kind"], "path": str(path), "complete": False}
                try:
                    if source["kind"] == "release":
                        model = json.loads((path / "model.json").read_text())
                        row.update(step=model["step"], complete=True, model_kind=model["kind"])
                    else:
                        row.update(step=int(path.name), complete=True,
                                   review_state=checkpoint_info(root, int(path.name))["review_state"])
                except (OSError, ValueError, KeyError) as error:
                    row["error"] = str(error)
                    row["complete"] = False
                row["blocked_reason"] = blocked or self.protection(path, references)
                rows.append(row)
        return {"checkpoints": rows, "trash": sorted(trash, key=lambda r: r["deleted_at"], reverse=True),
                "blocked_reason": blocked}

    def delete(self, source_id, item_id, confirmed):
        self.ensure_idle()
        if confirmed is not True:
            raise ValueError("删除checkpoint前必须确认所选实验和保存步数")
        source, references, recycle = self.source(source_id)
        if source["kind"] != "release" and not re.fullmatch(r"0|[1-9][0-9]*", str(item_id)):
            raise ValueError("只能删除完整数字保存点，不能单独删除训练辅助状态")
        path = recycle.child(item_id)
        if not path.is_dir() or (source["kind"] == "release" and not (path / "model.json").is_file()):
            raise ValueError("保存点或命名版本目录不存在，不能删除其他文件")
        reason = self.protection(path, references)
        if reason:
            raise ValueError(reason)
        names = [item_id]
        if source["kind"] == "gui_rl" and (recycle.root / f"review_state_{item_id}.json").exists():
            names.append(f"review_state_{item_id}.json")
        return recycle.put(names, {"kind": source["kind"], "source_label": source["label"], "item_id": item_id})

    def restore(self, source_id, trash_id):
        self.ensure_idle()
        return self.source(source_id)[2].restore(trash_id)

    def purge(self, source_id, trash_id, confirmation):
        self.ensure_idle()
        _, references, recycle = self.source(source_id)
        _, record = recycle._ready(trash_id)
        if any(self.protection(recycle.child(name), references) for name in record["names"]):
            raise ValueError("该回收模型又被配置引用，请先恢复或切换配置，不能永久删除")
        return recycle.purge(trash_id, confirmation)
