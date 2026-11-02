"""同盘可恢复删除；只接受调用方定义的直接子项名称，不接受任意路径。"""

import json
from pathlib import Path
import re
import shutil
import time
import uuid

from piperx_expo.client.recording import write_json


class RecycleBin:
    def __init__(self, root, name_pattern):
        self.root = Path(root).resolve()
        self.folder = self.root / ".trash"
        self.pattern = name_pattern

    def child(self, name, parent=None):
        if not isinstance(name, str) or not re.fullmatch(self.pattern, name):
            raise ValueError("删除对象名称无效，不能传入路径")
        path = (parent or self.root) / name
        if path.is_symlink():
            raise ValueError("不允许通过符号链接删除或恢复对象")
        return path

    def entry(self, trash_id):
        if not isinstance(trash_id, str) or not re.fullmatch(r"[0-9a-f]{32}", trash_id):
            raise ValueError("回收记录ID无效")
        path = self.folder / trash_id
        if (self.folder.is_symlink() or path.is_symlink() or (path / "payload").is_symlink()
                or (path / "receipt.json").is_symlink()):
            raise ValueError("回收目录不能是符号链接")
        return path

    def put(self, names, details=None):
        paths = [self.child(name) for name in names]
        if not paths or len(set(names)) != len(names) or any(not p.exists() for p in paths):
            raise ValueError("删除对象不存在或重复")
        trash_id = uuid.uuid4().hex
        entry = self.entry(trash_id)
        payload = entry / "payload"
        payload.mkdir(parents=True)
        record = {"id": trash_id, "deleted_at": time.time(), "names": list(names),
                  "details": details or {}, "status": "moving"}
        write_json(entry / "receipt.json", record)
        moved = []
        try:
            for path in paths:
                target = payload / path.name
                path.rename(target)
                moved.append((path, target))
            record["status"] = "ready"
            write_json(entry / "receipt.json", record)
        except BaseException:
            # checkpoint与配套review_state必须作为一组移动，失败时还原已移动部分。
            for source, target in reversed(moved):
                target.rename(source)
            record["status"] = "rolled_back"
            write_json(entry / "receipt.json", record)
            raise
        return record

    def records(self):
        if not self.folder.exists():
            return []
        if self.folder.is_symlink():
            raise ValueError("回收目录不能是符号链接")
        result = []
        for path in self.folder.glob("*/receipt.json"):
            entry = self.entry(path.parent.name)
            value = json.loads((entry / "receipt.json").read_text())
            if value["status"] == "ready":
                result.append(value)
        return sorted(result, key=lambda v: v["deleted_at"], reverse=True)

    def _ready(self, trash_id):
        entry = self.entry(trash_id)
        record = json.loads((entry / "receipt.json").read_text())
        if record.get("id") != trash_id or record.get("status") != "ready":
            raise ValueError("回收记录尚未完整提交或已经恢复")
        return entry, record

    def restore(self, trash_id):
        entry, record = self._ready(trash_id)
        targets = [self.child(name) for name in record["names"]]
        sources = [self.child(name, entry / "payload") for name in record["names"]]
        if any(p.exists() for p in targets):
            raise FileExistsError("原位置已有同名对象，禁止覆盖恢复")
        if any(not p.exists() for p in sources):
            raise ValueError("回收文件不完整，禁止部分恢复")
        moved = []
        try:
            for source, target in zip(sources, targets):
                source.rename(target)
                moved.append((source, target))
            record["status"] = "restored"
            write_json(entry / "receipt.json", record)
        except BaseException:
            for source, target in reversed(moved):
                target.rename(source)
            raise
        return record

    def purge(self, trash_id, confirmation):
        if confirmation != trash_id:
            raise ValueError("永久删除需要再次输入完整回收记录ID确认")
        entry, record = self._ready(trash_id)
        # 仅删除本应用已完整回收的payload；保留很小的确认记录供审计。
        # rmtree不会跟随内部符号链接，不能借回收项删除外部目录。
        shutil.rmtree(entry / "payload")
        record["status"] = "purged"
        record["purged_at"] = time.time()
        write_json(entry / "receipt.json", record)
        return record
