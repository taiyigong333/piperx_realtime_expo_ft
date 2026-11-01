"""机器人GUI通过既有SSH管理专属学习器；网络等待不占用控制/HTTP锁。"""

import copy
from pathlib import PurePosixPath
import queue
import re
import shlex
import subprocess
import threading
import json


class RemoteServer:
    def __init__(self, config):
        self.config = dict(config)
        host, project = config["ssh_host"], config["project_dir"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host):
            raise ValueError("远程SSH别名无效")
        if not PurePosixPath(project).is_absolute():
            raise ValueError("远程项目目录必须为绝对路径")
        self.lock = threading.Lock()
        self.actions = queue.Queue()
        self.pending = set()
        self.action_error = ""
        self.closed = threading.Event()
        self.value = {"phase": "unknown", "running": False, "log_tail": ""}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return {"enabled": True, **copy.deepcopy(self.value), "busy": bool(self.pending),
                    "stopping": "stop" in self.pending,
                    "ssh_host": self.config["ssh_host"]}

    def request(self, action):
        if action not in {"start", "stop", "status"}:
            raise ValueError("未知服务器操作")
        with self.lock:
            if action in self.pending:
                return
            self.pending.add(action)
        self.actions.put(action)

    def call(self, action):
        # HTTP不能覆盖这些配置或传入远端shell命令；路径统一shell引用，SSH禁用交互密码等待。
        command = (f"cd {shlex.quote(self.config['project_dir'])} && "
                   '"$HOME/.local/bin/uv" run --locked python -m piperx_expo.server.gui_service '
                   f"{action} --config {shlex.quote(self.config['config'])}")
        result = subprocess.run(
            ["ssh", "-oBatchMode=yes", "-oConnectTimeout=5", "-oServerAliveInterval=5",
             "-oServerAliveCountMax=2", self.config["ssh_host"], command],
            capture_output=True, text=True, timeout=40)
        if not result.stdout.strip():
            raise RuntimeError(result.stderr.strip()[-2000:] or "SSH未返回管理状态")
        value = json.loads(result.stdout)
        if result.returncode:
            raise RuntimeError(value.get("error", result.stderr.strip()[-2000:]))
        return value

    def _run(self):
        while not self.closed.is_set():
            try:
                action = self.actions.get(timeout=2)
            except queue.Empty:
                action = "status"
            try:
                value = self.call(action)
                with self.lock:
                    if action != "status":
                        self.action_error = ""
                    self.value = {**value, **({"error": self.action_error} if self.action_error else {})}
            except Exception as error:
                with self.lock:
                    if action != "status":
                        self.action_error = str(error)
                    # 管理SSH失败时显示未知状态，不能沿用旧的“已停止/已启动”结论。
                    self.value = {**self.value, "phase": "unknown", "error": str(error)}
            finally:
                with self.lock:
                    self.pending.discard(action)

    def shutdown(self):
        self.closed.set()
        self.thread.join(timeout=41)
        # GUI进程退出时同步尝试停止专属服务；不依赖即将退出的daemon线程。
        self.call("stop")
