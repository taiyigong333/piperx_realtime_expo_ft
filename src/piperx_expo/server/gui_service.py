"""GUI专用RL进程管理；Linux服务器执行，不创建机器人设备或接管其他进程。"""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import yaml

from piperx_expo.config import ROOT, load_config, resolve_path


def write_json(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def process_ticks(pid):
    # comm字段可能有空格；从最后一个右括号之后取starttime，不能直接split整行。
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def owned(record):
    try:
        pid = record["pid"]
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return (str(record["ticks"]) == process_ticks(pid)
                and record["id"].encode() in args and b"--worker" in args
                and os.getpgid(pid) == pid)
    except (KeyError, OSError, ProcessLookupError):
        return False


def listening(port):
    # 不主动连接WebSocket，以免被上游误认为唯一机器人客户端。
    for name in ("tcp", "tcp6"):
        for row in Path(f"/proc/net/{name}").read_text().splitlines()[1:]:
            columns = row.split()
            if columns[3] == "0A" and int(columns[1].rsplit(":", 1)[1], 16) == port:
                return True
    return False


class ServiceManager:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.runtime = self.root / "runtime/gui_server"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.active = self.runtime / "active.json"

    @contextmanager
    def lock(self):
        with (self.runtime / "manager.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def record(self):
        if not self.active.exists():
            return None, None
        job_id = json.loads(self.active.read_text())["id"]
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise ValueError("管理记录ID无效")
        path = self.runtime / job_id / "process.json"
        return path, json.loads(path.read_text())

    def status(self):
        path, record = self.record()
        if record is None:
            return {"phase": "idle", "running": False, "log_tail": ""}
        running = owned(record)
        stopping = (path.parent / "stop_requested").exists()
        if running:
            phase = "stopping" if stopping else "listening" if listening(record["port"]) else "loading"
        else:
            phase = "stopped" if stopping else "completed" if record.get("exit_code") == 0 else "failed"
        log = Path(record["log"])
        tail = ""
        if log.exists():
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 20000))
                tail = handle.read(20000).decode("utf-8", errors="replace")
        return {**record, "phase": phase, "running": running, "log_tail": tail}

    def start(self, config_path, *, mock=False):
        with self.lock():
            current = self.status()
            if current["running"]:
                if current.get("mock") != mock:
                    raise RuntimeError("已有另一运行模式的GUI服务，请先停止")
                return current
            source = resolve_path(config_path)
            config = load_config(str(source))
            if config.get("episode_review") is not True:
                raise ValueError("仅允许启动GUI专用复核RL配置")
            if listening(config["port"]):
                raise RuntimeError("模型端口已被其他服务占用，不接管、不终止")
            if not mock:
                from piperx_expo.server.launch import validate

                validate(config, "rl")
                gpu = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=10)
                if gpu.stdout.strip():
                    raise RuntimeError("GPU已有计算进程，请先结束既有训练/推理任务")
            job_id = uuid.uuid4().hex
            folder = self.runtime / job_id
            folder.mkdir()
            # 所有输出仍在原output_dir，只有run名独立，绝不resume/覆盖既有训练。
            config["run_name"] = f"{config['run_name']}_{time.strftime('%Y%m%d_%H%M%S')}_{job_id[:6]}"
            snapshot = folder / "experiment.yaml"
            snapshot.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            record = {"id": job_id, "run_name": config["run_name"], "port": config["port"],
                      "config": str(snapshot), "source_config": str(source), "mock": mock,
                      "log": str(folder / "server.log"), "started_at": time.time()}
            write_json(folder / "process.json", record)
            # 管理器是独立进程组的组长；即使SSH断开，也能继续记录退出码并管理整棵进程树。
            with Path(record["log"]).open("ab") as output:
                process = subprocess.Popen(
                    [sys.executable, "-m", "piperx_expo.server.gui_service", "--worker", job_id],
                    cwd=self.root, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    env={**os.environ, "PIPERX_EXPO_ROOT": str(self.root)},
                    start_new_session=True)
            record.update(pid=process.pid, ticks=process_ticks(process.pid))
            write_json(folder / "process.json", record)
            write_json(self.active, {"id": job_id})
            # worker等待pid记录完成才启动训练，避免父子进程覆盖退出状态的竞争。
            (folder / "ready").touch()
            return self.status()

    def stop(self):
        with self.lock():
            path, record = self.record()
            if record is None or not owned(record):
                return self.status()
            (path.parent / "stop_requested").touch()
            # 仅向经过PID/starttime/命令行/进程组验证的本管理器任务发信号。
            # 优先Ctrl-C；不承诺中断时额外保存最终checkpoint，日志/已ACK轨迹保留。
            for sig, seconds in ((signal.SIGINT, 12), (signal.SIGTERM, 5), (signal.SIGKILL, 1)):
                if not owned(record):
                    break
                try:
                    os.killpg(record["pid"], sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + seconds
                while owned(record) and time.monotonic() < deadline:
                    time.sleep(0.1)
            return self.status()

    def worker(self, job_id):
        folder = self.runtime / job_id
        # 组信号由子进程接收，监督进程保留到退出码写完；SIGKILL兜底时不会声称完整保存。
        signal.signal(signal.SIGINT, lambda *_: None)
        signal.signal(signal.SIGTERM, lambda *_: None)
        deadline = time.monotonic() + 10
        while not (folder / "ready").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("等待管理记录超时")
            time.sleep(0.01)
        path = folder / "process.json"
        record = json.loads(path.read_text())
        if record["mock"]:
            # 管理链路验收只开本机模拟监听，绝不导入模型或创建机器人设备。
            command = [sys.executable, "-u", "-c",
                       "import socket,time; s=socket.socket(); s.bind(('127.0.0.1',int(__import__('sys').argv[1]))); "
                       "s.listen(); print('MOCK GUI learner ready',flush=True); time.sleep(3600)",
                       str(record["port"])]
        else:
            command = ["bash", "scripts/server.sh", "rl", "--config", record["config"]]
        if (folder / "stop_requested").exists():
            record.update(exit_code=130, ended_at=time.time())
        else:
            process = subprocess.Popen(command, cwd=self.root)
            record.update(exit_code=process.wait(), ended_at=time.time())
        write_json(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("start", "stop", "status"))
    parser.add_argument("--config", default="configs/server/rotating_square.gui.yaml")
    parser.add_argument("--worker")
    parser.add_argument("--mock", action="store_true", help="仅命令行管理链路验收；不训练、不连接机器人")
    args = parser.parse_args()
    manager = ServiceManager()
    if args.worker:
        manager.worker(args.worker)
        return
    try:
        result = manager.start(args.config, mock=args.mock) if args.action == "start" else (
            manager.stop() if args.action == "stop" else manager.status())
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"phase": "error", "error": str(error)}, ensure_ascii=False))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
