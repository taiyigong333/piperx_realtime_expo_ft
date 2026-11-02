"""GUI专用RL进程管理；Linux服务器执行，不创建机器人设备或接管其他进程。"""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import yaml

from piperx_expo.config import ROOT, load_config, resolve_path
from piperx_expo.server.model_store import checkpoint_info, inference_config, verify_model


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

    def resume_info(self, job_id, *, for_inference=False):
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("续训实验ID无效")
        folder = self.runtime / job_id
        record = json.loads((folder / "process.json").read_text())
        config = yaml.safe_load((folder / "experiment.yaml").read_text())
        if config.get("episode_review") is not True or record.get("mock"):
            raise ValueError("只支持已有GUI复核训练，不支持模拟/普通SFT实验")
        output = Path(config["output_dir"])
        if not output.is_absolute():
            output = self.root / output
        checkpoint = output / f"{config['run_name']}_rl" / "checkpoints"
        info = checkpoint_info(checkpoint)
        latest = info["step"]
        if for_inference:
            return config, latest, str(checkpoint)
        # 续训还要求辅助状态和W&B身份；只推理不恢复经验池。
        state = json.loads((checkpoint / f"review_state_{latest}.json").read_text())
        if any(type(state.get(k)) is not int or state[k] < 0 for k in ("credit", "update_count")):
            raise ValueError("复核训练状态无效")
        if not (checkpoint / "wandb_id.txt").read_text().strip():
            raise ValueError("缺少原W&B实验身份")
        return config, latest, str(checkpoint)

    def runs(self):
        result, seen = [], set()
        records = []
        for path in self.runtime.glob("*/process.json"):
            try:
                value = json.loads(path.read_text())
                if not value.get("mock") and value.get("mode", "train") != "eval":
                    records.append(value)
            except (OSError, ValueError):
                continue
        for record in sorted(records, key=lambda r: r.get("started_at", 0), reverse=True):
            if record["run_name"] in seen:
                continue
            seen.add(record["run_name"])
            item = {"id": record["id"], "run_name": record["run_name"],
                    "started_at": record.get("started_at"), "resumable": False, "inferable": False}
            try:
                config, step, checkpoint = self.resume_info(record["id"], for_inference=True)
                item.update(inferable=True, step=step, checkpoint_dir=str(checkpoint),
                            max_steps=config["max_steps"], model_kind="rl")
            except (OSError, ValueError, KeyError) as error:
                item["reason"] = str(error)
            try:
                config, step, checkpoint = self.resume_info(record["id"])
                item.update(resumable=True, step=step, update_count=json.loads(
                    (Path(checkpoint) / f"review_state_{step}.json").read_text())["update_count"],
                    max_steps=config["max_steps"], checkpoint_dir=str(checkpoint), model_kind="rl")
            except (OSError, ValueError, KeyError) as error:
                item["reason"] = str(error)
            result.append(item)
        return result[:100]

    def _budget(self, record=None):
        path = Path(record["config"]) if record else self.root / "configs/server/rotating_square.gui.yaml"
        try:
            config = yaml.safe_load(path.read_text())
            return int(config["max_steps"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def status(self):
        path, record = self.record()
        if record is None:
            return {"phase": "idle", "running": False, "log_tail": "", "runs": self.runs(),
                    "total_steps": self._budget(), "completed_steps": 0}
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
        progress = {}
        try:
            progress = json.loads((path.parent / "training_status.json").read_text())
        except (OSError, ValueError):
            pass
        return {**record, "phase": phase, "running": running, "log_tail": tail,
                "training": progress, "runs": self.runs(), "total_steps": self._budget(record),
                "completed_steps": progress.get("env_step", record.get("resume_step") or 0)}

    def start(self, config_path, *, mock=False, resume_run=None, additional_steps=None, total_steps=None,
              mode="train", model_folder=None):
        with self.lock():
            if mode not in {"train", "eval"}:
                raise ValueError("运行方式只支持继续训练或只推理")
            if model_folder and (mode != "eval" or resume_run):
                raise ValueError("命名模型目录仅用于只推理，且不能同时选择续训实验")
            if mode == "eval" and (not (resume_run or model_folder) or additional_steps is not None or mock):
                raise ValueError("只推理需要选择已有完整模型，不接受追加训练步数或模拟服务")
            if total_steps is not None and (type(total_steps) is not int or not 1 <= total_steps <= 10_000_000):
                raise ValueError("训练总步数必须为1到10000000整数")
            if total_steps is not None and additional_steps is not None:
                raise ValueError("总步数和追加步数不能同时设置")
            if additional_steps is not None and (
                resume_run is None or type(additional_steps) is not int or not 1 <= additional_steps <= 10_000_000
            ):
                raise ValueError("续训追加步数必须为1到10000000整数，且必须选择历史实验")
            if mock and resume_run is not None:
                raise ValueError("模拟服务不支持模型续训")
            current = self.status()
            if current["running"]:
                if current.get("mode", "train") != mode:
                    raise RuntimeError("已有其他运行方式的会话，请先结束")
                if resume_run is not None or model_folder is not None or additional_steps is not None:
                    raise RuntimeError("请先停止当前训练，再选择续训实验")
                if total_steps is not None and total_steps != current.get("total_steps"):
                    raise RuntimeError("现有训练总预算不同，请先停止再修改")
                if current.get("mock") != mock:
                    raise RuntimeError("已有另一运行模式的GUI服务，请先停止")
                return current
            source = resolve_path(config_path)
            config = load_config(str(source))
            resume_step = None
            evaluation_source = None
            if mode == "eval":
                if model_folder:
                    model = verify_model(model_folder)
                    if model["kind"] != "rl":
                        raise ValueError("只推理需要完整RL策略；RTC-SFT基线仅用于初始化")
                    config = load_config(str(Path(model_folder).resolve() / "inference.yaml"))
                    model_step = model["step"]
                    model_dir = str(Path(model_folder).resolve() / "checkpoints")
                else:
                    config, model_step, model_dir = self.resume_info(resume_run, for_inference=True)
                config = inference_config(config, model_dir, model_step, max_steps=total_steps)
                evaluation_source = {"run_id": resume_run, "model_folder": model_folder,
                                     "checkpoint_dir": model_dir, "step": model_step}
            elif resume_run is not None:
                config, resume_step, _ = self.resume_info(resume_run)
                if additional_steps is not None:
                    # 只扩展采集预算；模型、数据、资产、run_name沿用原实验快照。
                    config["max_steps"] = max(config["max_steps"], resume_step + additional_steps)
                if total_steps is not None:
                    config["max_steps"] = total_steps
                if config["max_steps"] <= resume_step:
                    raise ValueError("训练总步数必须大于checkpoint保存步数，或填写续训追加步数")
            if total_steps is not None:
                config["max_steps"] = total_steps
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
            # 每次启动保留独立管理日志；续训复用原训练目录，不新建模型实验。
            if resume_run is None or mode == "eval":
                config["run_name"] = f"{config['run_name']}_{time.strftime('%Y%m%d_%H%M%S')}_{job_id[:6]}"
            if mode == "eval":
                config["evaluation_session_id"] = job_id
            snapshot = folder / "experiment.yaml"
            snapshot.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            record = {"id": job_id, "run_name": config["run_name"], "port": config["port"],
                      "config": str(snapshot), "source_config": str(source), "mock": mock,
                      "log": str(folder / "server.log"), "started_at": time.time(),
                      "resume_run": resume_run if mode == "train" else None, "resume_step": resume_step,
                      "mode": mode, "evaluation_source": evaluation_source, "total_steps": config.get("max_steps")}
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

    def manage_storage(self, action, *, source_id=None, item_id=None, trash_id=None, confirmation=None):
        from piperx_expo.server.storage import CheckpointStorage

        with self.lock():
            storage = CheckpointStorage(self)
            if action == "trash-checkpoint":
                result = storage.delete(source_id, item_id, confirmation == "confirmed")
            elif action == "restore-checkpoint":
                result = storage.restore(source_id, trash_id)
            elif action == "purge-checkpoint":
                result = storage.purge(source_id, trash_id, confirmation)
            elif action == "storage":
                result = None
            else:
                raise ValueError("未知checkpoint管理操作")
            return {**self.status(), "storage": storage.catalog(), "storage_result": result}

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
            if record.get("resume_run"):
                command.append("--resume")
        if (folder / "stop_requested").exists():
            record.update(exit_code=130, ended_at=time.time())
        else:
            process = subprocess.Popen(command, cwd=self.root, env={
                **os.environ, "PIPERX_GUI_STATUS_PATH": str(folder / "training_status.json")})
            record.update(exit_code=process.wait(), ended_at=time.time())
        write_json(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("start", "stop", "status", "storage",
                                                     "trash-checkpoint", "restore-checkpoint", "purge-checkpoint"))
    parser.add_argument("--config", default="configs/server/rotating_square.gui.yaml")
    parser.add_argument("--worker")
    parser.add_argument("--resume-run")
    parser.add_argument("--additional-steps", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--model-folder", help="只推理时使用已登记的固定RL版本；不从HTTP接受任意目录")
    parser.add_argument("--source-id")
    parser.add_argument("--item-id")
    parser.add_argument("--trash-id")
    parser.add_argument("--confirmation")
    parser.add_argument("--mock", action="store_true", help="仅命令行管理链路验收；不训练、不连接机器人")
    args = parser.parse_args()
    manager = ServiceManager()
    if args.worker:
        manager.worker(args.worker)
        return
    try:
        if args.action in {"storage", "trash-checkpoint", "restore-checkpoint", "purge-checkpoint"}:
            result = manager.manage_storage(args.action, source_id=args.source_id, item_id=args.item_id,
                                            trash_id=args.trash_id, confirmation=args.confirmation)
        else:
            result = manager.start(args.config, mock=args.mock, resume_run=args.resume_run,
                               additional_steps=args.additional_steps, total_steps=args.total_steps,
                               mode=args.mode, model_folder=args.model_folder) if args.action == "start" else (
                manager.stop() if args.action == "stop" else manager.status())
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"phase": "error", "error": str(error)}, ensure_ascii=False))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
