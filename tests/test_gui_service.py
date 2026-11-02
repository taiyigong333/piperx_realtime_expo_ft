import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import yaml

from piperx_expo.client.gui_control import GUIController
from piperx_expo.client.remote_server import RemoteServer
from piperx_expo.config import load_config
from piperx_expo.server.gui_service import ServiceManager, owned


def wait_for(predicate):
    deadline = time.monotonic() + 8
    while not predicate():
        assert time.monotonic() < deadline, "等待管理状态超时"
        time.sleep(0.02)


@pytest.fixture
def service(tmp_path):
    if sys.platform != "linux":
        pytest.skip("服务器进程身份/监听检查使用Linux /proc；在4090执行")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"episode_review": True, "port": port, "run_name": "test_gui"}))
    manager = ServiceManager(tmp_path)
    try:
        yield manager, config
    finally:
        manager.stop()


def test_managed_start_log_stop_and_unique_restart(service):
    manager, config = service
    record = manager.start(str(config), mock=True)
    assert record["running"] and owned(record)
    assert manager.start(str(config), mock=True)["id"] == record["id"]
    wait_for(lambda: manager.status()["phase"] == "listening")
    assert "MOCK GUI learner ready" in manager.status()["log_tail"]
    result = manager.stop()
    assert not result["running"] and result["phase"] == "stopped"
    assert Path(result["log"]).exists()
    second = manager.start(str(config), mock=True)
    assert second["run_name"] != record["run_name"] and second["id"] != record["id"]


def test_foreign_port_is_not_taken_over(service):
    manager, config = service
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", yaml.safe_load(config.read_text())["port"]))
        sock.listen()
        with pytest.raises(RuntimeError, match="占用"):
            manager.start(str(config), mock=True)
        assert manager.status()["phase"] == "idle"


def test_stolen_pid_never_signalled(service, monkeypatch):
    manager, config = service
    manager.start(str(config), mock=True)
    manager.stop()
    path, record = manager.record()
    record["pid"] = os.getpid()
    path.write_text(json.dumps(record))
    monkeypatch.setattr(os, "killpg", lambda *_: pytest.fail("不得向非本管理器进程发信号"))
    assert not manager.stop()["running"]


def test_remote_ssh_is_noninteractive_and_paths_are_quoted(monkeypatch):
    remote = object.__new__(RemoteServer)
    remote.config = {"ssh_host": "wired-server", "project_dir": "/tmp/project space", "config": "config.yaml"}
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs["timeout"] == 40
        return SimpleNamespace(returncode=0, stdout='{"phase":"idle"}', stderr="")

    monkeypatch.setattr("piperx_expo.client.remote_server.subprocess.run", run)
    assert remote.call("status")["phase"] == "idle"
    assert "-oBatchMode=yes" in calls[0] and "cd '/tmp/project space'" in calls[0][-1]
    with pytest.raises(ValueError, match="别名"):
        RemoteServer({**remote.config, "ssh_host": "-oProxyCommand=oops"})


def test_gui_remote_stop_locks_local_device_first(tmp_path):
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    events = []
    controller.remote_server = SimpleNamespace(request=lambda action: events.append(action))
    controller.env = SimpleNamespace(close=lambda: events.append("local_close"),
                                     emergency_stop=lambda: events.append("emergency"),
                                     closed=threading.Event())
    try:
        controller.command("server_start", {})
        assert events == ["start"]
        with pytest.raises(ValueError, match="覆盖"):
            controller.command("server_start", {"ssh_host": "elsewhere"})
        controller.command("server_stop", {})
        assert events[-3:] == ["emergency", "local_close", "stop"]
        assert controller.stop_event.is_set()
    finally:
        controller.env = controller.remote_server = None
        controller.stop()


def test_gui_can_create_fresh_session_after_stop(tmp_path):
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    try:
        controller.start(demo=True)
        wait_for(lambda: controller.phase == "awaiting_reset")
        old_env = controller.env
        controller.stop()
        controller.worker.join(2)
        controller.start(demo=True)
        wait_for(lambda: controller.phase == "awaiting_reset")
        assert old_env.closed.is_set() and old_env.device.stopped
        assert controller.env is not old_env and not controller.env.closed.is_set()
    finally:
        controller.stop()
        controller.worker.join(2)


def test_remote_requests_do_not_block_gui(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def slow(self, action):
        entered.set()
        assert release.wait(2)
        return {"phase": "idle"}

    monkeypatch.setattr(RemoteServer, "call", slow)
    remote = RemoteServer({"ssh_host": "wired", "project_dir": "/tmp/test", "config": "config.yaml"})
    try:
        remote.request("start")
        assert entered.wait(1)
        assert remote.snapshot()["busy"]
        remote.request("stop")
        assert "stop" in remote.pending
    finally:
        remote.closed.set()
        release.set()
        remote.thread.join(2)


def test_gui_stop_before_first_connection_can_restart(tmp_path):
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    controller.stop()
    try:
        controller.start(demo=True)
        wait_for(lambda: controller.phase == "awaiting_reset")
        assert not controller.stop_event.is_set()
    finally:
        controller.stop()
        controller.worker.join(2)


def test_start_error_survives_status_refresh(monkeypatch):
    def call(self, action):
        if action == "start":
            raise RuntimeError("GPU已有计算进程")
        return {"phase": "idle" if action == "status" else "stopped", "running": False}

    monkeypatch.setattr(RemoteServer, "call", call)
    remote = RemoteServer({"ssh_host": "wired", "project_dir": "/tmp/test", "config": "config.yaml"})
    try:
        remote.request("start")
        wait_for(lambda: remote.snapshot().get("error"))
        remote.request("status")
        wait_for(lambda: remote.snapshot()["phase"] == "idle")
        assert remote.snapshot()["error"] == "GPU已有计算进程"
        remote.request("stop")
        wait_for(lambda: remote.snapshot()["phase"] == "stopped")
        assert not remote.snapshot().get("error")
    finally:
        remote.closed.set()
        remote.request("status")
        remote.thread.join(2)


def saved_run(tmp_path, *, complete=True):
    manager = ServiceManager(tmp_path)
    job_id = "a" * 32
    folder = manager.runtime / job_id
    folder.mkdir()
    config = {"episode_review": True, "run_name": "original", "output_dir": str(tmp_path / "runs"),
              "max_steps": 100, "port": 8103}
    (folder / "experiment.yaml").write_text(yaml.safe_dump(config))
    (folder / "process.json").write_text(json.dumps({
        "id": job_id, "run_name": "original", "mock": False, "started_at": 1}))
    checkpoints = tmp_path / "runs/original_rl/checkpoints"
    (checkpoints / "80").mkdir(parents=True)
    (checkpoints / "wandb_id.txt").write_text("same-wandb-id")
    if complete:
        (checkpoints / "80/_CHECKPOINT_METADATA").write_text("{}")
        (checkpoints / "review_state_80.json").write_text(json.dumps({"credit": 2, "update_count": 3}))
    return manager, job_id, checkpoints


def test_resume_catalog_requires_matching_saved_state(tmp_path):
    manager, job_id, checkpoints = saved_run(tmp_path)
    assert manager.resume_info(job_id)[1] == 80
    assert manager.runs()[0]["resumable"]
    assert manager.runs()[0]["update_count"] == 3
    (checkpoints / "review_state_80.json").unlink()
    assert not manager.runs()[0]["resumable"]
    with pytest.raises(ValueError, match="ID"):
        manager.resume_info("../other")


def test_resume_reuses_experiment_and_passes_resume_flag(tmp_path, monkeypatch):
    import piperx_expo.server.gui_service as service_module

    manager, job_id, checkpoints = saved_run(tmp_path)
    monkeypatch.setattr(manager, "status", lambda: {"running": False})
    monkeypatch.setattr(service_module, "listening", lambda _: False)
    monkeypatch.setattr(service_module, "process_ticks", lambda _: "123")
    monkeypatch.setattr("piperx_expo.server.launch.validate", lambda *_: None)
    monkeypatch.setattr(service_module.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    calls = []
    monkeypatch.setattr(service_module.subprocess, "Popen",
                        lambda argv, **kw: calls.append(argv) or SimpleNamespace(pid=999, wait=lambda: 0))
    # 当前配置故意不同；续训必须沿用所选实验，而非当前默认run_name。
    source = tmp_path / "current.yaml"
    source.write_text(yaml.safe_dump({"run_name": "different"}))
    manager.start(str(source), resume_run=job_id, additional_steps=200)
    path, record = manager.record()
    config = yaml.safe_load(Path(record["config"]).read_text())
    assert config["run_name"] == "original" and config["max_steps"] == 280
    assert record["resume_step"] == 80 and record["resume_run"] == job_id
    monkeypatch.setattr(service_module.signal, "signal", lambda *_: None)
    manager.worker(record["id"])
    assert "--resume" in calls[-1]
    assert (checkpoints / "wandb_id.txt").read_text() == "same-wandb-id"


def test_partial_checkpoint_and_finished_budget_are_rejected(tmp_path):
    manager, job_id, checkpoints = saved_run(tmp_path, complete=False)
    assert not manager.runs()[0]["resumable"]
    with pytest.raises(ValueError, match="提交元数据"):
        manager.resume_info(job_id)
