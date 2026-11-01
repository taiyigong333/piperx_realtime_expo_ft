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
    controller.env = SimpleNamespace(close=lambda: events.append("local_close"))
    try:
        controller.command("server_start", {})
        assert events == ["start"]
        with pytest.raises(ValueError, match="覆盖"):
            controller.command("server_start", {"ssh_host": "elsewhere"})
        controller.command("server_stop", {})
        assert events[-2:] == ["local_close", "stop"]
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
