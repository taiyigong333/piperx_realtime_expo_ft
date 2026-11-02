import copy
import threading
import time

import pytest

from piperx_expo.client.gui_control import GUIController
from piperx_expo.config import load_config
from piperx_expo.review import PROTOCOL


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "等待流程状态超时"
        time.sleep(.01)


class Remote:
    def __init__(self):
        self.calls = []
        self.value = {"phase": "stopped", "running": False, "busy": False, "total_steps": 100}

    def request(self, action, **values):
        self.calls.append((action, values))
        if action == "start":
            self.value.update(phase="loading", running=True, total_steps=values.get("total_steps", 100))
        elif action == "stop":
            self.value.update(phase="stopped", running=False)

    def snapshot(self):
        return copy.deepcopy(self.value)


def test_prepare_only_starts_server_and_waits_for_separate_human_ready(tmp_path, monkeypatch):
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    remote = controller.remote_server = Remote()

    def run():
        controller.handle({"operation": "create_env", "task_spec": {
            **controller.config, "review_protocol": PROTOCOL, "action_mode": "absolute_pose_width_m"}})
        controller.handle({"operation": "prepare_policy", "env_id": controller.env_id})
        controller.handle({"operation": "policy_status", "env_id": controller.env_id,
                           "details": {"stage": "ready"}})
        controller.handle({"operation": "episode_context", "env_id": controller.env_id,
                           "start_step": 20, "total_steps": 100})
        controller.handle({"operation": "reset", "env_id": controller.env_id})

    monkeypatch.setattr(controller, "_run", run)
    try:
        controller.command("prepare", {"total_steps": 100})
        wait_for(lambda: controller.phase == "awaiting_reset")
        assert remote.calls == [("start", {"total_steps": 100})]
        assert controller.recorder is None and controller.env.steps == 0 and not controller.env.armed
        assert controller.state()["training_progress"] == {"total_steps": 100, "completed_steps": 20}
        controller.command("ready", {})
        wait_for(lambda: controller.phase == "running")
        assert controller.recorder and controller.env.steps == 0
    finally:
        controller.stop()
        controller.prepare_worker.join(2)
        controller.worker.join(2)


def test_finish_while_waiting_never_starts_next_reset(tmp_path):
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    controller.handle({"operation": "create_env", "task_spec": {
        **controller.config, "review_protocol": PROTOCOL, "action_mode": "absolute_pose_width_m"}})
    replies = []
    worker = threading.Thread(target=lambda: replies.append(controller.handle({
        "operation": "reset", "env_id": controller.env_id})))
    try:
        worker.start()
        wait_for(lambda: controller.phase == "awaiting_reset")
        controller.command("finish_training", {})
        worker.join(2)
        assert replies == [{"status": "success", "stop_after_episode": True}]
        assert controller.recorder is None and not controller.env.armed and not controller.env.fault
        with pytest.raises(ValueError, match="已经申请"):
            controller.command("ready", {})
    finally:
        controller.stop()
        worker.join(2)


def test_real_ready_requires_explicit_motion_confirmation(tmp_path):
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    controller.config["backend"] = "piperx"
    controller.phase = "awaiting_reset"
    try:
        with pytest.raises(ValueError, match="运动确认"):
            controller.command("ready", {})
        assert not controller.ready
        controller.command("ready", {"motion_confirmed": True})
        assert controller.ready
    finally:
        controller.stop()
