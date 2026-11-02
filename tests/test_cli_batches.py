"""命令行按轨迹数、批后传输、人工标签与急停继续的软件验收。"""

import json
import re
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from piperx_expo.client.cli import CLIController
from piperx_expo.config import load_config
from piperx_expo.data.online_batch import decode_episode
from piperx_expo.review import PROTOCOL
from piperx_expo.server.cli_loop import batch_options, run_cli_batches
from piperx_expo.server.review_loop import restore_reviewed
from test_batch_collection import Processor, Sampler


@pytest.mark.parametrize("interruption", ["none", "between_steps", "missing_feedback", "zero_steps", "homing"])
def test_cli_two_rounds_transfer_only_after_episode_count_and_continue_after_stop(tmp_path, interruption):
    config = load_config("configs/client/gui.mock.yaml")
    config.update(episode_steps=3, max_steps=1, replan_steps=2, control_hz=1000,
                  cli_batch={"episodes_per_batch": 2, "rounds": 2, "updates_per_batch": 3, "replay_capacity": 100})
    controller = CLIController(config, tmp_path / "client")
    controller.phase = "connecting"
    controller.handle({"operation": "create_env", "task_spec": {
        **config, "review_protocol": PROTOCOL, "cli_batch": True, "action_mode": "absolute_pose_width_m"}})
    calls, errors, processors, updates = [], [], [], []
    aborted = False
    reads = 0

    def call(operation, params):
        nonlocal aborted, reads
        calls.append(operation)
        if operation == "step" and not aborted:
            if interruption == "zero_steps" or (interruption == "between_steps" and controller.recorder.meta["steps"] == 1):
                aborted = True
                controller.interrupt_episode()
        if operation == "get_observation":
            reads += 1
            if interruption == "missing_feedback" and not aborted and reads == 2:
                aborted = True
                controller.interrupt_episode()
        return controller.handle({"operation": operation, **params})

    env = SimpleNamespace(env_id=controller.env_id, client=SimpleNamespace(_call_operation=call))
    call("prepare_policy", {"env_id": env.env_id, "batch_training": True, "episodes_per_batch": 2})
    call("policy_status", {"env_id": env.env_id, "details": {"stage": "ready"}})
    if interruption == "homing":
        def interrupted_home(**kwargs):
            nonlocal aborted
            controller.phase = "homing"
            controller.interrupt_episode()
            aborted = True
            raise RuntimeError("归位急停")
        controller.env.device.reset = interrupted_home

    def prepare():
        if not processors:
            assert calls.count("episode_ack") == 2
            assert calls.count("export_episode") == 2
            processors.append(Processor())
        return processors[0]

    def train(agent, processor, step, count):
        assert calls.count("episode_ack") in (2, 4)
        assert len(controller.batch_pending_ids) == 0
        updates.append((step, count))
        return agent

    def worker():
        try:
            run_cli_batches(env, Sampler(), None, config, tmp_path / "server",
                            prepare_training=prepare, train_and_save=train)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    deadline = time.monotonic() + 10
    try:
        while thread.is_alive():
            assert time.monotonic() < deadline, controller.phase
            with controller.condition:
                if controller.phase == "awaiting_reset" and not controller.ready:
                    if not processors:
                        with pytest.raises(RuntimeError, match="采够"):
                            call("export_batch", {"env_id": env.env_id})
                    controller.command("ready", {"motion_confirmed": True})
                elif controller.phase == "review_pending":
                    controller.submit(True, True)
            time.sleep(.001)
        assert not errors, errors
        assert controller.phase == "completed"
        assert len(updates) == 2 and all(count == 3 for _, count in updates)
        # max_steps=1故意小于单条长度，仍必须完成4条而非按step截断。
        assert calls.count("episode_ack") == 4 and calls.count("export_episode") == 4
        assert calls.index("export_episode") > [i for i, value in enumerate(calls) if value == "episode_ack"][1]
        metas = [json.loads((folder / "metadata.json").read_text()) for folder in controller.episodes.values()]
        assert len(metas) == 4
        if interruption not in {"none", "homing"}:
            assert metas[0]["review"]["success"] is False
            assert metas[0]["terminal_reason"] == "emergency_stop"
            assert metas[1]["review"]["success"] is True
            if interruption in {"missing_feedback", "zero_steps"}:
                assert metas[0]["review"]["admitted"] is False
            else:
                assert metas[0]["review"]["admitted"] is True
                assert processors[0].successes[0] is False
                assert processors[0].transitions[0]["masks"] == 0
        if interruption == "homing":
            assert aborted
        assert all(re.match(r"\d{8}_\d{6}_\d{6}_", folder.name) for folder in controller.episodes.values())
        batches = list((tmp_path / "server/collection_batches").iterdir())
        assert len(batches) == 2 and all(re.match(r"\d{8}_\d{6}_", path.name) for path in batches)
        restored = Processor()
        restore_reviewed(restored, tmp_path / "server", updates[-1][0])
        assert len(restored.transitions) == len(processors[0].transitions)
    finally:
        controller.shutdown()
        thread.join(2)


def test_cli_rejects_gui_server_and_early_export(tmp_path):
    controller = CLIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    try:
        with pytest.raises(ValueError, match="命令行"):
            controller.handle({"operation": "create_env", "task_spec": {"review_protocol": PROTOCOL}})
    finally:
        controller.shutdown()


@pytest.mark.parametrize("key", ["episodes_per_batch", "rounds", "updates_per_batch", "replay_capacity"])
def test_cli_options_reject_bool_and_zero(key):
    for value in (True, 0, -1, 1.5):
        options = dict(episodes_per_batch=2, rounds=2, updates_per_batch=1, replay_capacity=100)
        options[key] = value
        with pytest.raises(ValueError, match=key):
            batch_options({"cli_batch": options})


def test_transferred_episode_digest_is_checked(tmp_path):
    from piperx_expo.client.recording import EpisodeRecorder
    from piperx_expo.client.devices import MockDevices
    config = load_config("configs/client/gui.mock.yaml")
    obs = MockDevices(config).observe()
    recorder = EpisodeRecorder(tmp_path, obs, config)
    recorder.append(np.r_[obs["cartesian_position"], obs["gripper_position"]], obs, (0, 1, 1, 0))
    recorder.finish()
    review = {"steps": 1, "digest": "0" * 64, "auto_success": False, "manual_success": False, "admitted": True}
    with pytest.raises(ValueError, match="摘要"):
        decode_episode((recorder.folder / "episode.hdf5").read_bytes(), review, config["language_instruction"])


def test_cli_real_websocket_batch_transfer(tmp_path, monkeypatch):
    import sys
    from piperx_expo.config import ROOT
    sys.path.insert(0, str(ROOT / "third_party/expo"))
    from expo_ft.env.env_client import EnvClient, EnvClientWrapper
    from piperx_expo.server.launch import _install_rpc_contract
    for cls, names in ((EnvClient, ("create_env", "_call_operation")), (EnvClientWrapper, ("_call",))):
        for name in names:
            monkeypatch.setattr(cls, name, getattr(cls, name))
    config = load_config("configs/client/gui.mock.yaml")
    config.update(episode_steps=2, max_steps=1, replan_steps=2, control_hz=1000, episode_review=True,
                  cli_batch={"episodes_per_batch": 2, "rounds": 2, "updates_per_batch": 1, "replay_capacity": 100})
    _install_rpc_contract(config)
    rpc = EnvClient(host="127.0.0.1", port=0)
    rpc._ensure_server()
    config["server_port"] = rpc._server.socket.getsockname()[1]
    controller = CLIController(config, tmp_path / "client")
    processor, errors, updates = Processor(), [], []

    def learner():
        try:
            env = SimpleNamespace(client=rpc)
            env.env_id, _ = rpc.create_env({})
            assert rpc.last_response["cli_batch"] is True
            rpc._call_operation("prepare_policy", {"env_id": env.env_id, "batch_training": True,
                                                   "episodes_per_batch": 2})
            rpc._call_operation("policy_status", {"env_id": env.env_id, "details": {"stage": "ready"}})

            def train(agent, *args):
                updates.append(args[-1])
                return agent

            run_cli_batches(env, Sampler(), None, config, tmp_path / "server",
                            prepare_training=lambda: processor, train_and_save=train)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=learner, daemon=True)
    worker.start()
    try:
        controller.start()
        deadline = time.monotonic() + 10
        while worker.is_alive():
            assert time.monotonic() < deadline, (controller.phase, errors)
            with controller.condition:
                if controller.phase == "awaiting_reset" and not controller.ready:
                    controller.command("ready", {})
                elif controller.phase == "review_pending":
                    controller.submit(False, True)
            time.sleep(.001)
        assert not errors, errors
        assert updates == [1, 1] and len(processor.transitions) == 8
        assert controller.phase == "completed"
    finally:
        controller.shutdown()
        rpc.close()
        worker.join(3)


def test_stop_after_rollout_overrides_success_before_review(tmp_path):
    from piperx_expo.client.recording import EpisodeRecorder
    from piperx_expo.client.devices import MockDevices
    config = load_config("configs/client/gui.mock.yaml")
    controller = CLIController(config, tmp_path)
    try:
        device = MockDevices(config)
        obs = device.observe()
        from piperx_expo.client.environment import PiperXEnv
        controller.env = PiperXEnv(config, device=device)
        controller.recorder = EpisodeRecorder(tmp_path, obs, config)
        controller.recorder.append(np.r_[obs["cartesian_position"], obs["gripper_position"]], obs, (1, 0, 1, 1))
        controller.recorder.finish(auto_success=True)
        controller.phase = "review_pending"
        controller.interrupt_episode()
        assert controller.decision["auto_success"] is True
        assert controller.decision["manual_success"] is False
        assert controller.decision["success"] is False
        assert not controller.stop_event.is_set()
    finally:
        controller.shutdown()
