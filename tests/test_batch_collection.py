"""整批流程的软件验证；不加载模型，不连接CAN。"""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from piperx_expo.client.gui_control import GUIController
from piperx_expo.config import load_config
from piperx_expo.review import PROTOCOL
from piperx_expo.server.review_loop import DeferredBatch, collect_episode, restore_reviewed, update_credit
from piperx_expo.server.review_loop import evaluate_session
from piperx_expo.server.session_control import StopAfterEpisode


class Processor:
    def __init__(self):
        self.transitions, self.successes = [], []

    def on_episode_start(self):
        pass

    def insert_transition(self, item):
        self.transitions.append(item)

    def on_episode_done(self, success):
        self.successes.append(success)


class Sampler:
    def observe(self, obs):
        pass

    def launch(self, *args):
        pass

    def sample(self, agent, obs, metrics):
        action = np.r_[obs["cartesian_position"], obs["gripper_position"]]
        return np.tile(action, (8, 1)), agent, {}

    def on_episode_end(self):
        pass


@pytest.mark.parametrize("stop_at_next_reset", [False, True])
def test_multiple_episodes_remain_out_of_replay_until_collection_ends(tmp_path, stop_at_next_reset):
    config = load_config("configs/client/gui.mock.yaml")
    config.update(episode_steps=2, admission_mode="auto")
    controller = GUIController(config, tmp_path / "client")
    controller.phase = "connecting"
    controller.handle({"operation": "create_env", "task_spec": {
        **config, "review_protocol": PROTOCOL, "action_mode": "absolute_pose_width_m"}})

    def call(operation, params):
        return controller.handle({"operation": operation, **params})

    def step(action):
        reply = call("step", {"env_id": controller.env_id, "action": action})
        return reply["action"], reply["action_type"]

    env = SimpleNamespace(env_id=controller.env_id, client=SimpleNamespace(_call_operation=call), step=step)
    warm = call("prepare_policy", {"env_id": env.env_id, "batch_training": True})
    assert warm["batch_training"] is True
    call("policy_status", {"env_id": env.env_id, "details": {"stage": "ready"}})
    processor, pending = Processor(), DeferredBatch(tmp_path / "checkpoints")
    errors, ids = [], []

    def learner():
        try:
            for start in (0, 2):
                call("episode_context", {"env_id": env.env_id, "start_step": start, "total_steps": 4})
                result, _ = collect_episode(env, Sampler(), None, 8, 1000, 4 - start)
                pending.stage(result, start, start + 2)
                ids.append(result["review"]["episode_id"])
                call("episode_ack", {"env_id": env.env_id, "episode_id": ids[-1], "outcome": "queued"})
                reply = call("batch_status", {"env_id": env.env_id, "stage": "episode_queued"})
                assert not reply["stop_after_episode"]
                assert processor.transitions == []
                assert not (tmp_path / "checkpoints/reviewed_episodes").exists()
            if stop_at_next_reset:
                with pytest.raises(StopAfterEpisode):
                    collect_episode(env, Sampler(), None, 8, 1000, 2)
            call("batch_status", {"env_id": env.env_id, "stage": "processing_batch"})
            with pytest.raises(RuntimeError, match="整批轨迹"):
                call("session_complete", {"env_id": env.env_id})
            for review, count, end in pending.commit(processor, tmp_path / "checkpoints"):
                assert count == 2
                call("batch_episode_ack", {"env_id": env.env_id, "episode_id": review["episode_id"],
                                           "outcome": "accepted"})
            call("session_complete", {"env_id": env.env_id, "progress": {
                "completed_steps": 4, "checkpoint_step": 4, "update_count": 0}})
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=learner)
    worker.start()
    try:
        for _ in range(3 if stop_at_next_reset else 2):
            deadline = time.monotonic() + 3
            while controller.phase != "awaiting_reset":
                assert not errors, errors
                assert time.monotonic() < deadline
                time.sleep(.01)
            if len(ids) == 2:
                controller.command("finish_training", {})
            else:
                controller.command("ready", {})
            # 同一个awaiting_reset不能被当成下一条，等待当前确认被消费。
            while controller.phase == "awaiting_reset" and controller.ready:
                assert time.monotonic() < deadline
                time.sleep(.001)
        worker.join(3)
        assert not worker.is_alive() and not errors, errors
        assert controller.phase == "completed" and not controller.batch_pending_ids
        assert len(processor.transitions) == 4
        assert [t["dones"] for t in processor.transitions] == [False, True, False, True]
        for episode_id in ids:
            meta = json.loads((controller.root / episode_id / "metadata.json").read_text())
            assert meta["status"] == "accepted"
        restored = Processor()
        restore_reviewed(restored, tmp_path / "checkpoints", 4)
        assert len(restored.transitions) == 4
    finally:
        controller.stop()
        worker.join(3)


def test_rejected_batch_does_not_touch_replay_or_allocate_credit(tmp_path):
    pending, processor = DeferredBatch(tmp_path), Processor()
    pending.stage({"transitions": [{"unused": True}], "review": {
        "episode_id": "a" * 32, "admitted": False}}, 0, 1)
    assert list(pending.commit(processor, tmp_path)) == [({"episode_id": "a" * 32, "admitted": False}, 0, 1)]
    assert not processor.transitions and not processor.successes


def test_cold_start_and_update_credit_keep_original_threshold():
    assert update_credit(5, 9, 200, 1800, 4, 30) == (0, 5)
    assert update_credit(5, 10, 200, 2000, 4, 30) == (6, 25)
    assert update_credit(25, 11, 10, 2010, 4, 30) == (1, 5)
    assert update_credit(5, 10, 1, 3, 4, 30) == (0, 5)


@pytest.mark.parametrize("manual_finish", [False, True])
def test_saved_model_inference_only_records_results_and_never_admits_training_data(tmp_path, manual_finish):
    config = load_config("configs/client/gui.mock.yaml")
    config.update(episode_steps=2)
    controller = GUIController(config, tmp_path / "client")
    controller.phase = "connecting"

    def call(operation, params):
        return controller.handle({"operation": operation, **params})

    call("create_env", {"task_spec": {**config, "review_protocol": PROTOCOL,
                                      "action_mode": "absolute_pose_width_m"}})
    env = SimpleNamespace(env_id=controller.env_id, client=SimpleNamespace(_call_operation=call))
    env.step = lambda action: (call("step", {"env_id": env.env_id, "action": action})["action"], "policy")
    reply = call("prepare_policy", {"env_id": env.env_id, "session_mode": "eval"})
    assert reply["session_mode"] == "eval" and not reply["batch_training"]
    call("policy_status", {"env_id": env.env_id, "details": {"stage": "ready"}})
    errors = []

    def evaluate():
        try:
            evaluate_session(env, Sampler(), None, replan_steps=8, control_hz=1000, max_steps=4,
                             output_dir=tmp_path / "evaluations", checkpoint_step=419)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=evaluate)
    worker.start()
    try:
        for _ in range(1 if manual_finish else 2):
            for phase in ("awaiting_reset", "review_pending"):
                deadline = time.monotonic() + 3
                while controller.phase != phase:
                    assert not errors, errors
                    assert time.monotonic() < deadline
                    time.sleep(.005)
                if phase == "awaiting_reset":
                    controller.command("ready", {})
            if manual_finish:
                controller.command("finish_training", {})
            # 即使误传收录=True，只推理模式也必须强制禁止入训练池。
            controller.submit(True, True)
            while controller.phase == "submitted":
                assert time.monotonic() < deadline
                time.sleep(.005)
        worker.join(3)
        assert not worker.is_alive() and not errors, errors
        assert controller.phase == "completed" and controller.recorder.meta["status"] == "evaluated"
        assert controller.training_progress["checkpoint_step"] == 419
        assert controller.training_progress["update_count"] == 0
        assert controller.training_progress["completed_steps"] == (2 if manual_finish else 4)
        assert not (tmp_path / "checkpoints").exists()
        results = list((tmp_path / "evaluations/reviewed_episodes").glob("*/metadata.json"))
        assert len(results) == (1 if manual_finish else 2)
        for path in results:
            metadata = json.loads(path.read_text())
            assert metadata["evaluation_only"] and not metadata["admitted"]
            assert not (path.parent / "transitions.pkl").exists()
    finally:
        controller.stop()
        worker.join(3)
