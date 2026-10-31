import json
import threading
import time
import urllib.error
import urllib.request

import h5py
import numpy as np
import pytest

from piperx_expo.client.gui import create_server
from piperx_expo.client.gui_control import GUIController
from piperx_expo.client.rpc import Dispatcher
from piperx_expo.config import ROOT, load_config
from piperx_expo.review import PROTOCOL, commit_episode, resolve_review
from piperx_expo.server.review_loop import collect_episode, persist_reviewed, restore_reviewed


def wait_for(predicate, timeout=5):
    until = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= until:
            raise AssertionError("状态等待超时")
        time.sleep(0.01)


class Processor:
    def __init__(self):
        self.transitions, self.successes = [], []
        self.starts = 0

    def on_episode_start(self):
        self.starts += 1

    def insert_transition(self, transition):
        self.transitions.append(transition)

    def on_episode_done(self, success):
        self.successes.append(success)


class Sampler:
    def observe(self, obs):
        pass

    def launch(self, *args):
        pass

    def sample(self, agent, obs, metrics):
        action = np.r_[obs["cartesian_position"], obs["gripper_position"]]
        action[2] += 0.001
        return np.tile(action, (8, 1)), agent, {}

    def on_episode_end(self):
        pass


@pytest.mark.parametrize(
    "auto,manual,success,mask",
    [
        (True, False, False, 0),
        (False, True, True, 0),
        (True, None, True, 0),
        (False, None, False, 1),
    ],
)
def test_manual_priority(auto, manual, success, mask):
    result = resolve_review(auto, manual, True)
    assert result["success"] is success
    assert result["reward"] == float(success) and result["mask"] == mask
    with pytest.raises(ValueError):
        resolve_review(auto, "false", True)


def test_cli_remains_separate(tmp_path):
    cfg = load_config("configs/client/mock.yaml")
    cli = Dispatcher(cfg)
    try:
        assert cli.handle({"operation": "create_env"})["status"] == "success"
        with pytest.raises(ValueError, match="浏览器"):
            cli.handle({"operation": "create_env", "task_spec": {"review_protocol": PROTOCOL}})
        gui = GUIController(cfg, tmp_path)
        with pytest.raises(ValueError, match="浏览器"):
            gui.handle({"operation": "create_env"})
        gui.stop()
    finally:
        cli.close()


@pytest.mark.parametrize(
    "admission_mode,manual,admitted",
    [
        ("manual", True, True),
        ("manual", False, True),
        ("manual", True, False),
        ("auto", None, True),
    ],
)
def test_wire_episode_review_and_recovery(tmp_path, monkeypatch, admission_mode, manual, admitted):
    # 使用真实msgpack/WebSocket往返和原EnvClient，但不用模型/GPU或机器人。
    import sys

    sys.path.insert(0, str(ROOT / "third_party/expo"))
    from expo_ft.env.env_client import EnvClient, EnvClientWrapper
    from piperx_expo.server.launch import _install_rpc_contract

    for cls, names in ((EnvClient, ("create_env", "_call_operation")), (EnvClientWrapper, ("_call",))):
        for name in names:
            monkeypatch.setattr(cls, name, getattr(cls, name))
    cfg = load_config("configs/client/gui.mock.yaml")
    cfg.update(episode_steps=3, admission_mode=admission_mode)
    if manual is False:
        cfg["success_detector_config"].update(height_m=0.1, gripper_width_m=[0.03, 0.05], consecutive_steps=3)
    _install_rpc_contract({**cfg, "episode_review": True})
    learner = EnvClient(host="127.0.0.1", port=0)
    learner._ensure_server()
    cfg["server_port"] = learner._server.socket.getsockname()[1]
    controller = GUIController(cfg, tmp_path / "client")
    processor = Processor()
    errors, results = [], []

    def learner_main():
        try:
            env = object.__new__(EnvClientWrapper)
            env.client = learner
            env.env_id, env.task_description = learner.create_env({})
            assert learner.last_response["review_protocol"] == PROTOCOL
            result, _ = collect_episode(env, Sampler(), None, 8, 1000, 3)
            review, transitions = result["review"], result["transitions"]
            prepared = commit_episode(processor, transitions, review)
            persist_reviewed(tmp_path / "checkpoints", prepared, review, 0, 3)
            learner._call_operation(
                "episode_ack",
                {
                    "env_id": env.env_id,
                    "episode_id": review["episode_id"],
                    "outcome": "accepted" if admitted else "rejected",
                },
            )
            results.append(result)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=learner_main, daemon=True)
    thread.start()
    try:
        controller.start()
        wait_for(lambda: controller.phase == "awaiting_reset")
        controller.command("ready", {})
        if admission_mode == "manual":
            wait_for(lambda: controller.phase == "review_pending")
            assert processor.transitions == []  # 复核前不入池。
            controller.submit(manual, admitted)
        thread.join(5)
        assert not thread.is_alive() and not errors, errors
        assert results
        assert controller.recorder.meta["status"] == ("accepted" if admitted else "rejected")
        if admitted:
            effective = bool(manual) if manual is not None else False
            assert processor.starts == 1 and processor.successes == [effective]
            assert len(processor.transitions) == 3
            assert [t["rewards"] for t in processor.transitions] == [0, 0, float(effective)]
            assert [t["dones"] for t in processor.transitions] == [False, False, True]
        else:
            assert processor.starts == 0 and processor.transitions == []
        folder = controller.recorder.folder
        with h5py.File(folder / "episode.hdf5") as file:
            assert file["actions"].shape == (3, 7)
            assert file["observations/state"].shape == (4, 7)
            np.testing.assert_allclose(file["actions"][:, :3], file["observations/state"][1:, :3])
            assert file["observations/exterior_image_1_left"].shape == (4, 224, 224, 3)
        restored = Processor()
        restore_reviewed(restored, tmp_path / "checkpoints", 3)
        assert len(restored.transitions) == (3 if admitted else 0)
        # 超出模型checkpoint的轨迹隔离，不会混入下次恢复。
        restore_reviewed(Processor(), tmp_path / "checkpoints", 0)
        assert not list((tmp_path / "checkpoints/reviewed_episodes").glob("*/metadata.json"))
        assert len(list((tmp_path / "checkpoints/unrestored_episodes").glob("*/metadata.json"))) == 1
    finally:
        controller.stop()
        learner.close()
        controller.worker.join(5)


def test_http_auth_and_parameter_validation(tmp_path):
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    server = create_server(controller, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        page = urllib.request.urlopen(base).read().decode()
        assert "回合复核" in page and "__EXPO_TOKEN__" not in page
        request = urllib.request.Request(base + "/api/command/start", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(request)
        assert denied.value.code == 403
        import re

        token = re.search(r'name="expo-token" content="([^"]+)"', page)[1]
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(base + "/api/state")
        assert denied.value.code == 403
        state_request = urllib.request.Request(base + "/api/state", headers={"X-Expo-Token": token})
        assert json.loads(urllib.request.urlopen(state_request).read())["phase"] == "idle"
        request = urllib.request.Request(
            base + "/api/command/settings",
            data=json.dumps({"episode_steps": 16, "admission_mode": "auto"}).encode(),
            headers={"X-Expo-Token": token, "Origin": base, "Content-Type": "application/json"},
        )
        assert urllib.request.urlopen(request).status == 200
        assert controller.config["episode_steps"] == 16
        assert controller.config["admission_mode"] == "auto"
        with pytest.raises(ValueError):
            controller.settings({"episode_steps": True})
        with pytest.raises(ValueError):
            controller.settings({"workspace_m": [[0, 1]] * 3})
        with pytest.raises(ValueError):
            create_server(controller, "0.0.0.0")
    finally:
        controller.stop()
        server.shutdown()
        server.server_close()


def test_stop_while_waiting_for_reset_never_enables(tmp_path):
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    controller.start(demo=True)
    wait_for(lambda: controller.phase == "awaiting_reset")
    controller.stop()
    controller.worker.join(2)
    assert not controller.worker.is_alive()
    assert controller.env.closed.is_set() and controller.env.device.stopped
    assert controller.recorder is None
    with pytest.raises(RuntimeError, match="关闭"):
        controller.env.reset_confirm = None
        controller.env.reset()


def test_digest_mismatch_aborts_before_review(tmp_path):
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    controller.config["episode_steps"] = 1
    controller.start(demo=True)
    try:
        wait_for(lambda: controller.phase == "awaiting_reset")
        controller.command("ready", {})
        wait_for(lambda: controller.phase == "review_pending")
        with pytest.raises(RuntimeError, match="SHA256"):
            controller.handle(
                {"operation": "review_episode", "env_id": controller.env_id, "steps": 1, "digest": "wrong"}
            )
        assert controller.recorder.meta["status"] == "review_pending"
    finally:
        controller.stop()
        controller.worker.join(2)


def test_restore_ignores_uncommitted_staging(tmp_path):
    pending = tmp_path / "reviewed_episodes" / (".pending_" + "a" * 32)
    pending.mkdir(parents=True)
    (pending / "metadata.json").write_text(json.dumps({"admitted": True, "end_step": 1}))
    processor = Processor()
    restore_reviewed(processor, tmp_path, 2)
    assert processor.transitions == [] and pending.exists()
