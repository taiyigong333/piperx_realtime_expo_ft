import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from piperx_expo.client.gui_control import GUIController
from piperx_expo.config import load_config
from piperx_expo.server.gui_service import ServiceManager
from piperx_expo.server.storage import CheckpointStorage
from piperx_expo.trash import RecycleBin


def test_episode_delete_restore_and_permanent_delete_keep_server_data(tmp_path):
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path / "episodes")
    episode_id = "a" * 32
    folder = controller.root / episode_id
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps({"id": episode_id, "status": "accepted", "steps": 3}))
    (folder / "episode.hdf5").write_bytes(b"local raw trajectory")
    experience = tmp_path / "server/transitions.pkl"
    experience.parent.mkdir()
    experience.write_bytes(b"server replay must remain unchanged")
    try:
        controller.command("delete_episode", {"episode_id": episode_id, "confirmed": True})
        assert not folder.exists() and controller.history() == []
        record = controller.recycle.records()[0]
        controller.command("restore_episode", {"trash_id": record["id"]})
        assert (folder / "episode.hdf5").read_bytes() == b"local raw trajectory"
        controller.command("delete_episode", {"episode_id": episode_id, "confirmed": True})
        record = controller.recycle.records()[0]
        with pytest.raises(ValueError, match="确认"):
            controller.command("purge_episode", {"trash_id": record["id"], "confirmation": "wrong"})
        controller.command("purge_episode", {"trash_id": record["id"], "confirmation": record["id"]})
        assert not controller.recycle.records() and not folder.exists()
        audit = json.loads((controller.recycle.entry(record["id"]) / "receipt.json").read_text())
        assert audit["status"] == "purged" and "purged_at" in audit
        assert experience.read_bytes() == b"server replay must remain unchanged"
    finally:
        controller.stop()


@pytest.mark.parametrize("status", ["recording", "review_pending", "submitted", "queued"])
def test_unfinished_episode_cannot_be_deleted(tmp_path, status):
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    episode_id = "a" * 32
    folder = tmp_path / episode_id
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps({"status": status, "steps": 2}))
    try:
        with pytest.raises(ValueError, match="不能删除"):
            controller.command("delete_episode", {"episode_id": episode_id, "confirmed": True})
        assert folder.exists()
    finally:
        controller.stop()


def test_active_motion_blocks_episode_management(tmp_path):
    controller = GUIController(load_config("configs/client/gui.mock.yaml"), tmp_path)
    controller.worker = SimpleNamespace(is_alive=lambda: True)
    try:
        with pytest.raises(ValueError, match="结束当前会话"):
            controller.command("delete_episode", {"episode_id": "a" * 32, "confirmed": True})
    finally:
        controller.worker = None
        controller.stop()


def test_recycle_bin_rejects_traversal_symlinks_and_overwriting_restore(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    recycle = RecycleBin(root, r"[0-9a-f]{32}")
    with pytest.raises(ValueError):
        recycle.put(["../other"])
    other = tmp_path / "other"
    other.mkdir()
    (root / ("b" * 32)).symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        recycle.put(["b" * 32])
    name = "a" * 32
    (root / name).mkdir()
    record = recycle.put([name])
    (root / name).mkdir()
    with pytest.raises(FileExistsError):
        recycle.restore(record["id"])
    assert (recycle.entry(record["id"]) / "payload" / name).exists()
    assert other.exists()


def test_group_move_failure_rolls_back_without_losing_checkpoint_state(tmp_path, monkeypatch):
    recycle = RecycleBin(tmp_path, r"80|review_state_80\.json")
    (tmp_path / "80").mkdir()
    (tmp_path / "review_state_80.json").write_text("state")
    original = Path.rename

    def rename(path, target):
        if path == tmp_path / "review_state_80.json":
            raise OSError("simulated move failure")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(OSError):
        recycle.put(["80", "review_state_80.json"])
    assert (tmp_path / "80").exists()
    assert (tmp_path / "review_state_80.json").read_text() == "state"
    assert not recycle.records()


def saved_run(tmp_path):
    manager = ServiceManager(tmp_path / "code")
    run_id = "a" * 32
    folder = manager.runtime / run_id
    folder.mkdir()
    config = {"episode_review": True, "run_name": "example", "output_dir": str(tmp_path / "runs"), "max_steps": 100}
    (folder / "process.json").write_text(json.dumps({"id": run_id, "run_name": "example", "started_at": 1}))
    (folder / "experiment.yaml").write_text(yaml.safe_dump(config))
    root = tmp_path / "runs/example_rl/checkpoints"
    for step in (80, 90):
        for part in ("agent", "params"):
            path = root / str(step) / part
            path.mkdir(parents=True)
            (path / "_METADATA").write_text("{}")
        (root / str(step) / "_CHECKPOINT_METADATA").write_text("{}")
        (root / f"review_state_{step}.json").write_text('{"credit":0,"update_count":2}')
    (root / "wandb_id.txt").write_text("same-wandb")
    experience = root / "reviewed_episodes/episode/transitions.pkl"
    experience.parent.mkdir(parents=True)
    experience.write_bytes(b"keep training data")
    return manager, CheckpointStorage(manager), run_id, root, folder / "experiment.yaml"


def test_checkpoint_delete_moves_matching_state_and_changes_resume_boundary(tmp_path):
    manager, storage, run_id, root, _ = saved_run(tmp_path)
    assert manager.resume_info(run_id)[1] == 90
    with pytest.raises(ValueError, match="确认"):
        storage.delete(run_id, "90", False)
    record = storage.delete(run_id, "90", True)
    assert not (root / "90").exists() and not (root / "review_state_90.json").exists()
    assert manager.resume_info(run_id)[1] == 80
    assert (root / "reviewed_episodes/episode/transitions.pkl").read_bytes() == b"keep training data"
    earlier = storage.delete(run_id, "80", True)
    assert not manager.runs()[0]["resumable"]
    storage.restore(run_id, earlier["id"])
    storage.restore(run_id, record["id"])
    assert manager.resume_info(run_id)[1] == 90
    assert (root / "review_state_90.json").is_file()


def test_checkpoint_mutations_are_rejected_while_owned_service_is_active(tmp_path, monkeypatch):
    _, storage, run_id, root, _ = saved_run(tmp_path)
    monkeypatch.setattr("piperx_expo.server.gui_service.owned", lambda _: True)
    with pytest.raises(RuntimeError, match="先结束"):
        storage.delete(run_id, "90", True)
    assert (root / "90").exists()
    assert storage.catalog()["blocked_reason"]


def test_independent_service_on_configured_model_port_blocks_deletion(tmp_path, monkeypatch):
    _, storage, run_id, root, path = saved_run(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["port"] = 8103
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr("piperx_expo.server.gui_service.listening", lambda _: True)
    with pytest.raises(RuntimeError, match="端口"):
        storage.delete(run_id, "90", True)
    assert (root / "90").exists()


def test_management_rejects_path_and_auxiliary_state_deletion(tmp_path):
    _, storage, run_id, root, _ = saved_run(tmp_path)
    for item in ("../90", "review_state_90.json", "reviewed_episodes"):
        with pytest.raises(ValueError):
            storage.delete(run_id, item, True)
    with pytest.raises(ValueError):
        storage.delete("/arbitrary/path", "90", True)
    assert (root / "review_state_90.json").exists()


def test_config_referenced_rtc_model_is_protected_but_older_point_can_be_removed(tmp_path):
    _, storage, run_id, root, config_path = saved_run(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["rtc_weights"] = str(root / "90/params")
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="引用"):
        storage.delete(run_id, "90", True)
    record = storage.delete(run_id, "80", True)
    storage.purge(run_id, record["id"], record["id"])
    assert not (root / "80").exists() and (root / "90").exists()


def test_deleting_original_checkpoint_keeps_hardlinked_named_version(tmp_path):
    _, storage, run_id, root, _ = saved_run(tmp_path)
    source = root / "90/params/weights"
    source.write_bytes(b"shared immutable weights")
    release = tmp_path / "models/releases/fixed90"
    release.mkdir(parents=True)
    (release / "model.json").write_text('{"kind":"rl","step":90}')
    import os
    os.link(source, release / "weights")
    record = storage.delete(run_id, "90", True)
    storage.purge(run_id, record["id"], record["id"])
    assert (release / "weights").read_bytes() == b"shared immutable weights"
    row = next(r for r in storage.catalog()["checkpoints"] if r["kind"] == "release")
    record = storage.delete(row["source_id"], row["item_id"], True)
    assert not release.exists()
    storage.restore(row["source_id"], record["id"])
    assert (release / "weights").is_file()
