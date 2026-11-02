import json
import os
import shutil

import pytest
import yaml

from piperx_expo.server.model_store import checkpoint_info, inference_config, register_model, verify_model


def saved_model(tmp_path):
    root = tmp_path / "runs/example_rl/checkpoints"
    for item in ("agent", "params"):
        folder = root / "419" / item
        folder.mkdir(parents=True)
        (folder / "_METADATA").write_text("{}")
        (folder / "weights").write_bytes(b"immutable weight shard")
    (root / "419/_CHECKPOINT_METADATA").write_text('{"committed":true}')
    (root / "review_state_419.json").write_text('{"credit":0,"update_count":0}')
    assets = tmp_path / "assets/piperx/test"
    assets.mkdir(parents=True)
    for name in ("norm_stats", "vla_norm_stats", "provenance"):
        (assets / f"{name}.json").write_text("{}")
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    config = {"output_dir": str(tmp_path / "runs"), "run_name": "example", "episode_review": True,
              "max_steps": 4000, "assets_dir": str(tmp_path / "assets"), "asset_id": "piperx/test",
              "dataset_path": str(data), "rtc_weights": "/original/rtc/params"}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    return root, path, config


def test_registration_retains_model_after_source_removal_and_pins_assets(tmp_path):
    root, path, config = saved_model(tmp_path)
    store = tmp_path / "models"
    result = register_model(path, root, 419, store, "gui_step419")
    folder = store / "gui_step419"
    source_shard = root / "419/params/weights"
    pinned = folder / "checkpoints/419/params/weights"
    assert os.stat(source_shard).st_ino == os.stat(pinned).st_ino
    usage = yaml.safe_load((folder / "inference.yaml").read_text())
    assert usage["inference_checkpoint_step"] == 419
    assert usage["inference_checkpoint_dir"] == str(folder / "checkpoints")
    assert usage["assets_dir"] == str(folder / "assets")
    assert result["review_state"]["update_count"] == 0
    shutil.rmtree(root / "419")
    assert verify_model(folder)["step"] == 419 and pinned.read_bytes() == b"immutable weight shard"
    (folder / "assets/piperx/test/norm_stats.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="哈希"):
        verify_model(folder)


def test_partial_or_parameter_only_model_cannot_be_used(tmp_path):
    root, _, _ = saved_model(tmp_path)
    (root / "500").mkdir()
    with pytest.raises(ValueError, match="提交元数据"):
        checkpoint_info(root)
    assert checkpoint_info(root, 419)["step"] == 419
    shutil.rmtree(root / "419/agent")
    with pytest.raises(ValueError, match="agent"):
        checkpoint_info(root, 419)


def test_registration_never_overwrites_existing_model_or_uses_traversal(tmp_path):
    root, path, _ = saved_model(tmp_path)
    store = tmp_path / "models"
    register_model(path, root, 419, store, "fixed")
    with pytest.raises(FileExistsError):
        register_model(path, root, 419, store, "fixed")
    with pytest.raises(ValueError, match="模型名"):
        register_model(path, root, 419, store, "../escape")
    assert not list(store.glob(".pending_*"))


def test_rtc_sft_archive_is_initialization_not_full_rl_inference(tmp_path):
    root, path, _ = saved_model(tmp_path)
    store = tmp_path / "models"
    register_model(path, root, 419, store, "rtc_baseline", kind="rtc_sft")
    assert not (store / "rtc_baseline/inference.yaml").exists()
    config = yaml.safe_load((store / "rtc_baseline/initialization.yaml").read_text())
    assert config["rtc_weights"].endswith("rtc_baseline/checkpoints/419/params")


def test_inference_budget_is_independent_of_source_training_step(tmp_path):
    root, _, config = saved_model(tmp_path)
    value = inference_config(config, root, 419, max_steps=5)
    assert value["max_steps"] == 5 and config["max_steps"] == 4000
    assert value["output_dir"] == str(tmp_path / "evaluations")
    assert value["inference_checkpoint_step"] == 419
    assert not value["checkpoint_model"]
    with pytest.raises(ValueError):
        inference_config(config, root, 419, max_steps=True)
    assert json.loads((root / "review_state_419.json").read_text())["update_count"] == 0


def test_failed_link_cleans_only_own_staging_and_keeps_source(tmp_path, monkeypatch):
    root, path, _ = saved_model(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("cross device link")

    monkeypatch.setattr(os, "link", fail)
    store = tmp_path / "models"
    with pytest.raises(OSError):
        register_model(path, root, 419, store, "failed")
    assert not list(store.iterdir())
    assert (root / "419/params/weights").read_bytes() == b"immutable weight shard"


def test_modified_usage_config_is_rejected(tmp_path):
    root, path, _ = saved_model(tmp_path)
    store = tmp_path / "models"
    register_model(path, root, 419, store, "fixed")
    (store / "fixed/inference.yaml").write_text("inference_only: false\n")
    with pytest.raises(ValueError, match="配置哈希"):
        verify_model(store / "fixed")
