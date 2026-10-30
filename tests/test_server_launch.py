import json
import sys
import types

import pytest

from piperx_expo.server.launch import _install_checkpoint_init, _install_local_log


def test_local_metrics_hook_survives_wandb_init(monkeypatch, tmp_path):
    calls = []
    wandb = types.ModuleType("wandb")

    def initialize():
        wandb.log = lambda *args, **kwargs: calls.append((args, kwargs))
        return "run"

    wandb.init, wandb.log = initialize, None
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    _install_local_log({"output_dir": str(tmp_path), "run_name": "test"}, "sft")
    assert wandb.init() == "run"
    wandb.log({"training/actor_loss": 0.25, "training/actor_lr": 2.5e-5}, step=1)
    entry = json.loads((tmp_path / "test_sft/metrics.jsonl").read_text())
    assert entry["step"] == 1 and entry["training/actor_loss"] == 0.25
    assert len(calls) == 1


def test_new_empty_checkpoint_allowed_existing_data_preserved(monkeypatch, tmp_path):
    package, agents = types.ModuleType("expo_ft"), types.ModuleType("expo_ft.agents")

    def original(path, **kwargs):
        if path.exists():
            raise FileExistsError(path)
        path.mkdir()
        return "initialized"

    agents.initialize_checkpoint_dir = original
    package.agents = agents
    monkeypatch.setitem(sys.modules, "expo_ft", package)
    monkeypatch.setitem(sys.modules, "expo_ft.agents", agents)
    _install_checkpoint_init()
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    assert agents.initialize_checkpoint_dir(directory, keep_period=None, overwrite=False, resume=False) == "initialized"
    checkpoint = directory / "params"
    checkpoint.write_text("must survive")
    with pytest.raises(FileExistsError):
        agents.initialize_checkpoint_dir(directory, keep_period=None, overwrite=False, resume=False)
    assert checkpoint.read_text() == "must survive"
