import json
import sys
import types

import pytest

from piperx_expo.server.launch import _install_checkpoint_init, _install_local_log, _load_wandb_environment


def test_wandb_credentials_are_scoped_assignments(monkeypatch, tmp_path):
    for key in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_MODE", "WANDB_BASE_URL",
                "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "wandb.env"
    marker = tmp_path / "must-not-execute"
    path.write_text(f"export WANDB_API_KEY='test-credential'\nexport WANDB_ENTITY=old-team\n"
                    f"export WANDB_MODE=offline\ntouch {marker}\n")
    config = {"wandb_mode": "online", "wandb_env_file": str(path), "wandb_entity": "requested-team",
              "wandb_base_url": "https://api.wandb.ai", "wandb_proxy_url": "http://127.0.0.1:17890"}
    _load_wandb_environment(config)
    import os
    assert os.environ["WANDB_API_KEY"] == "test-credential"
    assert os.environ["WANDB_ENTITY"] == "requested-team"
    assert os.environ["WANDB_BASE_URL"] == "https://api.wandb.ai"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert os.environ[key] == "http://127.0.0.1:17890"
    assert "WANDB_MODE" not in os.environ and not marker.exists()
    monkeypatch.setenv("WANDB_API_KEY", "existing-credential")
    _load_wandb_environment(config)
    assert os.environ["WANDB_API_KEY"] == "existing-credential"


def test_offline_wandb_does_not_require_credential_file():
    _load_wandb_environment({"wandb_mode": "offline", "wandb_env_file": "/missing/wandb.env"})


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
