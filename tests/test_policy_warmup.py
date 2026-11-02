import json

import numpy as np
import pytest

from piperx_expo.server.policy_warmup import warmup_policy
from piperx_expo.server.progress import TrainingProgress


class Sampler:
    delay = 5

    def __init__(self):
        self.prefix = False
        self.pending = False
        self.boots = self.rtc = self.ended = 0

    def observe(self, obs):
        pass

    def launch(self, agent, obs, plan_len, kind):
        if self.prefix and plan_len == self.delay:
            self.pending = True

    def sample(self, agent, obs, metrics):
        if not self.prefix:
            self.boots += 1
        else:
            assert self.pending, "RTC路径必须通过pending/prefix实际预热"
            self.rtc += 1
        self.prefix, self.pending = True, False
        return np.zeros((8, 7)), agent + 1, {}

    def on_episode_end(self):
        self.ended += 1
        self.prefix = self.pending = False


def test_warmup_compiles_both_paths_and_discards_all_prefixes():
    sampler = Sampler()
    agent = 0
    stages = []
    timing = warmup_policy(sampler, agent, {}, samples=3, on_stage=stages.append)
    assert sampler.boots == 1 and sampler.rtc == 4 and sampler.ended == 1
    assert not sampler.prefix and not sampler.pending and agent == 0
    assert timing["samples"] == 3 and timing["steady_max_ms"] >= timing["steady_p95_ms"]
    assert stages[:2] == ["warmup_boot", "warmup_rtc"]


def test_bad_warmup_aborts_and_still_clears_prefix():
    sampler = Sampler()
    sampler.sample = lambda *args: (np.full((8, 7), np.nan), 0, {})
    with pytest.raises(ValueError, match="预热"):
        warmup_policy(sampler, 0, {})
    assert sampler.ended == 1 and not sampler.prefix


def test_structured_progress_preserves_saved_step_and_original_error(tmp_path):
    target = tmp_path / "training_status.json"
    progress = TrainingProgress(target)
    progress.update("saved", checkpoint_step=80)
    progress.update("failed", error="动作看门狗超时")
    value = json.loads(target.read_text())
    assert value["stage"] == "failed" and value["error"] == "动作看门狗超时"
    assert value["checkpoint_step"] == 80
