"""仅在 VLA 边界转换 delta；EXPO replay、critic、edit 和 RTC 前缀始终用绝对动作。"""

import json

import jax.numpy as jnp
from expo_ft.agents.vla import pi05

from piperx_expo.config import resolve_path

BasePi05Agent = pi05.Pi05Agent


class ActionCoordinates:
    def __init__(self, absolute_stats, vla_stats):
        self.absolute = absolute_stats
        self.vla = vla_stats

    @staticmethod
    def normalize(value, stats):
        low, high = jnp.asarray(stats["q01"][:7]), jnp.asarray(stats["q99"][:7])
        return (value - low) / (high - low + 1e-6) * 2 - 1

    @staticmethod
    def physical(value, stats):
        low, high = jnp.asarray(stats["q01"][:7]), jnp.asarray(stats["q99"][:7])
        return (value + 1) / 2 * (high - low + 1e-6) + low

    def convert(self, actions, state, *, to_vla):
        # state 使用原checkpoint统计；每条chunk全部相对同一观测，不能逐帧更换参考系。
        state = jnp.asarray(state)[..., :7]
        state = state.reshape(-1, 7)
        actions = jnp.asarray(actions)[..., :7]
        repeats = actions.shape[0] // state.shape[0]
        if actions.shape[0] % state.shape[0]:
            raise ValueError("动作候选与观测batch不一致")
        anchor = jnp.repeat(self.physical(state, self.vla["state"]), repeats, axis=0)[:, None, :]
        if to_vla:
            physical = self.physical(actions, self.absolute["actions"])
            physical = physical.at[..., :6].add(-anchor[..., :6])
            return self.normalize(physical, self.vla["actions"])
        physical = self.physical(actions, self.vla["actions"])
        physical = physical.at[..., :6].add(anchor[..., :6])
        return self.normalize(physical, self.absolute["actions"])


class DeltaCheckpointAgent(BasePi05Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from piperx_expo.config import load_config
        import os

        experiment = load_config(os.environ["PIPERX_EXPERIMENT_CONFIG"])
        root = resolve_path(experiment["assets_dir"]) / experiment["asset_id"]
        absolute = json.loads((root / "norm_stats.json").read_text())["norm_stats"]
        vla = json.loads((root / "vla_norm_stats.json").read_text())["norm_stats"]
        self.coordinates = ActionCoordinates(absolute, vla)

    def _prefix(self, prefix, inputs):
        value = self.coordinates.convert(prefix, inputs["state"], to_vla=True)
        return jnp.pad(value, ((0, 0), (0, 0), (0, self.model_config.action_dim - 7)))

    def sample_actions(self, transformed_inputs, *args, **kwargs):
        actions, timing = super().sample_actions(transformed_inputs, *args, **kwargs)
        return self.coordinates.convert(actions, transformed_inputs["state"], to_vla=False), timing

    def sample_actions_with_prefix(self, transformed_inputs, train_state, rng, prefix_padded, num_samples=1):
        prefix = self._prefix(jnp.asarray(prefix_padded)[None], transformed_inputs)[0]
        actions = super().sample_actions_with_prefix(transformed_inputs, train_state, rng, prefix, num_samples)
        return self.coordinates.convert(actions, transformed_inputs["state"], to_vla=False)

    def sample_training_actions(self, transformed_inputs, *args, **kwargs):
        actions, timing = super().sample_training_actions(transformed_inputs, *args, **kwargs)
        return self.coordinates.convert(actions, transformed_inputs["state"], to_vla=False), timing

    def sample_training_actions_with_prefix(self, transformed_inputs, train_state, rng, prefix_padded,
                                            num_samples=1, noise=None):
        prefix = self._prefix(prefix_padded, transformed_inputs)
        actions, timing = super().sample_training_actions_with_prefix(
            transformed_inputs, train_state, rng, prefix, num_samples, noise,
        )
        return self.coordinates.convert(actions, transformed_inputs["state"], to_vla=False), timing

    def prepare_batch_for_actor(self, batch):
        actions = batch["full_actions"].reshape(-1, self.model_config.action_horizon, 7)
        actions = self.coordinates.convert(actions, batch["state"], to_vla=True)
        return super().prepare_batch_for_actor({**batch, "full_actions": actions.reshape(actions.shape[0], -1)})


def install():
    # 只替换本次启动进程的适配类，不修改固定提交的上游源码或训练公式。
    pi05.Pi05Agent = DeltaCheckpointAgent
