"""验证真实 GPU 栈、上游回放/变换与 RTC 预取；不加载完整 pi0.5 权重。"""

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/expo"))


def main():
    import h5py
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from ml_collections import ConfigDict

    # 导入实际算法，暴露依赖版本或上游 API 不兼容，不启动学习循环。
    from expo_ft.agents.alg import realtime_expo_ft, rtc
    from expo_ft.env.droid_utils import process_droid_dataset
    from expo_ft.data.replay_buffer import create_replay_buffer
    from expo_ft.utils.loop_utils import AsyncChunkSampler
    from piperx_expo.data.norm import compute_norm
    from piperx_expo.server.openpi_config import CONFIG_NAME, register
    from configs.model.rtc_pi_config import get_config

    register()
    devices = jax.devices()
    assert any(device.platform == "gpu" for device in devices), devices
    params = jnp.ones((8, 8))
    optimizer = optax.adamw(1e-3)
    state = optimizer.init(params)

    @jax.jit
    def update(params, state):
        loss, grad = jax.value_and_grad(lambda p: jnp.mean((p @ p) ** 2))(params)
        updates, state = optimizer.update(grad, state, params)
        return optax.apply_updates(params, updates), state, loss

    before = float(jnp.mean((params @ params) ** 2))
    params, state, loss = update(params, state)
    after = float(jnp.mean((params @ params) ** 2))
    assert after < before
    with tempfile.TemporaryDirectory(prefix="piperx-stack-") as temporary:
        root = Path(temporary)
        dataset = root / "replay"
        episode = dataset / "0"
        episode.mkdir(parents=True)
        n = 20
        with h5py.File(episode / "traj.hdf5", "w") as file:
            file["saved_observation/cartesian_position"] = np.zeros((n, 6), dtype=np.float32)
            file["saved_observation/gripper_position"] = np.full((n, 1), 0.04, dtype=np.float32)
            file.create_dataset(
                "saved_observation/prompt",
                data=["grab the rotating red square"] * n,
                dtype=h5py.string_dtype(),
            )
            for name in ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"):
                file[f"saved_observation/{name}"] = np.full((n, 224, 224, 3), 128, dtype=np.uint8)
            file["action/cartesian_position"] = np.zeros((n, 6), dtype=np.float32)
            file["action/gripper_position"] = np.full(n, 0.04, dtype=np.float32)
        (dataset / "manifest.json").write_text("{}")
        compute_norm(dataset, root / "assets/test")
        task = ConfigDict({"action_space": "cartesian_position", "gripper_action_space": "position"})
        transitions = process_droid_dataset(str(dataset), task)
        assert len(transitions) == n and transitions[-1]["rewards"] == 1
        config = get_config()
        config.pi05_config_name = CONFIG_NAME
        config.pi05_assets_dir = str(root / "assets")
        config.pi05_asset_id = "test"
        config.valids_keep_terminal_windows = True
        replay = create_replay_buffer(
            config, np.zeros((1, 7)), n, "grab the rotating red square", 8, 42, delay=5
        )
        replay.insert_dataset(transitions)
        assert len(replay) == n
        assert replay.dataset_dict["state"].shape == (n, 32)
        batch = replay.sample_jax(2)
        assert batch is not None

    @dataclasses.dataclass
    class FakeAgent:
        rng: int = 0

        def replace(self, **kwargs):
            return dataclasses.replace(self, **kwargs)

        def sample_pre_cache(self, obs, prefix_padded=None):
            if prefix_padded is not None:
                assert prefix_padded.shape == (5, 32)
            return jnp.ones((16, 32)), self.replace(rng=self.rng + 1)

        def sample_actions(self, obs, precached=None, delay=0):
            return np.ones((8, 7)), self, {"executed_padded": jnp.ones((8, 32))}

    agent = FakeAgent()
    sampler = AsyncChunkSampler(agent, delay=5, replan_steps=8)
    try:
        _, agent, _ = sampler.sample(agent, {}, {})
        sampler.launch(agent, {}, 5, "policy")
        assert sampler.pending is not None
        _, agent, _ = sampler.sample(agent, {}, {})
        assert agent.rng == 2
        sampler.on_episode_end()
        assert sampler.prefix_cache is None
    finally:
        sampler.close()
    report = {
        "result": "pass",
        "jax_devices": [str(d) for d in devices],
        "gpu_optimizer_loss_before": before,
        "gpu_optimizer_loss_after": after,
        "upstream_replay_frames": n,
        "rtc_delay": 5,
        "replan_steps": 8,
        "algorithms_imported": [rtc.__name__, realtime_expo_ft.__name__],
        "scope": "GPU算子/优化器、真实回放变换、RTC控制流；非完整pi0.5训练",
    }
    output = ROOT / "runtime/validation/server_stack.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
