"""核对真实checkpoint形状、相机槽位、回放与动作坐标适配；不加载完整模型或连接机器人。"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/expo"))


def main():
    import h5py
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    from flax import nnx, traverse_util
    from openpi import transforms
    from openpi.training.config import AssetsConfig
    from piperx_expo.config import load_config, resolve_path
    from piperx_expo.server.launch import validate
    from piperx_expo.server.openpi_config import register
    from piperx_expo.server.checkpoint_adapter import ActionCoordinates, DeltaCheckpointAgent
    import dataclasses

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server/rotating_square.h30.yaml")
    parser.add_argument("--infer-weights", help="可选：实际加载此params目录并做离线前缀推理，不访问机器人")
    args = parser.parse_args()
    experiment = load_config(args.config)
    os.environ["PIPERX_EXPERIMENT_CONFIG"] = str(resolve_path(args.config))
    validate(experiment, "sft")
    train = register(experiment)
    train = dataclasses.replace(train, data=dataclasses.replace(train.data, assets=AssetsConfig(
        assets_dir=experiment["assets_dir"], asset_id=experiment["asset_id"])))
    data = train.data.create(train.assets_dirs, train.model)
    assets = resolve_path(experiment["assets_dir"]) / experiment["asset_id"]
    absolute = json.loads((assets / "norm_stats.json").read_text())["norm_stats"]
    vla = json.loads((assets / "vla_norm_stats.json").read_text())["norm_stats"]
    bridge = ActionCoordinates(absolute, vla)
    with h5py.File(resolve_path(experiment["dataset_path"]) / "0/traj.hdf5") as file:
        state = np.r_[file["saved_observation/cartesian_position"][0],
                      file["saved_observation/gripper_position"][0]].astype(np.float32)
        actions = np.c_[file["action/cartesian_position"][:30], file["action/gripper_position"][:30]]
        raw = {"cartesian_position": state[:6], "gripper_position": state[6:], "actions": actions.copy(),
               "exterior_image_1_left": file["saved_observation/exterior_image_1_left"][0],
               "wrist_image_left": file["saved_observation/wrist_image_left"][0],
               "prompt": experiment["language_instruction"]}
    raw["exterior_image_2_left"] = raw["exterior_image_1_left"].copy()
    pipeline = transforms.compose([*data.repack_transforms.inputs, *data.data_transforms.inputs])
    observation = pipeline(raw)
    assert observation["image_mask"]["right_wrist_0_rgb"]
    assert not observation["image_mask"]["left_wrist_0_rgb"]
    np.testing.assert_array_equal(observation["image"]["right_wrist_0_rgb"], raw["wrist_image_left"])
    np.testing.assert_allclose(observation["actions"], actions)
    state_norm = bridge.normalize(state[None], vla["state"])
    abs_norm = bridge.normalize(actions[None], absolute["actions"])
    delta_norm = bridge.convert(abs_norm, state_norm, to_vla=True)
    physical_delta = np.asarray(bridge.physical(delta_norm, vla["actions"]))
    expected = actions.copy()
    expected[:, :6] -= state[:6]
    np.testing.assert_allclose(physical_delta[0], expected, atol=2e-6)
    np.testing.assert_allclose(bridge.convert(delta_norm, state_norm, to_vla=False), abs_norm, atol=2e-5)
    # 实际actor batch必须将所有30帧锚定到同一个state。
    actor = object.__new__(DeltaCheckpointAgent)
    actor.model_config, actor.action_dim, actor.coordinates = train.model, 7, bridge
    normalized_obs = transforms.Normalize(data.norm_stats, use_quantiles=True)(observation)
    modeled = transforms.compose(data.model_transforms.inputs)(normalized_obs)
    from openpi.models.model import Observation
    batch = Observation.from_dict({k: jax.tree.map(lambda x: np.asarray(x)[None], value)
                                   for k, value in modeled.items() if k != "actions"}).to_dict()
    batch["full_actions"] = abs_norm.reshape(1, -1)
    _, targets = actor.prepare_batch_for_actor(batch)
    np.testing.assert_allclose(np.asarray(targets)[..., :7], delta_norm, atol=2e-5)
    # 下一次观测位置变化时，同一绝对前缀必须用新的state重算delta。
    moved = state.copy()
    moved[0] += 0.01
    moved_norm = bridge.normalize(moved[None], vla["state"])
    prefix = actor._prefix(abs_norm[:, :5], {"state": moved_norm})
    expected_prefix = expected[:5].copy()
    expected_prefix[:, 0] -= 0.01
    np.testing.assert_allclose(bridge.physical(prefix[..., :7], vla["actions"])[0], expected_prefix, atol=2e-6)

    # 只构造抽象参数；不把约34亿参数实际分配到GPU。
    reference = jax.eval_shape(lambda: nnx.state(train.model.create(jax.random.key(42))).to_pure_dict())
    reference = traverse_util.flatten_dict(reference)
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(resolve_path(experiment["base_weights"]))
    loaded = traverse_util.flatten_dict(metadata["params"])
    if all(key[-1] == "value" for key in loaded):
        loaded = {key[:-1]: value for key, value in loaded.items()}
    missing = set(reference) - set(loaded)
    mismatched = ["/".join(key) for key in set(reference) & set(loaded)
                  if reference[key].shape != tuple(loaded[key].shape)]
    if missing or mismatched:
        raise ValueError(f"checkpoint不兼容：missing={missing}, shape_mismatch={mismatched}")
    report = {"result": "pass", "checkpoint": experiment["base_weights"],
              "action_horizon": train.model.action_horizon, "checkpoint_parameter_leaves": len(loaded),
              "right_wrist": True, "absolute_delta_roundtrip": True, "rtc_prefix_reanchor": True,
              "scope": "真实checkpoint元数据形状、真实示教变换；未完整加载/训练VLA，未执行真机"}
    if args.infer_weights:
        import time
        import jax.numpy as jnp
        from expo_ft.utils.train_utils import FlexibleCheckpointWeightLoader
        from openpi.training import sharding

        train = dataclasses.replace(train, weight_loader=FlexibleCheckpointWeightLoader(
            str(resolve_path(args.infer_weights))))
        mesh = sharding.make_mesh(1)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        real_actor, state, _ = DeltaCheckpointAgent.initialize(
            train, mesh, jax.random.PRNGKey(42), resume=False,
            default_prompt=experiment["language_instruction"], data_sharding=data_sharding,
            replicated_sharding=replicated, freeze_pi05_encoder=True,
        )
        real_actor.action_dim = real_actor.state_dim = 7
        inputs = real_actor.process_raw_inputs(raw.copy(), 7, 224)
        start = time.monotonic()
        sampled, _ = real_actor.sample_actions(inputs, state, jax.random.PRNGKey(43), num_samples=1)
        jax.block_until_ready(sampled)
        sampled_physical = real_actor.process_transformed_outputs(sampled)
        assert sampled_physical.shape == (1, 30, 7) and np.isfinite(sampled_physical).all()
        prefix = jnp.pad(sampled[0, :5], ((0, 0), (0, 25)))
        conditioned = real_actor.sample_actions_with_prefix(inputs, state, jax.random.PRNGKey(44), prefix)
        conditioned_physical = real_actor.process_transformed_outputs(conditioned)
        np.testing.assert_allclose(conditioned_physical[:, :5], sampled_physical[:, :5], atol=2e-5)
        report.update(inference_weights=args.infer_weights, real_inference_shape=list(sampled_physical.shape),
                      real_prefix_preserved=True, cold_two_inferences_s=time.monotonic() - start,
                      scope="完整VLA权重加载与离线普通/前缀推理通过；单候选含编译时间，非实时延迟/真机验证")
    output = ROOT / "runtime/validation/checkpoint_compatibility.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
