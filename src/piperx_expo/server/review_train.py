"""独立的浏览器复核驱动；复用固定上游模型、RTC采样和梯度更新，不修改算法源码。"""

import json
import re
from pathlib import Path

import logging
import os

import numpy as np
from absl import app


import jax
import etils.epath as epath

import wandb
from expo_ft.agents import initialize_checkpoint_dir
from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
from expo_ft.data.replay_buffer import create_replay_buffer, _critic_key_to_storage
from expo_ft.data.batch_processor import BatchProcessor
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.utils.loop_utils import AsyncChunkSampler
from expo_ft.utils.timer import Timer
from expo_ft.utils.train_utils import (
    init_logging,
    init_wandb,
    set_compilation_cache_dir,
)
from expo_ft.agents.vla.pi05 import build_pi05

import openpi.training.sharding as openpi_sharding


# 上游模块声明原有absl参数，本驱动复用参数接口和模型初始化约定。
from train_pi_robo import FLAGS
from piperx_expo.client.recording import write_json
from piperx_expo.review import PROTOCOL, commit_episode
from piperx_expo.server.review_loop import collect_episode, persist_reviewed, restore_reviewed


def main(_):
    init_logging()
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    if FLAGS.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {FLAGS.batch_size} must be divisible by the number of devices {jax.device_count()}"
        )
    set_compilation_cache_dir(f"sync-{FLAGS.config.model_cls}")

    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name)
    os.makedirs(log_dir, exist_ok=True)
    train_video_dir = os.path.join(log_dir, "train_videos")
    os.makedirs(train_video_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_dir_path = epath.Path(checkpoint_dir)
    checkpoint_manager, resuming = initialize_checkpoint_dir(
        checkpoint_dir_path,
        keep_period=FLAGS.keep_period,
        overwrite=FLAGS.overwrite,
        resume=FLAGS.resume,
    )

    init_wandb(checkpoint_dir_path, resuming, FLAGS.project_name, FLAGS.run_name)
    wandb.config.update(FLAGS.flag_values_dict(), allow_val_change=resuming)

    if FLAGS.config_task.env_type == "droid":
        dataset = process_droid_dataset(
            FLAGS.dataset_path,
            FLAGS.config_task,
            num_data=FLAGS.num_data,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {FLAGS.config_task.env_type}")
    if not dataset:
        raise ValueError(f"No demos loaded from {FLAGS.dataset_path}.")
    example_action = dataset[0]["actions"][np.newaxis]

    model_cls = FLAGS.config.model_cls
    # BCLearner uses human-intervention chunks for the actor batch only (no critic).
    use_dagger_hil_sampling = model_cls == "BCLearner"
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "RealTimeEXPOFTLearner":
        from expo_ft.agents.alg.realtime_expo_ft import load_agent, restore_checkpoint, save_checkpoint
    else:
        raise ValueError(f"Unsupported model class: {model_cls}")

    # Same string run_client returns as task_description, without needing the env yet.
    task_description = FLAGS.config_task.language_instruction

    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        FLAGS.config,
        FLAGS.seed,
        mesh,
        data_sharding,
        replicated_sharding,
        resuming,
        task_description,
    )

    critic_camera_keys = tuple(getattr(FLAGS.config_task, "critic_camera_keys", CRITIC_CAMERA_KEYS))
    rb_args = dict(
        config=FLAGS.config,
        example_action=example_action,
        capacity=max(FLAGS.max_steps + 1, 2001),
        task_description=task_description,
        replan_steps=FLAGS.replan_steps,
        seed=FLAGS.seed,
        delay=FLAGS.delay,
        critic_camera_keys=critic_camera_keys,
    )
    replay_buffer = create_replay_buffer(**rb_args)
    offline_replay_buffer = create_replay_buffer(**rb_args)

    actor_success_only = getattr(FLAGS.config, "actor_success_only", False)
    batch_processor = BatchProcessor(
        replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        data_sharding=data_sharding,
        batch_size=FLAGS.batch_size,
        utd_ratio=FLAGS.utd_ratio,
        offline_ratio=FLAGS.offline_ratio,
        actor_success_only=actor_success_only,
        use_dagger_hil_sampling=use_dagger_hil_sampling,
        dataset=dataset,
    )

    critic_example = {
        _critic_key_to_storage(k): offline_replay_buffer.dataset_dict[_critic_key_to_storage(k)][0][
            np.newaxis
        ]
        for k in critic_camera_keys
    }
    critic_example["state"] = offline_replay_buffer.dataset_dict["state"][0][np.newaxis]
    critic_example["actions"] = offline_replay_buffer.dataset_dict["actions"][0][np.newaxis]
    agent_example_observation, agent_example_state, agent_example_action = (
        offline_replay_buffer.convert_to_critic_format(critic_example)
    )
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]

    if model_cls == "RealTimeEXPOFTLearner":
        agent_kwargs["delay"] = FLAGS.delay
    agent_kwargs["critic_camera_keys"] = critic_camera_keys

    agent = load_agent(
        seed=FLAGS.seed,
        example_observation=agent_example_observation.squeeze(),
        example_action=agent_example_action.squeeze(),
        example_state=agent_example_state.squeeze(),
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        agent_kwargs=agent_kwargs,
        metadata=vla_metadata,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        resume=resuming,
        replan_steps=FLAGS.replan_steps,
        default_prompt=task_description,
        edit_action_xyzg=FLAGS.config_task.edit_action_xyzg,
    )

    start_step = 0
    if resuming:
        agent = restore_checkpoint(checkpoint_manager, agent)
        if hasattr(agent, "cache_infer_params"):
            agent = agent.cache_infer_params()
        steps = tuple(checkpoint_manager.all_steps())
        latest_step = max(steps) if steps else None
        if latest_step is not None:
            start_step = latest_step
            logging.info("Resuming from step %d", start_step)
        restore_reviewed(batch_processor, checkpoint_dir_path, latest_step)

    # The env is created only now: EnvClientWrapper blocks until the rollout client
    # dials in, and the agent build above is the slow part.
    train_env_creation_request = {
        "example_action": example_action,
        "env_usage": "train",
        "video_dir": train_video_dir,
    }
    logging.info("Creating environment...")
    env = EnvClientWrapper(
        env_creation_request=train_env_creation_request, host=FLAGS.client_host, port=FLAGS.client_port
    )
    response = env.client.last_response
    if response.get("review_protocol") != PROTOCOL:
        raise ValueError("此服务器只接受支持逐条复核协议的浏览器客户端")
    if FLAGS.update_type != "episode" or FLAGS.step_interval < 1 or FLAGS.num_updates:
        raise ValueError("复核驱动使用episode更新和step_interval，不支持其他更新触发方式")
    audit_root = Path(checkpoint_dir) / "reviewed_episodes"
    accepted_episodes = sum(
        1
        for p in audit_root.glob("*/metadata.json")
        if re.fullmatch(r"[0-9a-f]{32}", p.parent.name)
        and json.loads(p.read_text()).get("admitted")
        and json.loads(p.read_text())["end_step"] <= start_step
    )
    sampler = AsyncChunkSampler(
        agent,
        delay=FLAGS.delay,
        replan_steps=FLAGS.replan_steps,
        sim_latency_ms=FLAGS.sim_latency,
        timer=Timer(),
    )
    combine_rng = jax.random.PRNGKey(FLAGS.seed + 100)
    step, credit, last_checkpoint, update_count = start_step, 0, start_step, 0
    if resuming:
        state_file = Path(checkpoint_dir) / f"review_state_{start_step}.json"
        if not state_file.is_file():
            raise ValueError("缺少复核训练状态，不能把原自动模式checkpoint作为GUI续训目录")
        state = json.loads(state_file.read_text())
        credit, update_count = state["credit"], state["update_count"]

    def save_training_checkpoint():
        # 先写同step辅助状态，再提交模型；多出的辅助文件不会被当作模型checkpoint。
        write_json(
            Path(checkpoint_dir) / f"review_state_{step}.json",
            {"credit": credit, "update_count": update_count},
        )
        save_checkpoint(checkpoint_manager, agent, step)
        checkpoint_manager.wait_until_finished()

    try:
        while step < FLAGS.max_steps:
            result, agent = collect_episode(
                env, sampler, agent, FLAGS.replan_steps, FLAGS.config_task.control_hz, FLAGS.max_steps - step
            )
            transitions, review = result["transitions"], result["review"]
            start = step
            step += len(transitions)
            if len(transitions) >= replay_buffer._capacity:
                raise ValueError("单条轨迹必须短于经验池容量")
            prepared = commit_episode(batch_processor, transitions, review)
            # 保存整条已核对的轨迹目录，再返回ACK；不使用全成功示教读取器。
            persist_reviewed(checkpoint_dir_path, prepared, review, start, step)
            env.client._call_operation(
                "episode_ack",
                {
                    "env_id": env.env_id,
                    "episode_id": review["episode_id"],
                    "outcome": "accepted" if review["admitted"] else "rejected",
                    "details": {
                        "end_step": step,
                        "transitions": len(prepared),
                        "success": review["success"],
                        "durable_episode": True,
                    },
                },
            )
            metrics = {
                "review/admitted": int(review["admitted"]),
                "review/auto_success": int(review["auto_success"]),
                "review/effective_success": int(review["success"]),
                "review/manual_override": int(review["manual_success"] is not None),
                "review/episode_steps": len(transitions),
            }
            if review["admitted"]:
                accepted_episodes += 1
                # 冷启动只计算收录回合；拒收不产生更新额度，也不触发历史额度。
                if accepted_episodes >= 10 and step >= FLAGS.batch_size:
                    credit += len(prepared)
                    updates, credit = divmod(credit, FLAGS.step_interval)
                    for _ in range(updates):
                        batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
                        agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))
                        agent, update_info = agent.update(agent, batch, FLAGS.utd_ratio, actor_batch)
                        update_count += 1
                        metrics.update({f"training/{k}": v for k, v in update_info.items()})
            metrics.update(
                {"review/accepted_episodes": accepted_episodes, "training/update_count": update_count}
            )
            wandb.log(metrics, step=step)
            logging.info(
                "env_step=%d episode=%s admitted=%s success=%s updates=%d",
                step,
                review["episode_id"],
                review["admitted"],
                review["success"],
                update_count,
            )
            if (
                FLAGS.checkpoint_model
                and FLAGS.checkpoint_interval > 0
                and step - last_checkpoint >= FLAGS.checkpoint_interval
            ):
                save_training_checkpoint()
                last_checkpoint = step
        if FLAGS.checkpoint_model and step != last_checkpoint:
            save_training_checkpoint()
        env.client._call_operation("session_complete", {"env_id": env.env_id})
    finally:
        sampler.close()
        env.client.close()
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
