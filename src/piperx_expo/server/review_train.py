"""独立的浏览器复核驱动；复用固定上游模型、RTC采样和梯度更新，不修改算法源码。"""

import json
import re
import sys
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
from piperx_expo.review import PROTOCOL
from piperx_expo.server.review_loop import DeferredBatch, collect_episode, evaluate_session, restore_reviewed, update_credit
from piperx_expo.server.model_store import checkpoint_info, readonly_checkpoint_manager
from piperx_expo.server.policy_warmup import warmup_policy
from piperx_expo.server.progress import TrainingProgress
from piperx_expo.server.session_control import StopAfterEpisode
from piperx_expo.config import load_config


def main(_):
    init_logging()
    progress = TrainingProgress()
    progress.update("initializing")
    experiment = load_config(os.environ["PIPERX_EXPERIMENT_CONFIG"])
    inference_only = experiment.get("inference_only") is True
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

    if inference_only:
        source = checkpoint_info(experiment["inference_checkpoint_dir"], experiment["inference_checkpoint_step"])
        checkpoint_manager = readonly_checkpoint_manager(source["checkpoint_dir"])
        resuming = True  # 完整恢复actor/critic/edit/target；不从RTC基线重新初始化。
        checkpoint_dir_path = epath.Path(log_dir)
    else:
        checkpoint_dir_path = epath.Path(checkpoint_dir)
        checkpoint_manager, resuming = initialize_checkpoint_dir(
            checkpoint_dir_path, keep_period=FLAGS.keep_period, overwrite=FLAGS.overwrite, resume=FLAGS.resume,
        )

    init_wandb(checkpoint_dir_path, resuming and not inference_only, FLAGS.project_name, FLAGS.run_name)
    wandb.config.update(FLAGS.flag_values_dict(), allow_val_change=resuming)

    progress.update("loading_demonstrations")
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

    progress.update("loading_model")
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
    progress.update("building_learner")
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
        agent = restore_checkpoint(checkpoint_manager, agent, step=source["step"] if inference_only else None)
        if hasattr(agent, "cache_infer_params"):
            agent = agent.cache_infer_params()
        steps = tuple(checkpoint_manager.all_steps())
        latest_step = max(steps) if steps else None
        if inference_only:
            start_step = source["step"]
        elif latest_step is not None:
            start_step = latest_step
            logging.info("Resuming from step %d", start_step)
        if not inference_only:
            restore_reviewed(batch_processor, checkpoint_dir_path, latest_step)

    sampler = AsyncChunkSampler(
        agent,
        delay=FLAGS.delay,
        replan_steps=FLAGS.replan_steps,
        sim_latency_ms=FLAGS.sim_latency,
        timer=Timer(),
    )
    if os.environ.get("PIPERX_POLICY_PREFLIGHT") == "1":
        # 完整agent仍从正式RTC权重构建，但用离线观测；不创建RPC监听或机器人环境。
        try:
            timing = warmup_policy(
                sampler, agent, dataset[0]["observations"],
                samples=experiment.get("policy_warmup_samples", 3), on_stage=progress.update,
            )
            progress.update("preflight_complete", timing=timing)
        except BaseException as error:
            progress.update("failed", error=str(error))
            raise
        finally:
            sampler.close()
            if inference_only:
                checkpoint_manager.close()
            wandb.finish(exit_code=1 if sys.exc_info()[0] else 0)
        return

    progress.update("waiting_client")
    # The env is created only now: EnvClientWrapper blocks until the rollout client
    # dials in, and the agent build above is the slow part.
    train_env_creation_request = {
        "example_action": example_action,
        "env_usage": "eval" if inference_only else "train",
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
    warmup = env.client._call_operation("prepare_policy", {
        "env_id": env.env_id, "batch_training": not inference_only,
        "session_mode": "eval" if inference_only else "train",
    })
    if inference_only and warmup.get("session_mode") != "eval":
        raise ValueError("客户端尚不支持只推理，请先更新GUI")
    if not inference_only and warmup.get("batch_training") is not True:
        raise ValueError("客户端尚不支持整轮采集后训练，请先更新GUI再连接")

    def report_warmup(stage):
        progress.update(stage)
        env.client._call_operation("policy_status", {
            "env_id": env.env_id, "details": {"stage": stage},
        })

    try:
        timing = warmup_policy(
            sampler, agent, warmup["observation"],
            samples=experiment.get("policy_warmup_samples", 3), on_stage=report_warmup,
        )
        # 编译时间不参与门槛；预热后的完整RTC路径仍不能超过动作看门狗预算。
        if timing["steady_max_ms"] >= float(warmup["watchdog_s"]) * 1000:
            raise RuntimeError(f"预热后RTC推理耗时{timing['steady_max_ms']:.1f}ms超过动作看门狗预算，禁止采集")
        progress.update("ready", timing=timing)
        env.client._call_operation("policy_status", {
            "env_id": env.env_id, "details": {"stage": "ready", "timing": timing},
        })
    except BaseException as error:
        progress.update("failed", error=str(error))
        logging.exception("运动前推理预热失败，禁止采集")
        sampler.close()
        env.client.close()
        wandb.finish(exit_code=1)
        raise

    if inference_only:
        try:
            evaluate_session(env, sampler, agent, replan_steps=FLAGS.replan_steps,
                             control_hz=FLAGS.config_task.control_hz, max_steps=FLAGS.max_steps,
                             output_dir=log_dir, checkpoint_step=source["step"], on_stage=progress.update)
        except BaseException as error:
            progress.update("failed", error=str(error))
            raise
        finally:
            sampler.close()
            env.client.close()
            checkpoint_manager.close()
            wandb.finish(exit_code=1 if sys.exc_info()[0] else 0)
        return

    audit_root = Path(checkpoint_dir) / "reviewed_episodes"
    accepted_episodes = sum(
        1
        for p in audit_root.glob("*/metadata.json")
        if re.fullmatch(r"[0-9a-f]{32}", p.parent.name)
        and json.loads(p.read_text()).get("admitted")
        and json.loads(p.read_text())["end_step"] <= start_step
    )
    combine_rng = jax.random.PRNGKey(FLAGS.seed + 100)
    step, credit, last_checkpoint, update_count = start_step, 0, start_step if resuming else None, 0
    finish_requested = False
    if resuming:
        state_file = Path(checkpoint_dir) / f"review_state_{start_step}.json"
        if not state_file.is_file():
            raise ValueError("缺少复核训练状态，不能把原自动模式checkpoint作为GUI续训目录")
        state = json.loads(state_file.read_text())
        credit, update_count = state["credit"], state["update_count"]

    progress.update("ready", env_step=start_step, total_steps=FLAGS.max_steps,
                    accepted_episodes=accepted_episodes, update_count=update_count,
                    checkpoint_step=last_checkpoint)

    def save_training_checkpoint():
        # 先写同step辅助状态，再提交模型；多出的辅助文件不会被当作模型checkpoint。
        write_json(
            Path(checkpoint_dir) / f"review_state_{step}.json",
            {"credit": credit, "update_count": update_count},
        )
        progress.update("saving", env_step=step, update_count=update_count)
        save_checkpoint(checkpoint_manager, agent, step)
        checkpoint_manager.wait_until_finished()
        progress.update("saved", checkpoint_step=step)

    pending = DeferredBatch(checkpoint_dir)
    staged_accepted = accepted_episodes

    def report(stage, **details):
        progress.update(stage, env_step=step, **details)
        return env.client._call_operation("batch_status", {
            "env_id": env.env_id, "stage": stage, "progress": {
                "completed_steps": step, "total_steps": FLAGS.max_steps,
                "update_count": update_count, "checkpoint_step": last_checkpoint, **details,
            },
        })

    try:
        while step < FLAGS.max_steps:
            env.client._call_operation("episode_context", {
                "env_id": env.env_id, "start_step": step, "total_steps": FLAGS.max_steps,
            })
            try:
                result, agent = collect_episode(
                    env, sampler, agent, FLAGS.replan_steps, FLAGS.config_task.control_hz, FLAGS.max_steps - step,
                    on_stage=progress.update,
                )
            except StopAfterEpisode:
                # 等待下一次开始时可结束本轮；随后仍须处理已审核的整批数据。
                finish_requested = True
                break
            transitions, review = result["transitions"], result["review"]
            start = step
            step += len(transitions)
            if len(transitions) >= replay_buffer._capacity:
                raise ValueError("单条轨迹必须短于经验池容量")
            pending.stage(result, start, step)
            staged_accepted += int(review["admitted"])
            # 采集期ACK只确认原始轨迹已归档，不能显示为已入池/已更新模型。
            env.client._call_operation(
                "episode_ack",
                {
                    "env_id": env.env_id,
                    "episode_id": review["episode_id"],
                    "outcome": "queued" if review["admitted"] else "rejected",
                    "details": {
                        "end_step": step,
                        "transitions": len(transitions) if review["admitted"] else 0,
                        "success": review["success"],
                        "durable_episode": True,
                        "training_pending": bool(review["admitted"]),
                    },
                },
            )
            wandb.log({
                "review/admitted": int(review["admitted"]),
                "review/auto_success": int(review["auto_success"]),
                "review/effective_success": int(review["success"]),
                "review/manual_override": int(review["manual_success"] is not None),
                "review/episode_steps": len(transitions),
            }, step=step)
            reply = report("episode_queued", accepted_episodes=staged_accepted,
                           pending_episodes=len(pending.entries))
            finish_requested = bool(reply.get("stop_after_episode"))
            # 不让已落盘的大图像仍被主循环引用；下一条只占一条回合的内存。
            del result, transitions
            if finish_requested:
                break

        report("processing_batch", pending_episodes=len(pending.entries))
        updates = 0
        for processed, (review, count, end) in enumerate(pending.commit(batch_processor, checkpoint_dir_path), 1):
            if review["admitted"]:
                accepted_episodes += 1
                scheduled, credit = update_credit(credit, accepted_episodes, count, end,
                                                  FLAGS.batch_size, FLAGS.step_interval)
                updates += scheduled
            env.client._call_operation("batch_episode_ack", {
                "env_id": env.env_id, "episode_id": review["episode_id"],
                "outcome": "accepted" if review["admitted"] else "rejected",
            })
            report("processing_batch", accepted_episodes=accepted_episodes,
                   processed_episodes=processed,
                   pending_episodes=len(pending.entries))
        report("updating" if updates else "training_skipped", accepted_episodes=accepted_episodes,
               planned_updates=updates, skip_reason="cold_start_or_credit" if not updates else None)
        for index in range(updates):
            batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
            agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))
            agent, update_info = agent.update(agent, batch, FLAGS.utd_ratio, actor_batch)
            # JAX异步派发必须等实际计算完成，才可报告更新计数。
            jax.block_until_ready(update_info)
            update_count += 1
            wandb.log({**{f"training/{k}": v for k, v in update_info.items()},
                       "training/update_count": update_count,
                       "review/accepted_episodes": accepted_episodes}, step=step)
            report("updating", accepted_episodes=accepted_episodes, planned_updates=updates,
                   completed_updates=index + 1)
        if (FLAGS.checkpoint_model or finish_requested) and step != last_checkpoint:
            report("saving", accepted_episodes=accepted_episodes)
            save_training_checkpoint()
            last_checkpoint = step
        progress.update("completed", env_step=step, total_steps=FLAGS.max_steps,
                        update_count=update_count, checkpoint_step=last_checkpoint,
                        stop_reason="operator_saved_stop" if finish_requested else "budget_complete")
        env.client._call_operation("session_complete", {
            "env_id": env.env_id, "progress": {
                "completed_steps": step, "total_steps": FLAGS.max_steps,
                "checkpoint_step": last_checkpoint, "update_count": update_count,
                "stop_reason": "operator_saved_stop" if finish_requested else "budget_complete",
            },
        })
    except BaseException as error:
        progress.update("failed", error=str(error))
        # 先输出根因再收尾，避免上传等待或二次中断掩盖最初的客户端错误。
        logging.exception("GUI采集/训练失败，正在关闭连接与日志")
        raise
    finally:
        sampler.close()
        env.client.close()
        # 异常退出必须记录失败，不能被默认finish(exit_code=0)标记为训练成功。
        wandb.finish(exit_code=1 if sys.exc_info()[0] else 0)


if __name__ == "__main__":
    app.run(main)
