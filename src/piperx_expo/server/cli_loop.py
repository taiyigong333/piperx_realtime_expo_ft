"""按轨迹数量采集→整批传输→训练→继续rollout；不以动作数触发批次。"""

import time
from collections import deque
from datetime import datetime
import re

import numpy as np

from piperx_expo.data.online_batch import decode_episode
from piperx_expo.review import EpisodeDigest
from piperx_expo.server.review_loop import DeferredBatch
from piperx_expo.server.session_control import reset_reviewed_episode


def batch_options(config):
    options = config.get("cli_batch")
    if not isinstance(options, dict):
        raise ValueError("cli_batch需要episodes_per_batch、rounds、updates_per_batch、replay_capacity")
    for key in ("episodes_per_batch", "rounds", "updates_per_batch", "replay_capacity"):
        if type(options.get(key)) is not int or options[key] < 1:
            raise ValueError(f"cli_batch.{key}必须为正整数")
    return options


def collect_cli_episode(env, sampler, agent, replan_steps, control_hz):
    obs = reset_reviewed_episode(env)
    digest, plan, dispatch = EpisodeDigest(), deque(), None
    digest.observation(obs)
    steps = 0
    try:
        while True:
            sampler.observe(obs)
            sampler.launch(agent, obs, len(plan), "policy")
            if not plan:
                chunk, agent, _ = sampler.sample(agent, obs, {})
                chunk = np.asarray(chunk[:replan_steps], dtype=np.float32)
                if chunk.ndim != 2 or chunk.shape[1] != 7 or not len(chunk) or not np.isfinite(chunk).all():
                    raise ValueError("模型返回无效动作块")
                plan.extend(chunk)
                sampler.launch(agent, obs, len(plan), "policy")
            if dispatch is not None:
                time.sleep(max(0, 1 / control_hz - (time.monotonic() - dispatch)))
            dispatch = time.monotonic()
            reply = env.client._call_operation("step", {"env_id": env.env_id, "action": plan.popleft().tolist()})
            if reply.get("episode_interrupted"):
                break
            executed = reply["action"]
            reply = env.client._call_operation("get_observation", {"env_id": env.env_id})
            if reply.get("episode_interrupted"):
                break
            obs = reply["observation"]
            digest.transition(executed, obs)
            steps += 1
            if reply["done"]:
                break
        review = env.client._call_operation("review_episode", {
            "env_id": env.env_id, "steps": steps, "digest": digest.hexdigest()})["review"]
        if (review["steps"] != steps or review["digest"] != digest.hexdigest()
                or not re.fullmatch(r"[0-9a-f]{32}", str(review.get("episode_id", "")))):
            raise ValueError("终端复核数据与推理摘要不一致")
        # 采集期仅保留标签和摘要；训练图像在本批采满后才从机器人端读取。
        return review, agent
    finally:
        sampler.on_episode_end()


def run_cli_batches(env, sampler, agent, config, checkpoint_dir, *, start_step=0,
                    prepare_training, train_and_save, on_stage=None):
    options = batch_options(config)
    on_stage = on_stage or (lambda *_a, **_k: None)
    step, total_episodes = start_step, 0
    for round_index in range(options["rounds"]):
        reviews, boundaries = [], []
        for _ in range(options["episodes_per_batch"]):
            # 此字段仅兼容终端进度显示，不作为批次/回合截断预算。
            env.client._call_operation("episode_context", {
                "env_id": env.env_id, "start_step": step, "total_steps": max(step + 1, config["max_steps"])})
            on_stage("collecting", round=round_index + 1, completed_episodes=total_episodes)
            review, agent = collect_cli_episode(env, sampler, agent, config["replan_steps"], config["control_hz"])
            start = step
            step += review["steps"]
            if review["admitted"] and review["steps"] >= options["replay_capacity"]:
                raise ValueError("单条轨迹超过训练池容量；请增大replay_capacity")
            reviews.append(review)
            boundaries.append((start, step))
            total_episodes += 1
            env.client._call_operation("episode_ack", {
                "env_id": env.env_id, "episode_id": review["episode_id"],
                "outcome": "queued" if review["admitted"] else "rejected",
                "details": {"training_pending": True, "end_step": step}})
        manifest = env.client._call_operation("export_batch", {"env_id": env.env_id})["episodes"]
        if manifest != reviews:
            raise ValueError("整批清单与采集审核结果不一致")
        name = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        pending = DeferredBatch(checkpoint_dir, batch_name=f"{name}_round{round_index + 1:04d}")
        for review, (start, end) in zip(reviews, boundaries):
            payload = env.client._call_operation("export_episode", {
                "env_id": env.env_id, "episode_id": review["episode_id"]})["hdf5"]
            result = decode_episode(payload, review, config["language_instruction"])
            pending.stage(result, start, end)
            del payload, result
        env.client._call_operation("batch_received", {
            "env_id": env.env_id, "episode_ids": [review["episode_id"] for review in reviews]})
        env.client._call_operation("batch_status", {"env_id": env.env_id, "stage": "processing_batch"})
        # 到此才加载历史经验/示教、分配实际训练池；后续轮次复用同一个池。
        processor = prepare_training()
        accepted = 0
        for review, count, _ in pending.commit(processor, checkpoint_dir):
            accepted += int(review["admitted"] and count > 0)
            env.client._call_operation("batch_episode_ack", {
                "env_id": env.env_id, "episode_id": review["episode_id"],
                "outcome": "accepted" if review["admitted"] else "rejected"})
        # CLI显式每批固定更新次数，取代旧10条冷启动/按step发额度。
        agent = train_and_save(agent, processor, step, options["updates_per_batch"] if accepted else 0)
        on_stage("batch_complete", round=round_index + 1, completed_episodes=total_episodes, env_step=step)
    env.client._call_operation("session_complete", {"env_id": env.env_id, "progress": {
        "completed_steps": step, "completed_episodes": total_episodes, "rounds": options["rounds"]}})
    return agent
