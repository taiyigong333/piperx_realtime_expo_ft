"""可独立模拟验证的在线采集、整回合准入与恢复逻辑。"""

import json
import pickle
import re
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np

from piperx_expo.client.recording import write_json
from piperx_expo.review import EpisodeDigest, commit_episode, resolve_review
from piperx_expo.server.session_control import reset_reviewed_episode


def collect_episode(env, sampler, agent, replan_steps, control_hz, remaining_steps, *, on_stage=None):
    if remaining_steps < 1:
        raise ValueError("采集预算已耗尽")
    on_stage = on_stage or (lambda *_: None)
    on_stage("awaiting_reset")
    obs = reset_reviewed_episode(env)
    on_stage("collecting")
    digest = EpisodeDigest()
    digest.observation(obs)
    transitions, plan = [], deque()
    dispatch_time = None
    try:
        for index in range(remaining_steps):
            sampler.observe(obs)
            sampler.launch(agent, obs, len(plan), "policy")
            if not plan:
                chunk, agent, _ = sampler.sample(agent, obs, {})
                chunk = np.asarray(chunk[:replan_steps], dtype=np.float32)
                if chunk.ndim != 2 or chunk.shape[1] != 7 or not len(chunk) or not np.isfinite(chunk).all():
                    raise ValueError("模型返回无效动作块")
                plan.extend(chunk)
                sampler.launch(agent, obs, len(plan), "policy")
            if dispatch_time is not None:
                time.sleep(max(0, 1 / control_hz - (time.monotonic() - dispatch_time)))
            dispatch_time = time.monotonic()
            executed, _ = env.step(plan.popleft().tolist())
            # 最后一条预算也必须得到实测后继观测，并明确截断，不能丢掉未完成回合。
            response = env.client._call_operation(
                "get_observation", {"env_id": env.env_id, "force_done": index == remaining_steps - 1}
            )
            next_obs = response["observation"]
            digest.transition(executed, next_obs)
            transitions.append(
                {"observations": obs, "actions": np.asarray(executed, dtype=np.float32), "is_hil": False}
            )
            obs = next_obs
            if response["done"]:
                break
        # RTC后台推理此时不再需要；复核/模型更新期间不保留跨回合动作前缀。
        sampler.on_episode_end()
        on_stage("review_pending")
        reply = env.client._call_operation(
            "review_episode", {"env_id": env.env_id, "steps": len(transitions), "digest": digest.hexdigest()}
        )
        review = reply["review"]
        if (
            review.get("steps") != len(transitions)
            or review.get("digest") != digest.hexdigest()
            or review.get("auto_success") is not bool(response["success"])
            or not re.fullmatch(r"[0-9a-f]{32}", str(review.get("episode_id", "")))
        ):
            raise ValueError("复核标签与本回合传输数据不符")
        labels = resolve_review(review["auto_success"], review["manual_success"], review["admitted"])
        review = {**review, **labels}
        return {"transitions": transitions, "review": review}, agent
    except BaseException:
        sampler.on_episode_end()
        raise


def persist_reviewed(checkpoint_dir, prepared, review, start_step, end_step):
    root = Path(checkpoint_dir) / "reviewed_episodes"
    root.mkdir(parents=True, exist_ok=True)
    target = root / review["episode_id"]
    staging = root / (".pending_" + review["episode_id"])
    staging.mkdir()
    if prepared:
        with (staging / "transitions.pkl").open("xb") as file:
            pickle.dump(prepared, file, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(staging / "metadata.json", {**review, "start_step": start_step, "end_step": end_step})
    staging.rename(target)


class DeferredBatch:
    """采集阶段仅落盘原始经验，整轮结束才整理入池；内存只保留一条轨迹。"""

    def __init__(self, checkpoint_dir):
        # 与可恢复模型经验隔离；未完成训练的采集不能冒充checkpoint覆盖的数据。
        self.root = Path(checkpoint_dir) / "collection_batches" / uuid.uuid4().hex
        self.entries = []

    def stage(self, result, start_step, end_step):
        review = result["review"]
        persist_reviewed(self.root, result["transitions"] if review["admitted"] else [],
                         {**review, "training_pending": True}, start_step, end_step)
        self.entries.append((review, start_step, end_step))

    def commit(self, processor, checkpoint_dir):
        for review, start, end in self.entries:
            transitions = []
            if review["admitted"]:
                path = self.root / "reviewed_episodes" / review["episode_id"] / "transitions.pkl"
                # 只读取本进程写出的文件，不接受用户上传的pickle。
                with path.open("rb") as file:
                    transitions = pickle.load(file)
            prepared = commit_episode(processor, transitions, review) if transitions else []
            persist_reviewed(checkpoint_dir, prepared, review, start, end)
            yield review, len(prepared), end
            del prepared, transitions


def update_credit(credit, accepted_episodes, admitted_steps, end_step, batch_size, step_interval):
    """保留原10条冷启动和按步更新额度；只延后执行，不为拒收/冷启动补发额度。"""
    if accepted_episodes < 10 or end_step < batch_size:
        return 0, credit
    return divmod(credit + admitted_steps, step_interval)


def restore_reviewed(processor, checkpoint_dir, up_to_step):
    root = Path(checkpoint_dir) / "reviewed_episodes"
    entries = [
        (json.loads(p.read_text()), p.parent)
        for p in root.glob("*/metadata.json")
        if re.fullmatch(r"[0-9a-f]{32}", p.parent.name)
    ]
    for review, folder in sorted(entries, key=lambda item: item[0]["end_step"]):
        if review["end_step"] > (up_to_step or 0):
            # 模型未包含的后续经验保留到隔离目录，避免之后恢复时重复加入历史分支。
            orphaned = Path(checkpoint_dir) / "unrestored_episodes"
            orphaned.mkdir(exist_ok=True)
            folder.rename(orphaned / folder.name)
        elif review["admitted"]:
            # 这里只读本应用自己的归档；pickle不能接收浏览器上传或不可信来源。
            with (folder / "transitions.pkl").open("rb") as file:
                transitions = pickle.load(file)
            commit_episode(processor, transitions, review)
