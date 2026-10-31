"""可独立模拟验证的在线采集、整回合准入与恢复逻辑。"""

import json
import pickle
import re
import time
from collections import deque
from pathlib import Path

import numpy as np

from piperx_expo.client.recording import write_json
from piperx_expo.review import EpisodeDigest, commit_episode, resolve_review


def collect_episode(env, sampler, agent, replan_steps, control_hz, remaining_steps):
    if remaining_steps < 1:
        raise ValueError("采集预算已耗尽")
    obs = env.reset()
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
