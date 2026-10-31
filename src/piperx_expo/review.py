"""浏览器复核与学习器共享的轨迹身份和标签规则，不依赖 JAX。"""

import hashlib
import numpy as np
from piperx_expo.observation import IMAGE_KEYS

PROTOCOL = "episode_review_v1"


def observation_state(obs):
    state = np.r_[obs["cartesian_position"], obs["gripper_position"]].astype("<f4")
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("观测状态必须是有限7维")
    return state


class EpisodeDigest:
    def __init__(self):
        self.hash = hashlib.sha256(PROTOCOL.encode())
        self.steps = 0

    def observation(self, obs):
        self.hash.update(observation_state(obs).tobytes())
        for key in IMAGE_KEYS:
            array = np.asarray(obs[key])
            if array.shape != (224, 224, 3) or array.dtype != np.uint8:
                raise ValueError("传输观测必须为224×224 RGB uint8")
            self.hash.update(array.tobytes())

    def transition(self, action, next_obs):
        action = np.asarray(action, dtype="<f4")
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("执行目标必须是有限7维")
        self.hash.update(action.tobytes())
        self.observation(next_obs)
        self.steps += 1

    def hexdigest(self):
        return self.hash.hexdigest()


def resolve_review(auto_success, manual_success, admitted):
    if type(auto_success) is not bool or type(admitted) is not bool:
        raise ValueError("自动标签和经验准入必须是布尔值")
    if manual_success is not None and type(manual_success) is not bool:
        raise ValueError("人工标签只能是成功、失败或未设置")
    success = auto_success if manual_success is None else manual_success
    return {
        "auto_success": auto_success,
        "manual_success": manual_success,
        "admitted": admitted,
        "success": success,
        "reward": float(success),
        # 人工明确失败是终止；自动未检出成功的时间限制仍允许bootstrap。
        "mask": 0.0 if success or manual_success is False else 1.0,
    }


def commit_episode(processor, transitions, review):
    """只有整条复核完毕才接触replay，拒收连成功池边界也不能改变。"""
    labels = resolve_review(review["auto_success"], review["manual_success"], review["admitted"])
    if not transitions:
        raise ValueError("空轨迹不能加入经验")
    if not labels["admitted"]:
        return []
    prepared = []
    for i, transition in enumerate(transitions):
        action = np.asarray(transition["actions"], dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("轨迹存在无效动作")
        observation_state(transition["observations"])
        terminal = i == len(transitions) - 1
        prepared.append(
            {
                **transition,
                "actions": action,
                "rewards": labels["reward"] if terminal else 0.0,
                "masks": labels["mask"] if terminal else 1.0,
                "dones": terminal,
            }
        )
    processor.on_episode_start()
    for transition in prepared:
        processor.insert_transition(transition)
    processor.on_episode_done(labels["success"])
    return prepared
