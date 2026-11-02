"""命令行整批传输格式：只接受HDF5数组和JSON标签，不跨机器反序列化pickle。"""

import io

import h5py
import numpy as np

from piperx_expo.observation import IMAGE_KEYS
from piperx_expo.review import EpisodeDigest, resolve_review


def decode_episode(payload, review, prompt):
    transitions, digest = [], EpisodeDigest()
    with h5py.File(io.BytesIO(payload), "r") as file:
        actions, states = file["actions"], file["observations/state"]
        count = len(actions)
        if actions.shape != (count, 7) or states.shape != (count + 1, 7) or count != review["steps"]:
            raise ValueError("训练轨迹必须满足T动作/T+1观测")
        for key in IMAGE_KEYS:
            images = file[f"observations/{key}"]
            if images.shape != (count + 1, 224, 224, 3) or images.dtype != np.uint8:
                raise ValueError("训练图像格式错误")

        def observation(index):
            state = np.asarray(states[index], dtype=np.float32)
            return {"exterior_image_2_left": file[f"observations/{IMAGE_KEYS[0]}"][index],
                    "cartesian_position": state[:6], "gripper_position": state[6:],
                    **{key: file[f"observations/{key}"][index] for key in IMAGE_KEYS}, "prompt": prompt}

        obs = observation(0)
        digest.observation(obs)
        for index, action in enumerate(actions):
            following = observation(index + 1)
            digest.transition(action, following)
            transitions.append({"observations": obs, "actions": np.asarray(action, dtype=np.float32),
                                "is_hil": False})
            obs = following
    if digest.hexdigest() != review["digest"]:
        raise ValueError("整批训练数据摘要与rollout不一致")
    labels = resolve_review(review["auto_success"], review["manual_success"], review["admitted"])
    return {"transitions": transitions, "review": {**review, **labels}}
