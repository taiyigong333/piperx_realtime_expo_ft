"""服务器环境执行：验证跨观测参考系的转换，而不加载完整VLA。"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/expo"))
pytest.importorskip("expo_ft.agents.vla.pi05")

from piperx_expo.server.checkpoint_adapter import ActionCoordinates


def test_delta_bridge_anchors_whole_chunk_and_preserves_gripper():
    stats = {"state": {"q01": [-1.] * 7, "q99": [1.] * 7},
             "actions": {"q01": [-2.] * 7, "q99": [2.] * 7}}
    vla = {"state": stats["state"], "actions": {"q01": [-0.1] * 7, "q99": [0.1] * 7}}
    bridge = ActionCoordinates(stats, vla)
    state = np.array([[0.3, 0.1, 0.2, 0.2, -0.1, 0.5, 0.0366]], np.float32)
    actions = np.tile(state[:, None], (2, 30, 1))
    actions[:, :, 0] += np.linspace(0.001, 0.03, 30)
    normalized = bridge.normalize(actions, stats["actions"])
    delta = bridge.convert(normalized, bridge.normalize(state, stats["state"]), to_vla=True)
    physical = np.asarray(bridge.physical(delta, vla["actions"]))
    np.testing.assert_allclose(physical[..., 0], actions[..., 0] - state[0, 0], atol=1e-6)
    np.testing.assert_allclose(physical[..., 6], actions[..., 6], atol=1e-6)
    restored = bridge.convert(delta, bridge.normalize(state, stats["state"]), to_vla=False)
    np.testing.assert_allclose(restored, normalized, atol=2e-6)
    # RTC下一块推理观测已移动：同一绝对前缀必须重新锚定，不能复用旧delta。
    moved = state.copy()
    moved[:, 0] += 0.01
    prefix = bridge.convert(normalized[:, :5], bridge.normalize(moved, stats["state"]), to_vla=True)
    np.testing.assert_allclose(np.asarray(bridge.physical(prefix, vla["actions"]))[..., 0],
                               physical[:, :5, 0] - 0.01, atol=1e-6)
