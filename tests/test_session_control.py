from types import SimpleNamespace

import pytest

from piperx_expo.server.session_control import StopAfterEpisode, complete_episode, reset_reviewed_episode


@pytest.mark.parametrize("flags", [(True, False), (False, True), (False, False)])
def test_finish_request_before_or_after_save_decision_is_saved(flags):
    events = []
    replies = iter(flags)

    def call(operation, request):
        events.append((operation, dict(request)))
        return {"stop_after_episode": next(replies, False)}

    env = SimpleNamespace(env_id="test", client=SimpleNamespace(_call_operation=call))
    finish, checkpoint = complete_episode(
        env, step=13, total_steps=100, update_count=2, accepted_episodes=11,
        last_checkpoint=10, should_save=False, save_checkpoint=lambda: events.append(("save", {})),
    )
    assert finish is any(flags)
    assert checkpoint == (13 if finish else 10)
    if finish:
        assert [name for name, _ in events].count("save") == 1
        assert events[-1][1]["progress"]["checkpoint_step"] == 13
    else:
        assert not any(name == "save" for name, _ in events)


def test_failed_checkpoint_never_reports_episode_saved():
    calls = []

    def call(operation, request):
        calls.append(operation)
        return {"stop_after_episode": True}

    def save():
        raise OSError("磁盘写入失败")

    env = SimpleNamespace(env_id="test", client=SimpleNamespace(_call_operation=call))
    with pytest.raises(OSError, match="写入失败"):
        complete_episode(env, step=3, total_steps=100, update_count=0, accepted_episodes=1,
                         last_checkpoint=None, should_save=False, save_checkpoint=save)
    assert calls == ["training_control"]


def test_stop_at_reset_boundary_is_normal_and_does_not_require_observation():
    env = SimpleNamespace(env_id="test", client=SimpleNamespace(
        _call_operation=lambda *args: {"status": "success", "stop_after_episode": True}))
    with pytest.raises(StopAfterEpisode):
        reset_reviewed_episode(env)
