"""回合边界的正常收尾与checkpoint确认，不干预动作执行或上游算法。"""


class StopAfterEpisode(Exception):
    """客户端在下一次归位前请求正常停止。"""


def reset_reviewed_episode(env):
    reply = env.client._call_operation("reset", {"env_id": env.env_id})
    if reply.get("stop_after_episode"):
        raise StopAfterEpisode()
    return reply["observation"]


def complete_episode(env, *, step, total_steps, update_count, accepted_episodes,
                     last_checkpoint, should_save, save_checkpoint):
    """先询问收尾意图再保存，保存后再确认，覆盖更新/保存期间的新请求。"""
    details = {"completed_steps": step, "total_steps": total_steps, "update_count": update_count,
               "accepted_episodes": accepted_episodes, "checkpoint_step": last_checkpoint}

    def call(operation):
        return env.client._call_operation(operation, {"env_id": env.env_id, "progress": dict(details)})

    finish = bool(call("training_control").get("stop_after_episode"))
    if should_save or (finish and last_checkpoint != step):
        save_checkpoint()  # 失败必须向上传播，不能把未完成保存显示为已正常停止。
        last_checkpoint = details["checkpoint_step"] = step
    reply = call("episode_complete")
    finish = finish or bool(reply.get("stop_after_episode"))
    if finish and last_checkpoint != step:
        # 请求可能在正常保存判断之后才到达；无论周期如何，都补齐本回合保存。
        save_checkpoint()
        last_checkpoint = details["checkpoint_step"] = step
        call("episode_complete")
    return finish, last_checkpoint
