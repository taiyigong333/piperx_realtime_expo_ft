"""运动前编译普通/RTC推理路径并测时；所有动作丢弃，不推进训练agent或执行设备。"""

import time

import numpy as np


def warmup_policy(sampler, agent, observation, *, samples=3, on_stage=None):
    if type(samples) is not int or samples < 1:
        raise ValueError("policy_warmup_samples 必须是正整数")
    on_stage = on_stage or (lambda *_: None)
    current = agent
    timings = []
    try:
        for index in range(samples + 2):
            stage = "warmup_boot" if index == 0 else "warmup_rtc" if index == 1 else "warmup_measure"
            on_stage(stage)
            # 第一次编译delay=0；后续通过真实的prefix/pending接口编译delay>0。
            sampler.observe(observation)
            start = time.monotonic()
            sampler.launch(current, observation, int(getattr(sampler, "delay", 0)), "policy")
            chunk, current, _ = sampler.sample(current, observation, {})
            chunk = np.asarray(chunk)  # 等待设备计算完成，不能只记录JAX异步派发耗时。
            elapsed = time.monotonic() - start
            if chunk.ndim != 2 or chunk.shape[1] != 7 or not len(chunk) or not np.isfinite(chunk).all():
                raise ValueError("预热返回无效动作块，禁止开始真实采集")
            timings.append(elapsed * 1000)
        return {
            "boot_compile_ms": timings[0], "rtc_compile_ms": timings[1],
            "steady_p50_ms": float(np.percentile(timings[2:], 50)),
            "steady_p95_ms": float(np.percentile(timings[2:], 95)),
            "steady_max_ms": max(timings[2:]), "samples": samples,
        }
    finally:
        # 清除预热动作前缀；真正的首条rollout从干净的delay=0状态开始。
        sampler.on_episode_end()
