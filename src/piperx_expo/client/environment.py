"""本地执行环境：绝对位姿动作、任务检测和失联停止。"""

import importlib
import threading
import time

from piperx_expo.client.devices import MockDevices, PiperXDevices
from piperx_expo.client.safety import validate_action, validate_client_config


class PiperXEnv:
    def __init__(self, config: dict, allow_motion=False, device=None):
        self.config = config
        self.real = config["backend"] == "piperx"
        if config["backend"] not in {"mock", "piperx"}:
            raise ValueError("backend 只支持 mock / piperx")
        if self.real and not allow_motion:
            raise ValueError("真实设备必须显式传 --allow-motion")
        if config.get("workspace_m") is None:
            raise ValueError("必须先配置现场工作空间")
        validate_client_config(config)
        self.detector = None
        if config.get("success_detector"):
            module, function = config["success_detector"].split(":", 1)
            self.detector = getattr(importlib.import_module(module), function)
        elif self.real:
            raise ValueError("真实任务必须先实现 success_detector，不能猜测成功标签")
        self.device = device or (PiperXDevices(config) if self.real else MockDevices(config))
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.armed = False
        self.fault = False
        self.steps = 0
        self.terminal = False
        self.last_action_time = time.monotonic()
        self.thread = threading.Thread(target=self._watchdog, daemon=True)
        self.thread.start()

    def _watchdog(self):
        while not self.closed.wait(0.02):
            with self.lock:
                if self.armed and time.monotonic() - self.last_action_time > self.config["watchdog_s"]:
                    self.fault = True
                    self.armed = False
                    self.device.stop_hold()

    def reset(self):
        with self.lock:
            if self.fault:
                raise RuntimeError("看门狗/控制错误已停止设备，需要现场检查后重启客户端")
            self.armed = False
            self.device.reset()
            self.steps = 0
            self.terminal = False
            self.last_action_time = time.monotonic()
            observation = self.device.observe()
            # 第一条动作前不启动看门狗：模型首次编译可能耗时很长。
            return observation

    def step(self, action):
        with self.lock:
            if self.fault:
                raise RuntimeError("设备已停止，需要重启客户端")
            if self.terminal:
                # 上游循环在得到终止反馈后仍会发一次 step，再启动 reset。
                # 该动作不会进入 replay；本地保持当前位置，禁止多执行一步。
                return self.device.read_state()
            try:
                executed = validate_action(action, self.device.read_state(), self.config)
                self.device.send(executed)
            except BaseException:
                self.fault = True
                self.armed = False
                self.device.stop_hold()
                raise
            self.steps += 1
            self.armed = True
            self.last_action_time = time.monotonic()
            return executed

    def get_observation(self):
        return self.device.observe()

    def get_info_for_step(self, observation=None):
        observation = observation if observation is not None else self.get_observation()
        success = bool(self.detector(self, observation)) if self.detector else False
        done = success or self.steps >= self.config["episode_steps"]
        if done:
            with self.lock:
                self.armed = False
                self.terminal = True
        # 达到时间限制属于截断；只有任务成功时关闭 bootstrap。
        return done, success, float(success), 0.0 if success else 1.0

    def close(self):
        self.closed.set()
        with self.lock:
            self.armed = False
            self.device.close()
        self.thread.join(timeout=2)
