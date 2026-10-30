"""复用 EXPO 的夹持、抬高与连续满足规则；夹爪量纲为米。"""

import numpy as np


class GraspSuccessDetector:
    def __init__(self, config):
        self.height = float(config["height_m"])
        self.width = np.asarray(config["gripper_width_m"], dtype=float)
        self.required = config["consecutive_steps"]
        if not np.isfinite(self.height) or self.width.shape != (2,):
            raise ValueError("成功高度/夹爪区间格式错误")
        if not np.isfinite(self.width).all() or not 0 < self.width[0] < self.width[1]:
            raise ValueError("成功夹爪区间必须为正的米制宽度，不能包含空夹零宽度")
        if type(self.required) is not int or self.required < 1:
            raise ValueError("consecutive_steps 必须是正整数")
        self.reset()

    def reset(self):
        self.count = 0
        self.last_step = 0
        self.success = False

    def __call__(self, env, observation):
        # 同一步 get_observation/get_info/render 重复查询不能累计成功次数。
        if env.steps <= self.last_step:
            return self.success
        pose = np.asarray(observation["cartesian_position"], dtype=float)
        width = np.asarray(observation["gripper_position"], dtype=float).reshape(-1)
        if pose.shape != (6,) or width.shape != (1,) or not np.isfinite(np.r_[pose, width]).all():
            raise ValueError("成功检测收到无效位姿/夹爪反馈")
        command = env.last_action
        # 示教动作是下一帧实际宽度，因此允许闭合后的保持目标；不伪造 DROID 速度信号。
        holding = command is not None and 0 <= command[6] <= self.width[1]
        matched = self.width[0] <= width[0] <= self.width[1] and pose[2] > self.height and holding
        self.count = self.count + 1 if matched and env.steps == self.last_step + 1 else int(matched)
        self.last_step = env.steps
        self.success = self.success or self.count >= self.required
        return self.success
