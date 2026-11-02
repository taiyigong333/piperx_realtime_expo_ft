# Piper-X Real-Time EXPO-FT：当前接手摘要

更新：2026-11-02（Asia/Shanghai）。先读 [AGENTS.md](AGENTS.md)，文档入口见 [docs/README.md](docs/README.md)；详细状态见[项目交接](docs/项目交接.md)，下一步见[实验计划](docs/实验计划.md)。

## 目标与固定边界

- 任务：抓取旋转红色方块；piperx执行和采集，4090负责模型推理、RTC-SFT与在线RL。
- 动作为7维米制绝对法兰XYZ＋RPY(rad)＋夹爪宽度(m)；VLA内部使用原checkpoint的delta，右腕进入right槽位，不能改成DROID速度或夹爪概率。
- client Python3.10、server Python3.11、根轻量环境独立；仅用uv及已提交锁文件，外部代码保持固定子模块。
- CLI保留；GUI支持服务器启停/选择历史实验续训、逐条复核/自动收录、双相机回放和独立机械臂按钮。
- 本轮修改：正式策略速度90%、位移0.3m、每轴旋转0.5rad、夹爪0～0.10m；归位仍10%。GUI相机随进程常驻，rollout开始才写盘；顶部按钮/Esc先锁存当前位置并禁止后续动作，保持失败回退阻尼急停。软件回归及部署结果见下述记录。

## 最近已确认状态

- 最近功能修复为fb68b90，交接记录提交18c706d；均已推送并同步双机。后续整理提交以实际git HEAD为准。
- 2026-11-02只读反馈：六驱动使能标志True，但控制器急停/待机、err_code=0。已补充控制器恢复、CAN模式和有限使能重试，尚未现场验收真实支撑能力。
- 最近软件验证：4090正式目录78 passed、1 skipped及模拟RPC通过；piperx Python3.10编译/模拟RPC通过。本地完整集成缺EXPO子模块，已执行局部测试；不能把服务器测试说成本地全套通过。
- 最后一次GUI重启检查为idle、未连接学习器、无手动操作。本条是当次检查结果，不是持续监控；本次文档整理不复查或改动运行进程。
- 正式batch32 RTC-SFT最后一次训练结果核对为2026-11-01：退出码0，最后日志索引3999，2000/4000参数目录存在。最终4000权重恢复、推理及延迟验收仍待完成。
- 已出现用户手动操作和报错；尚无真实策略回合、GUI在线RL更新或成功率评估通过的验收记录。

## 双机位置与入口

| 角色 | SSH | 项目目录 |
| --- | --- | --- |
| 4090 | gh4090piper | /data/gcj_data/rl/real_time_expo_ft/code |
| piperx | piperx | /home/cv/gcj/rl/baseline/realtime_expo_ft |

GUI在piperx启动：`bash scripts/gui.sh --config configs/client/rotating_square.calibrated.yaml --record-dir /home/cv/gcj/rl/data/realtime_expo_ft/gui_episodes --allow-motion --port 8888`。本轮按用户要求只保留8888；模型通信仍为192.168.50.2:8103。[完整操作说明](docs/浏览器客户端使用说明.md)。

- 2026-11-02本轮只读核对：现有两次GUI实验均无数字模型checkpoint，仅有wandb_id，不能从其中恢复未保存模型。新GUI配置每个完整回合保存；有配套checkpoint/复核状态的实验才可在页面续训。
- 本轮验证/部署进行中，最终结果见[本轮记录](docs/历史/数据与控制/2026-11-02_动作放宽常驻相机与GUI续训.md)。

## 接手优先事项

1. 刷新GUI后，由现场人员验收恢复使能、保持/取消、固定归位；本次维护未触发真实命令。
2. 核验RTC-SFT最终4000参数的恢复、推理和p50/p95延迟。
3. 完成原采集预检、成功检测负例、工作空间、归位路径、保持/单步/短动作块验收。
4. 再做GUI短回合、复核/拒收、最小在线更新及checkpoint恢复，最后扩大采集和独立评估。

## 文档维护

本文件只放摘要；[项目交接](docs/项目交接.md)维护当前事实，[实验计划](docs/实验计划.md)维护下一步。旧版完整原文已保留为[历史快照](docs/历史/项目记录/2026-11-02_HANDOFF历史快照.md)。新历史记录按[文档分类规则](docs/README.md)归档，不向当前入口持续追加大段过程日志。
