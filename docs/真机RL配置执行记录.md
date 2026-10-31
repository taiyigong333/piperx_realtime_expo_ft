# 真机 RL 配置执行记录

日期：2026-10-30（Asia/Shanghai）。完整操作流程另见 `Real-Time-EXPO-FT完整训练流程.md`。

## 目标、输入与边界

- 复用官方抓取成功规则：实际夹爪部分闭合、末端抬高、连续满足条件。
- 用户已将机器人摆到成功姿态，并明确确认四批 394 条示教全部成功。
- 选用 batch2+3+4+5 的 pi0.5 JAX LoRA 10000 checkpoint，预测长度30。
- 本轮仅部署配置、转换数据、软件验证与只读采样；不启动真实运动或在线RL。
- 既有 checkpoint 实际使用右腕图像槽位、六维位姿 delta、夹爪绝对宽度。EXPO 原接口假设动作在统一的绝对坐标中；接入时必须在 VLA 边界转换，不能直接让每帧不同参考系的 delta 写入动作块。

## 只读标定

使用原 collector 的 PiperXRobot，连接 CAN 广播反馈后采样30次，随后只断开连接。未调用 enable/move/夹爪控制/急停。

- 位姿：`[0.166187, 0.001691, 0.286807, -2.646495104676562, -0.07922049474802262, -1.5342840388431753]`，米/弧度。
- 夹爪宽度：`0.0366 m`；30次采样数值一致。
- 初始候选阈值：高度高于 `0.281807 m`（成功样本以下5mm），宽度 `[0.0346, 0.0386] m`（样本±2mm），连续6个实际动作步。
- 这是正样本标定，不代表已完成空夹/张开/低位等现场负例验证，也不证明识别了红色物体。
- 工作空间沿用既有 GUI 的现场配置来源，动作步长和速度继续沿用 EXPO 的较保守设置。

## 实施计划

1. 成功检测器及 reset/终止/重复RPC计数回归。
2. checkpoint 相机/长度/归一化与绝对动作适配。
3. 将已确认成功的现有 LeRobot 数据导入 EXPO 回放，保存来源与哈希。
4. 双机部署、根测试、RPC smoke、服务器兼容检查。
5. 单独编写完整训练流程；更新交接与实验计划中的实际验证边界。

## 实际验证结果

| 项目 | 本轮结果与边界 |
| --- | --- |
| 根测试 | 23 passed、1 skipped；跳过的是根环境缺少JAX的坐标测试，同类逻辑另在服务器真实数据验证通过 |
| RPC | 服务器根环境和piperx Python3.10环境均通过原EnvClient的3步mock回环；无真实动作 |
| 静态检查 | 源码/脚本语法、Ruff及远程git diff --check通过 |
| GPU小算子 | cuda:0上loss 64.0→63.625823974609375；不等于VLA训练 |
| 数据转换 | 全部394条/44721帧导入；manifest SHA256=`1918585f58756f7018decce142b38418671f61ffed81b63bceefe850fd03e735` |
| 参数兼容 | 原checkpoint的71个参数叶子与30帧模型抽象形状匹配 |
| 数据/坐标 | 实际示教右腕槽位、绝对/delta互转、完整chunk统一参考系和RTC前缀重新锚定通过 |
| 完整VLA最小更新 | 两条真实示教、batch1；第一次离线RTC-SFT更新并保存step1，然后恢复step1再更新一次并保存step2 |
| 恢复后指标 | actor_state_step=1，actor_loss=0.02027590572834015，actor_lr≈2.5e-5，grad_norm≈0.45745；均为有限值 |
| 完整VLA离线推理 | 从验证step2/params实际加载，输出[1,30,7]绝对动作，5帧条件前缀保持通过；双次冷推理含编译约18秒，不能作为30Hz或p95指标 |
| 真机只读 | 部署后的check_success.py连续30次几何条件通过；未使能、未发动作/夹爪控制 |
| 离线成功示教覆盖 | 394条中380条曾连续6帧满足当前高度/实际宽度条件；仅几何覆盖96.45%，不含目标命令条件，不能称为检测准确率；其余14条未被重标成失败 |
| 进程收尾 | 验证进程正常退出，未遗留本任务GPU进程或8103监听 |

最小模型验证产物位于服务器：

```text
/data/gcj_data/rl/real_time_expo_ft/runs/rotating_square_h30_smoke_20261030_sft/checkpoints/{1,2}
/data/gcj_data/rl/real_time_expo_ft/code/runtime/validation/h30_sft_smoke.log
/data/gcj_data/rl/real_time_expo_ft/code/runtime/validation/h30_sft_resume.log
/data/gcj_data/rl/real_time_expo_ft/code/runtime/validation/h30_inference.log
/data/gcj_data/rl/real_time_expo_ft/code/runtime/validation/checkpoint_compatibility.json
```

首次单步运行还暴露wandb.init会覆盖提前绑定的wandb.log，导致原本的本地metrics钩子失效；改为init之后安装钩子，恢复验证中已实测终端和metrics.jsonl都记录loss/lr，并有独立回归。

双机原文件备份在各工程的`runtime/backups/20261030/before.tar`；服务器另备份原server.uv.lock。一次PyArrow导入失败留下的空输出保留为`data/expo_batch2345_v1.failed_pyarrow`，未删除原数据。

## 尚需完成

- 正式4000步RTC-SFT及所选batch4的实际显存/速度验证；两步smoke不能替代该阶段，正式rtc_weights仍指向将来4000产物。
- 空夹/低位/张开等现场负例，真实保持/单步/动作块检查，N=8与实际相机/网络情况下的推理p95。
- 真正在线RL、完整RL学习器的保存恢复和固定分布独立评估。本轮未执行真实运动、完整RTC-SFT或在线RL。

## 2026-10-31 W&B在线记录配置

- 正式配置启用online，团队`gongcj-beijing`、项目`piperx_realtime_expo_ft`；两个训练阶段自动以`_sft`/`_rl`区分run，同时保留终端与metrics.jsonl。
- 引用旧pi0.5训练的本机`.wandb_env`，仅解析W&B凭据白名单赋值，不执行shell、不复制或展示密钥；明确指定原W&B服务地址。
- 25项根测试通过、1项跳过；mock RPC及Ruff通过。未启动训练。
- 认证及直连GraphQL（使用/绕过环境代理）均返回HTML HTTP403，未验证在线上传、未创建线上测试run。待确认可用代理或专用服务地址后重试，不能将此记录为W&B在线验收通过。
