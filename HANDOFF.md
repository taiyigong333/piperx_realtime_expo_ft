# Piper-X Real-Time EXPO-FT：当前交接快照

更新日期：2026-11-01（Asia/Shanghai）。本文件保存在本地项目根目录，供新会话和后续维护直接接手。先读 `AGENTS.md`，再读本文；历史实施过程见 `docs/项目交接.md`。本文维护当前状态，历史文档中的“正在训练”“权重未生成”等表述以各自记录日期为准。

## 当前任务和用户要求

- 目标是抓取旋转红色方块的 Real-Time EXPO-FT 真机强化学习。
- 保留原命令行入口，同时提供浏览器客户端。浏览器默认逐条人工复核，也可选自动收录；最大步数与收录模式从下一条轨迹生效。
- 自动结果、人工成功/失败、是否加入经验分别记录；训练优先使用人工标签。拒收不入经验池、成功池或更新额度，服务器ACK后才显示收录。
- 数据通过现有有线WebSocket逐步直接回传，不需要手工转换、上传整条在线轨迹。
- 使用uv及已提交锁文件，训练保留实时日志、checkpoint和W&B。持续使用Git维护，阶段完成后提交、推送并同步双机。

## 本轮最新修改（2026-11-01）

- 数据基准是data-transfer的`piperx_to_openpi_vla@3a26229`；旧OpenPI部署只参考归位。统一OpenCV INTER_AREA/JPEG解码，去除额外RPY折回/unwrap；原394条44721帧导入集全量数值与源哈希一致，视频抽18帧、原归一化通过。
- 已核论文和固定上游：论文执行末端速度，本工程为兼容既有Piper-X checkpoint保留绝对法兰控制、VLA内部delta；这是方法迁移，不能称控制空间/所有超参数严格复现。详见[表示合同](docs/数据表示与固定归位核对_20261101.md)。
- `edit_action_xyzg`默认true，按论文只编辑XYZ和夹爪；false可选全7维编辑。基础VLA始终输出完整7维动作，开关不要求重训RTC-SFT，不同RL模式分run保存。
- 回合reset由人工确认方块和路径后，按`configs/client/home.openpi.yaml`固定六关节目标以10%速度`move_j`归位；新反馈最大误差≤0.035rad才开始下一条。超时30秒/异常停止，GUI归位中可中断。未做真机路径验收、真实运动或在线RL。
- 软件验证：本地及服务器隔离副本根测试46 passed、1 skipped，原CLI模拟RPC、Ruff和JS语法通过；server原checkpoint表示/双模式edit验证及GPU小算子更新通过。根/客户端/服务器uv.lock已更新，OpenCV加入根环境，server/client原有OpenCV版本未升级。

## 最新状态（本次实际核对）

2026-11-01只读检查服务器：

- 本轮修改前本地、双机及远程main均为 `6f19b9d`；浏览器功能最初实现提交为`d3dccbf`。当前最新代码以`git HEAD`及本轮提交为准。
- 正式RTC-SFT运行名为 `rotating_square_h30_b32_20261031_sft`，已写入退出码 **0**。最后一条指标时间为2026-11-01 02:31:24+08:00，日志step索引3999，actor loss约0.00247036。
- `checkpoints/2000/params`、`checkpoints/4000/params`目录均存在。此次只确认进程退出和目录存在，**没有进行最终4000权重的恢复、参数计数或推理验收**。
- 训练采用全部394条已确认成功示教，44721帧；动作预测长度30，batch32，配置预算4000次更新、每2000保存。原batch4实验在step112中断，不能当作当前训练或可恢复checkpoint。
- 浏览器功能已实现并同步双机。历史软件验收为39 passed、1 skipped，原CLI RPC、Ruff、前端JS语法通过；server真实训练入口仅CPU导入和dry-run通过。piperx Python3.10模拟自动收录验证为12动作/13观测。
- 历史浏览器验收包括自动成功后人工标失败、下一条12步自动收录、双相机回放。临时模拟预览服务已经关闭。
- 尚未进行真机运动、真实在线RL更新、真实成功率评测或GUI闭环GPU训练。本次维护交接文档也未执行这些操作。

## 本地与双机位置

| 角色 | 连接 | 项目目录 |
| --- | --- | --- |
| 本地开发 | 当前Mac | `/Users/gongtaiyi/Documents/ChatGPT/expoft` |
| GPU服务器 | `ssh gh4090piper` | `/data/gcj_data/rl/real_time_expo_ft/code` |
| 机器人电脑 | `ssh piperx` | `/home/cv/gcj/rl/baseline/realtime_expo_ft` |

- 模型通信走有线：服务器 `192.168.50.2:8103`，机器人 `192.168.50.1`。
- GUI默认在机器人本机 `http://127.0.0.1:8767`；远程浏览使用相同端口SSH转发。它不是模型通信端口。
- Git远程为 `git@github.com:taiyigong333/piperx_realtime_expo_ft.git`。不提交数据、权重、密钥、环境、日志或 `configs/local/`；不强推、不覆盖现场配置。

## 训练和数据位置

- 训练输出：`/data/gcj_data/rl/real_time_expo_ft/runs/rotating_square_h30_b32_20261031_sft`，其中有 `training.log`、`metrics.jsonl`、`exit_code` 和 `checkpoints/`。
- 现有配置引用的RTC权重：上述输出下 `checkpoints/4000/params`。
- 原普通SFT来源为 `pi05_lora_batch2345_h30_10k_20261025/10000/params`，完整路径见 `configs/server/rotating_square.h30.yaml`；不要与RTC-SFT产物混淆。
- 示教：`/data/gcj_data/rl/real_time_expo_ft/data/expo_batch2345_v1`；资产：`/data/gcj_data/rl/real_time_expo_ft/assets/piperx/batch2345_h30_v1`。
- W&B run为 `ek9lzkgs`：[训练页面](https://forge.coreweave.com/wandb/gongcj-beijing/piperx_realtime_expo_ft/runs/ek9lzkgs)。独立代理使用服务器 `127.0.0.1:17890`；凭据文件和代理私有配置均不提交，也不在交接中复制内容。
- 正式GUI归档建议放在机器人 `/home/cv/gcj/rl/data/realtime_expo_ft/gui_episodes`；默认是项目内被忽略的 `runtime/gui_episodes`。

## 入口和关键实现

| 用途 | 入口/配置 |
| --- | --- |
| 原CLI机器人端 | `bash scripts/client.sh --config configs/client/rotating_square.calibrated.yaml --allow-motion` |
| 原CLI服务器 | `bash scripts/server.sh rl --config configs/server/rotating_square.h30.yaml` |
| 浏览器机器人端 | `bash scripts/gui.sh --config configs/client/rotating_square.calibrated.yaml --allow-motion` |
| 浏览器服务器 | `bash scripts/server.sh rl --config configs/server/rotating_square.gui.yaml` |
| 无硬件界面体验 | `bash scripts/gui.sh --config configs/client/gui.mock.yaml`，然后点击“模拟体验” |

以上真机命令是入口索引，执行前仍需完成下节验收。GUI服务器用 `episode_review: true` 选择独立驱动；与原CLI服务器占用同一端口，每次只运行一种。GUI逐条复核/自动收录在页面“会话参数”选择；不修改原CLI默认行为。

- `client/environment.py`、`devices.py`、`success.py`：设备执行、动作限制、看门狗和成功检测。
- `client/gui.py`、`gui_control.py`、`recording.py`、`client/web/`：本机HTTP界面、会话状态机、异步HDF5与回放。
- `review.py`、`server/review_loop.py`、`review_train.py`：整条摘要、人工优先标签、准入、持久化和独立训练驱动。
- `server/checkpoint_adapter.py`：VLA边界的绝对/delta互转及RTC前缀重新锚定。replay/critic保持绝对动作，夹爪始终为米制宽度，右腕映射不能改回默认左腕。
- 在线归档是独立的 `piperx_review_v1`。失败/拒收轨迹不能直接传给原全成功示教reader，否则末帧可能被误标奖励1。已入池的轨迹不支持GUI追溯改标/撤回。

## 下一步与未验证事项

1. 先核验正式4000步checkpoint的完整性、恢复和推理，测真实模型延迟；不能仅凭目录存在宣称权重验收通过。
2. 完成原采集预检、检测器负例、工作空间与保持/单步/短动作块验收。当前成功规则是z>0.281807m、夹爪宽34.6～38.6mm、目标未要求张开，连续6个实际动作步；此前只做成功姿态只读标定，不能识别物体身份。
3. 测量CAN/相机反馈相对目标下发的滞后。当前只检查反馈新鲜度，T动作/T+1观测和SHA256证明记录、传输一致，不能证明曝光同步或目标已经到达。
4. 先完成固定关节归位路径与到位验收，再做GUI短回合、人工复核/拒收、最小在线更新和checkpoint恢复，最后扩大采集。`delay=5`仍需按实测验证；RL batch独立为4，不能把离线batch32容量结论套用到critic/UTD。

## 验证与维护方式

- 客户端Python3.10、服务器Python3.11、根轻量测试环境分开，使用 `uv run --locked`；不要合并环境或修改固定third_party算法。
- 改控制/数据流后运行 `uv run --locked python -m pytest -q` 和 `uv run --locked python scripts/smoke_rpc.py`。训练依赖变更另运行 `uv run --project server --locked python scripts/validate_server.py`。
- 本次仅新增/更新交接文档，没有重跑上述代码测试；39项测试是此前实际记录。
- 每完成重要功能、实验或部署后更新本文件的日期、当前状态、验证证据、未完成项与下一步；历史过程继续记录在 `docs/项目交接.md`，实验安排维护在 `docs/实验计划.md`。

## 详细资料

- [浏览器客户端使用说明](docs/浏览器客户端使用说明.md)
- [数据采集、转换与传输检查](docs/真机RL数据采集转换传输检查.md)
- [真机强化学习快速开始](docs/真机强化学习快速开始.md)
- [完整训练流程](docs/Real-Time-EXPO-FT完整训练流程.md)
- [RTC-SFT吞吐调优与正式运行](docs/RTC-SFT吞吐调优_20261031.md)
- [历史项目交接](docs/项目交接.md)与[实验计划](docs/实验计划.md)
