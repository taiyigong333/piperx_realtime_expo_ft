# Piper-X Real-Time EXPO-FT

用于“抓取旋转红色方块”的复现工程：机器人客户端负责数据采集、相机与机械臂执行，GPU 服务器负责 pi0.5 推理、RTC-SFT 与 Real-Time EXPO-FT 更新。

服务器位置：`gh4090piper:/data/gcj_data/rl/real_time_expo_ft/code`；客户端部署位置：`piperx:/home/cv/gcj/rl/baseline/realtime_expo_ft`。通过有线网192.168.50.2/192.168.50.1通信，部署步骤见[有线部署说明](docs/有线部署.md)。

当前四批数据的30帧pi0.5 checkpoint接入与完整训练步骤，见 [Real-Time EXPO-FT完整训练流程](docs/Real-Time-EXPO-FT完整训练流程.md)。本次现场配置和已执行验证见 [真机RL配置执行记录](docs/真机RL配置执行记录.md)。已完成数据/模型接入与只读标定，尚未启动完整RTC-SFT或真实RL；下方原始示例保留用于从原始数据搭建新任务。

## 目录

```text
client/                  # Python 3.10 客户端环境、依赖锁
server/                  # Python 3.11 JAX/CUDA 学习器环境、依赖锁
src/piperx_expo/
  client/                # Piper-X、RealSense、动作限制、看门狗、EXPO RPC
  data/                  # 原始 HDF5 -> EXPO 回放/LeRobot，归一化统计
  server/                # 独立模型/任务配置，SFT/RL/eval 启动
configs/
  collection/            # 参考相机参数与采集配置，默认不自动移动到初始位姿
  client/                # 模拟与真机配置样例
  server/                # 任务、数据、权重、RTC 参数
third_party/
  collector/             # 原采集仓库，含 pyAgxArm 子模块
  converter/             # 原 LeRobot 数据转换仓库
  expo/                  # 官方 EXPO 算法/训练循环
  openpi/                # pd-perry/openpi real-time-expo-ft 固定提交
scripts/                 # 环境准备、采集、启动与软件栈验证
tests/                   # 动作保护、协议、数据与归一化回归
docs/                    # 架构、操作说明、实验计划、交接
```

所有外部项目都是 Git 子模块；不复制算法源文件。`scripts/bootstrap.py` 生成 EXPO 所需 OpenPI 路径的相对符号链接，实际代码仍来自 `third_party/openpi`。不需要 DROID 或 ZED SDK。

## 安装与模拟验证

下面是4090上的完整开发验证；piperx只需 `bash scripts/setup.sh client`，不要安装服务器GPU环境。现场GitHub HTTPS不可用时使用 `PIPERX_GITHUB_SSH=1 bash scripts/setup.sh client`。

```bash
cd /data/gcj_data/rl/real_time_expo_ft/code
bash scripts/setup.sh core
bash scripts/setup.sh server
bash scripts/setup.sh client
uv run --locked python -m pytest -q
uv run --locked python scripts/smoke_rpc.py
uv run --project server --locked python scripts/validate_server.py
bash scripts/server.sh sft --dry-run
```

三个环境分别安装并记录 `uv.lock`。客户端保持 Python 3.10，兼容现有 ROS Humble/Pika；服务端使用 Python 3.11，与 JAX/OpenPI 依赖匹配。首次安装需要网络；GPU 栈验证还会获取 OpenPI 的 PaliGemma tokenizer 小型资源，不下载完整 pi0.5 权重。

## 数据与训练最短路径

1. 按[操作说明](docs/操作说明.md)准备采集配置，未来在机器人机器上运行预检、遥操采集。
2. 将已人工确认成功的 `batch_*/.../forward/trajectory.hdf5` 送到 GPU 服务器；反向轨迹不作为成功示教。
3. 转换回放、计算当前 Piper-X 数据的归一化资产：

```bash
uv run --project server --locked piperx-convert --input /绝对路径/raw --output /绝对路径/replay_v1 \
  --batches 2 3 4 --confirm-successful-demos
uv run --project server --locked piperx-norm --dataset /绝对路径/replay_v1 \
  --output assets/piperx_pi05/piperx/rotating_square
```

4. 复制 `configs/server/experiment.yaml` 到 `configs/local/experiment.yaml`，填写 `dataset_path`、`base_weights`，运行 RTC-SFT：

```bash
bash scripts/server.sh sft --config configs/local/experiment.yaml
```

5. 把生成的 SFT `.../checkpoints/<step>/params` 填入 `rtc_weights`。先通过现场动作范围、相机、延迟与成功判定验证，再分别启动真实客户端与学习器：

```bash
# GPU 机器，host 改为机器人可访问的 GPU 接口（例如 0.0.0.0）
bash scripts/server.sh rl --config configs/local/experiment.yaml
# 未来的机器人机器：先填写工作空间和真实成功检测器，再显式允许运动
bash scripts/client.sh --config configs/local/client.yaml --allow-motion
```

6. 填入 `eval_checkpoint_dir` 后执行 `bash scripts/server.sh eval --config configs/local/experiment.yaml`。断点恢复显式加 `--resume`。不要混用任务、资产和 RTC 参数。

单卡使用同步更新；`delay=5`、`replan_steps=8` 保留上游 RTC 分块预取。30Hz 是配置目标，真实端到端延迟、CAN 执行质量和显存峰值必须实测后再调。默认 batch=4 是单卡起点，不代表已经完成训练容量验证。

实现依据：[官方 EXPO-FT](https://github.com/pd-perry/expo-ft)、[Real-Time OpenPI 分支](https://github.com/pd-perry/openpi/tree/real-time-expo-ft)。版本和验证边界见[项目交接](docs/项目交接.md)。
