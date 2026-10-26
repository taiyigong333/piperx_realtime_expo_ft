# 工程约定

- 面向用户一律中文；重要注释解释单位、动作和硬件行为的原因。
- 4090服务器目录 `/data/gcj_data/rl/real_time_expo_ft/code`；piperx客户端目录 `/home/cv/gcj/rl/baseline/realtime_expo_ft`。两台机器通过192.168.50.2/192.168.50.1的有线接口通信。
- 外部代码使用 third_party 下 Git 子模块固定提交。先理解接口，尽量只修改 src/piperx_expo、配置、测试和文档。不要直接改上游算法。
- 客户端 Python3.10（ROS/Pika兼容），服务器 Python3.11（JAX/OpenPI），根环境用于轻量测试。不要合并三个环境。
- 动作是7维米制绝对法兰 XYZ+RPY 和夹爪宽度；禁止换成 DROID 速度或默认夹爪概率。
- 数据、权重、密钥、虚拟环境、运行日志不提交。私钥只在主仓库本地 core.sshCommand 中引用。
- 环境统一使用 uv sync/uv run 管理，不使用 pip/conda 混装。三个 uv.lock 应提交；运行使用 --locked。
- 改控制/数据流时运行 `uv run --locked python -m pytest -q` 和 `uv run --locked python scripts/smoke_rpc.py`；训练依赖变更时运行 `uv run --project server --locked python scripts/validate_server.py`。
- 真实运动前必须完成原采集预检与现场配置。默认进行模拟/只读验证；成功检测、工作空间、数据/权重不得猜测。
- docs/项目交接.md、docs/实验计划.md 及时维护。明确区分软件验证、GPU小算子更新、完整模型训练和真实机器人评估，不能互相替代。
