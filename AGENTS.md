# 工程约定

- 面向用户一律中文；重要注释解释单位、动作和硬件行为的原因。
- 4090服务器目录 `/data/gcj_data/rl/real_time_expo_ft/code`；piperx客户端目录 `/home/cv/gcj/rl/baseline/realtime_expo_ft`。两台机器通过192.168.50.2/192.168.50.1的有线接口通信。
- 外部代码使用 third_party 下 Git 子模块固定提交。先理解接口，尽量只修改 src/piperx_expo、配置、测试和文档。不要直接改上游算法。
- 客户端 Python3.10（ROS/Pika兼容），服务器 Python3.11（JAX/OpenPI），根环境用于轻量测试。不要合并三个环境。
- 动作是7维米制绝对法兰 XYZ+RPY 和夹爪宽度；禁止换成 DROID 速度或默认夹爪概率。
- 数据、权重、密钥、虚拟环境、运行日志不提交。私钥只在主仓库本地 core.sshCommand 中引用。
- 使用Git维护项目：每完成一项可独立复核的阶段性修改并通过相应验证，及时提交并推送到远程仓库；结束工作前检查状态并报告提交/推送结果。推送前先核对远程更新，不使用强制推送。双机同步同一提交，保留被忽略的现场配置与运行产物。
- 环境统一使用 uv sync/uv run 管理，不使用 pip/conda 混装。三个 uv.lock 应提交；运行使用 --locked。
- 改控制/数据流时运行 `uv run --locked python -m pytest -q` 和 `uv run --locked python scripts/smoke_rpc.py`；训练依赖变更时运行 `uv run --project server --locked python scripts/validate_server.py`。
- 真实运动前必须完成原采集预检与现场配置。默认进行模拟/只读验证；成功检测、工作空间、数据/权重不得猜测。
- docs/项目交接.md维护当前事实，docs/实验计划.md维护下一步，及时更新；历史过程按docs/README.md分类归档到docs/历史/<类别>/YYYY-MM-DD_主题.md。明确区分软件验证、GPU小算子更新、完整模型训练和真实机器人评估，不能互相替代。
- 本地根目录 `HANDOFF.md` 作为新会话接手的当前状态快照，重要修改/实验/部署后及时更新；历史证据保留在docs/历史/分类日期文件中，docs/项目交接.md只保留当前结论与历史入口。

- 常用操作、接口约定及持续维护文档直接放在docs/根目录，不加日期前缀；一次性实验、部署、设计复盘放历史分类目录并加实际记录日期。跨日期原文快照以前缀标识快照日期，不篡改正文事件日期。迁移后同步修正链接，并保留原始历史内容。
