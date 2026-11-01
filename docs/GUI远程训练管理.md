# 机器人GUI远程训练管理

日期：2026-11-01。目标：机器人电脑单一GUI完成4090端RL开始、停止、状态和日志查看，恢复论文XYZ/夹爪edit模式。输入为固定本地配置及已有SSH别名；输出为独立RL运行、配置快照和日志。保留原CLI入口，不改变模型、数据表示、训练算法或三个uv环境。

## 使用

在机器人电脑运行原GUI命令并打开`http://127.0.0.1:8767`。具体命令见[浏览器使用说明](浏览器客户端使用说明.md)。点击“启动服务器 RL”→勾选现场运动确认并“连接学习器”→现场复位确认；结束时点击“停止服务器 RL”或顶部“停止并锁定”。不再需要手工SSH或tmux。停止后可以从同一页面启动新run，并重新确认现场、创建新客户端会话。

`configs/client/rotating_square.calibrated.yaml`的`remote_server`配置固定以下字段，HTTP不允许覆盖：

```yaml
remote_server:
  ssh_host: gh4090piper-wired
  project_dir: /data/gcj_data/rl/real_time_expo_ft/code
  config: configs/server/rotating_square.gui.yaml
```

SSH复用piperx已有密钥与主机校验，不写入密码或私钥，不改SSH权限。BatchMode及连接超时避免交互卡住GUI。没有此配置的mock或旧客户端继续使用原手工服务方式。两份旋转方块服务器配置均为`edit_action_xyzg: true`，允许改false做全7维edit对比；基础VLA始终输出完整动作。

## 依赖链与执行设计

`client/web`按钮→带原同源/令牌校验的HTTP command→`GUIController`→`client/remote_server.py`后台操作队列→有线SSH→服务器根环境`uv run --locked python -m piperx_expo.server.gui_service`→独立监督进程→原`bash scripts/server.sh rl --config <快照>`。

- SSH调用不占控制锁，不阻塞HTTP心跳；页面读取缓存，不在每次HTTP请求现场执行SSH。每约2秒更新状态与最后20KB日志，错误显示“状态未确认”。
- 启动前核对GUI专用配置、原有数据/权重/资产及端口，并拒绝GPU上已有计算进程；不自动停掉其他训练。GPU进程检查是启动前检查，不是跨所有外部程序的全局GPU锁。
- 使用文件锁串行化本管理器的启动/停止；重复启动返回已有受管任务，不重复加载模型。
- 每次生成`<原run_name>_<时间>_<随机后缀>`，保存`runtime/gui_server/<ID>/experiment.yaml`、`process.json`及`server.log`。实际模型、metrics和W&B继续使用原output_dir加新run名；配置快照保留原权重与归一化路径，不覆盖旧结果，不自动resume。
- 监督进程独立于SSH连接，并作为独立进程组组长。停止前同时检查PID、Linux starttime、命令行中的任务ID/worker标记及进程组身份，避免PID复用后误杀其他任务。
- 状态分为未启动、模型初始化、RPC端口已监听、正在停止、已停止、正常结束、异常退出。监听检查只读Linux `/proc/net/tcp*`，不发伪WebSocket连接，不抢占机器人客户端。
- 停止先关闭本机设备/连接，再向受管进程组SIGINT；等待12秒后仍在则SIGTERM，再等待5秒后SIGKILL兜底。只终止本管理器确认的进程组，不用pkill、不按端口杀进程。已结束任务保留退出码和日志。

## 会话与恢复边界

“启动服务器 RL”只启动学习器，不创建/使能机器人设备。机械臂仍受`--allow-motion`、现场运动勾选、逐回合复位确认、看门狗、工作空间与动作检查保护。重新建立会话时检查旧线程已退出、丢弃已关闭硬件对象并重新创建，不能解除旧对象的故障锁存；页面会清除运动确认勾选。

停止可能中断模型初始化或梯度更新，因此不宣称停止时自动新增最后checkpoint。已完成checkpoint、已ACK轨迹和日志保留。中断时最新模型与最新轨迹可能不同步；继续训练应使用既有GUI恢复流程及本次配置快照，不能把再次点击“启动”当作resume。SSH断开时监督进程会继续存在，页面未知状态不等于停止；恢复SSH后可再次点击停止。本地设备仍先执行停止。

服务器管理日志和配置快照位于被忽略的runtime，不提交Git；环境依赖未新增，uv.lock不变。

## 验证设计

本地验证HTTP/GUI命令边界、异步请求、先停本机后停远端、停止前尚未连接的重启及旧硬件对象锁存；Linux隔离副本使用真实子进程和独立模拟监听验证受管启停、重复启动、独立run、日志、端口冲突、PID身份及停止清理。模拟服务仅输出`MOCK GUI learner ready`，不加载模型、不访问CAN/相机。通过浏览器检查按钮、状态和日志布局；模拟和软件验收不能替代完整模型加载、真实在线RL或真机运动。

## 本轮实际验收

验证结果：本地52 passed、4 skipped（JAX及Linux /proc项）；4090隔离副本55 passed、1 skipped。模拟RPC、Ruff、JS语法通过。机器人Python3.10 GUI经既有有线SSH实际启动/停止4090的独立18107端口mock服务，页面显示就绪与日志，停止后独立确认running=false、phase=stopped、exit_code=-2；临时GUI和转发均已关闭。未加载完整模型、未启动真实RL或真机运动。

端到端验收仅在临时Python测试进程给SSH启动命令附加管理器已有的`--mock`，正式客户端及配置不含此覆盖。正式GUI按钮启动的仍是真实RL。模拟截图保存在本地被忽略的`runtime/validation/gui_remote/ssh_mock_ready.png`及`ssh_mock_stopped.png`。
