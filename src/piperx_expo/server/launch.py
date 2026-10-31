"""在不修改上游仓库的前提下注册配置、日志，并运行正式学习器。"""

import argparse
import json
import logging
import os
import runpy
import sys
from datetime import datetime

from piperx_expo.config import ROOT, load_config, resolve_path


def validate(config, stage):
    for key in ("dataset_path", "base_weights" if stage == "sft" else "rtc_weights"):
        if not config.get(key):
            raise ValueError(f"请填写 {key}，目前数据/权重尚未提供")
        if not resolve_path(config[key]).exists():
            raise FileNotFoundError(f"{key} 路径不存在：{resolve_path(config[key])}")
    horizon = config.get("action_horizon", 16)
    if not 1 <= config["replan_steps"] or 2 * config["replan_steps"] > horizon:
        raise ValueError("要求 action_horizon >= 2 * replan_steps，给RTC前缀与执行窗口保留空间")
    if not 0 <= config["delay"] <= config["replan_steps"]:
        raise ValueError("delay 必须在 [0,replan_steps] 内")
    if config["max_steps"] < config["batch_size"] or config["batch_size"] < 1:
        raise ValueError("max_steps/batch_size 不合法")
    dataset = resolve_path(config["dataset_path"])
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest["action_mode"] != "absolute_pose_width_m" or not manifest["confirmed_success"]:
        raise ValueError("必须使用已确认成功的 Piper-X 米制绝对动作回放")
    if manifest["language_instruction"] != config["language_instruction"]:
        raise ValueError("示教与任务指令不一致")
    if manifest["fps"] != config["control_hz"]:
        raise ValueError("示教与控制频率不一致")
    assets = resolve_path(config["assets_dir"]) / config["asset_id"]
    if not (assets / "norm_stats.json").is_file():
        raise FileNotFoundError(f"请先运行 piperx-norm：{assets}")
    provenance = json.loads((assets / "provenance.json").read_text())
    import hashlib

    if provenance["manifest_sha256"] != hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest():
        raise ValueError("归一化资产与本次数据 manifest 不匹配")
    if config.get("checkpoint_profile") == "piper_delta_right_wrist":
        if provenance.get("action_adapter") != "absolute_replay_delta_vla_v1":
            raise ValueError("已有delta checkpoint必须使用专用动作适配资产")
        if hashlib.sha256((assets / "vla_norm_stats.json").read_bytes()).hexdigest() != provenance["checkpoint_norm_sha256"]:
            raise ValueError("VLA归一化资产哈希不符")
    if stage == "eval" and not config.get("eval_checkpoint_dir"):
        raise ValueError("评估需要填写 eval_checkpoint_dir，避免静默评估未训练模型")
    if stage == "eval" and not resolve_path(config["eval_checkpoint_dir"]).is_dir():
        raise FileNotFoundError("eval_checkpoint_dir 不存在")
    if stage == "eval" and not any(
        path.is_dir() and path.name.isdigit()
        for path in resolve_path(config["eval_checkpoint_dir"]).iterdir()
    ):
        raise ValueError("评估目录没有数字命名的已保存 checkpoint，拒绝静默评估未训练模型")


def build_args(config, stage):
    package = ROOT / "src/piperx_expo/server"
    args = [
        f"--config_task={package / 'task_config.py'}",
        f"--config={package / 'model_config.py'}",
        f"--dataset_path={resolve_path(config['dataset_path']) if config.get('dataset_path') else '<待填写>'}",
        f"--num_data={config['num_data']}",
        f"--seed={config['seed']}",
        f"--replan_steps={config['replan_steps']}",
        "--fsdp_devices=1",
    ]
    if stage == "eval":
        args.extend(
            [
                f"--checkpoint_dir={resolve_path(config['eval_checkpoint_dir']) if config.get('eval_checkpoint_dir') else '<待填写>'}"
            ]
        )
    else:
        args.extend(
            [
                f"--batch_size={config['batch_size']}",
                f"--max_steps={config['max_steps']}",
                "--project_name=piperx_realtime_expo_ft",
                f"--run_name={config['run_name']}_{stage}",
                f"--output_dir={resolve_path(config['output_dir'])}",
                f"--checkpoint_interval={config['checkpoint_interval']}",
                "--checkpoint_model=True",
            ]
        )
    if stage != "sft":
        args.extend(
            [
                f"--delay={config['delay']}",
                f"--client_host={config['host']}",
                f"--client_port={config['port']}",
            ]
        )
    if stage == "rl":
        args.extend(
            [
                "--update_type=episode",
                f"--step_interval={config['step_interval']}",
                f"--offline_ratio={config['offline_ratio']}",
                "--checkpoint_buffer=True",
            ]
        )
    return args


def _install_local_log(config, stage):
    import numpy as np
    import wandb

    path = resolve_path(config["output_dir"]) / f"{config['run_name']}_{stage}" / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    original = None
    original_init = wandb.init

    def log(data, *args, **kwargs):
        scalars = {}
        for key, value in data.items():
            try:
                array = np.asarray(value)
                if array.size == 1 and array.dtype.kind in "bifu":
                    scalars[key] = float(array.item())
            except (TypeError, ValueError):
                pass
        entry = {"time": datetime.now().astimezone().isoformat(), "step": kwargs.get("step"), **scalars}
        with path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        metrics = {
            k: v
            for k, v in scalars.items()
            if any(s in k.lower() for s in ("loss", "lr", "success", "return"))
        }
        if metrics:
            logging.info("step=%s %s", kwargs.get("step"), metrics)
        return original(data, *args, **kwargs)

    def initialize(*args, **kwargs):
        nonlocal original
        run = original_init(*args, **kwargs)
        # wandb.init会重新绑定wandb.log，必须在init之后安装本地实时日志。
        original = wandb.log
        wandb.log = log
        return run

    wandb.init = initialize


def _load_wandb_environment(config):
    import shlex

    if config["wandb_mode"] != "online":
        return
    if config.get("wandb_env_file"):
        # 只读取已有凭据文件中的白名单赋值，不执行shell，也不把密钥复制进项目。
        for line in resolve_path(config["wandb_env_file"]).read_text().splitlines():
            key, separator, value = line.strip().removeprefix("export ").strip().partition("=")
            key = key.strip()
            if separator and key in {"WANDB_API_KEY", "WANDB_ENTITY", "WANDB_BASE_URL"}:
                parts = shlex.split(value, comments=True)
                if len(parts) != 1:
                    raise ValueError(f"W&B凭据文件中的 {key} 必须是单值赋值")
                if not os.environ.get(key):
                    os.environ[key] = parts[0]
    if config.get("wandb_entity"):
        os.environ["WANDB_ENTITY"] = config["wandb_entity"]
    if config.get("wandb_base_url"):
        os.environ["WANDB_BASE_URL"] = config["wandb_base_url"]


def _install_rpc_contract(config):
    import time
    from expo_ft.env.env_client import EnvClient, EnvClientWrapper
    from openpi_client import msgpack_numpy

    original = EnvClient.create_env

    def create_env(self, request):
        spec = {key: config[key] for key in ("language_instruction", "control_hz", "episode_steps")}
        spec["action_mode"] = "absolute_pose_width_m"
        return original(self, {**request, "task_spec": spec})

    EnvClient.create_env = create_env

    def call_once(self, operation, request):
        # 断线时无法知道上一条动作是否已执行，禁止上游自动重连后重发动作。
        try:
            connection = self._get_connection()
            connection.send(self._packer.pack({"operation": operation, **request}))
            start = time.monotonic()
            response = msgpack_numpy.unpackb(connection.recv())
            self.network_ms += (time.monotonic() - start) * 1000
            if response.get("status") == "error":
                raise RuntimeError(response.get("message", "客户端错误"))
            return response
        except Exception:
            self._close_connection()
            raise

    EnvClient._call_operation = call_once
    EnvClientWrapper._call = lambda self, operation, thunk: thunk()


def _install_checkpoint_init():
    import expo_ft.agents
    from pathlib import Path

    original = expo_ft.agents.initialize_checkpoint_dir

    def initialize(path, *, keep_period, overwrite, resume):
        folder = Path(path)
        if folder.is_dir() and not overwrite and not resume and not any(folder.iterdir()):
            # 固定上游入口先mkdir再初始化，导致全新run也被误判为覆盖。
            # rmdir只接受空目录，绝不删除已有checkpoint或其他文件。
            folder.rmdir()
        return original(path, keep_period=keep_period, overwrite=overwrite, resume=resume)

    expo_ft.agents.initialize_checkpoint_dir = initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["sft", "rl", "eval"])
    parser.add_argument("--config", default="configs/server/experiment.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args, extra = parser.parse_known_args()
    config = load_config(args.config)
    entry = {"sft": "train_offline_rtc.py", "rl": "train_pi_robo.py", "eval": "eval_droid_policy.py"}[
        args.stage
    ]
    argv = [str(ROOT / "third_party/expo" / entry), *build_args(config, args.stage), *extra]
    if args.resume and args.stage != "eval":
        argv.append("--resume=True")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "argv": argv,
                    "note": "仅展示配置，不加载模型、不连接设备；正式启动会检查必填路径",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    validate(config, args.stage)
    _load_wandb_environment(config)
    os.environ.update(
        PIPERX_EXPERIMENT_CONFIG=str(resolve_path(args.config)),
        PIPERX_EXPO_STAGE=args.stage,
        CUDA_VISIBLE_DEVICES=str(config["cuda_visible_devices"]),
        WANDB_MODE=config["wandb_mode"],
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.85",
    )
    os.chdir(ROOT / "third_party/expo")
    sys.path.insert(0, str(ROOT / "third_party/expo"))
    logging.basicConfig(level=logging.INFO)
    from piperx_expo.server.openpi_config import register

    register(config)
    if config.get("checkpoint_profile") == "piper_delta_right_wrist":
        from piperx_expo.server.checkpoint_adapter import install
        install()
    _install_local_log(config, args.stage)
    if args.stage != "eval":
        _install_checkpoint_init()
    if args.stage != "sft":
        _install_rpc_contract(config)
    sys.argv = argv
    runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
