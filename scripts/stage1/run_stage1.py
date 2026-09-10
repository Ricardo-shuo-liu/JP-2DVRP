"""Stage 1：单簇/单车 TSP 路由模型训练。

用法（项目根目录下）：
    python -m scripts.stage1.run_stage1 -m AttentionModel --num-loc 20
    python -m scripts.stage1.run_stage1 -m POMO --num-loc 20 --epochs 5
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import parse_key_value
from utils.config.stage1 import Config
from utils.data import make_env


SUPPORTED_ENVS = ("TSPEnv",)
SUPPORTED_MODELS = ("AttentionModel", "POMO", "MDAM", "PointerNetwork")
MODEL_DEFAULTS = {"POMO": "shared"}


def get_model(env, config: Config):
    from rl4co import models

    model_cls = getattr(models, config.model, None)
    if model_cls is None or config.model not in SUPPORTED_MODELS:
        raise ValueError(
            f"不支持模型 {config.model!r}；可用: {', '.join(SUPPORTED_MODELS)}"
        )

    baseline = config.baseline
    if baseline == "auto":
        baseline = MODEL_DEFAULTS.get(config.model, "rollout")
    if config.model == "POMO" and baseline != "shared":
        raise ValueError("POMO 只支持 shared baseline")

    if config.model == "PointerNetwork":
        policy_kwargs = {"embed_dim": config.embed_dim, "hidden_dim": config.embed_dim}
    else:
        policy_kwargs = {
            "embed_dim": config.embed_dim,
            "num_encoder_layers": config.num_encoder_layers,
            "num_heads": config.num_heads,
            "normalization": config.normalization,
        }
        if config.model in ("AttentionModel", "POMO"):
            policy_kwargs["tanh_clipping"] = config.logit_clipping
            policy_kwargs["feedforward_hidden"] = config.feedforward_hidden
        if config.model == "MDAM":
            policy_kwargs["num_paths"] = config.mdam_num_paths
            policy_kwargs["eg_step_gap"] = config.mdam_eg_step_gap
            policy_kwargs["tanh_clipping"] = config.logit_clipping

    if config.model == "MDAM" and config.mdam_kl_loss != 0.0:
        from scripts.stage1.mdam_kl import MDAMWithKL

        model_cls = MDAMWithKL
    model_kwargs = dict(
        policy_kwargs=policy_kwargs,
        baseline=baseline,
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        test_batch_size=config.test_batch_size,
        train_data_size=config.train_data_size,
        val_data_size=config.val_data_size,
        optimizer_kwargs={
            "lr": config.lr,
            "weight_decay": config.weight_decay,
        },
    )
    if config.model == "MDAM":
        from scripts.stage1.mdam_fixed import FixedMDAMPolicy

        model_kwargs["policy"] = FixedMDAMPolicy(
            env_name=env.name, **policy_kwargs
        )
        model_kwargs.pop("policy_kwargs", None)
    if config.model == "POMO":
        model_kwargs["num_starts"] = config.num_starts
    if config.model == "MDAM" and config.mdam_kl_loss != 0.0:
        model_kwargs["kl_loss"] = config.mdam_kl_loss
    if config.lr_scheduler == "multi_step":
        from torch.optim.lr_scheduler import MultiStepLR

        model_kwargs["lr_scheduler"] = MultiStepLR
        model_kwargs["lr_scheduler_kwargs"] = {
            "milestones": config.lr_milestones,
            "gamma": config.lr_gamma,
        }
        model_kwargs["lr_scheduler_interval"] = "epoch"

    return model_cls(env, **model_kwargs)


def build_parser() -> ArgumentParser:
    defaults = Config()
    parser = ArgumentParser(description="Stage 1: 单簇 TSP 路由模型训练")
    parser.add_argument(
        "-e",
        "--env",
        default=defaults.env,
        help="固定为 TSPEnv；Stage1 是单车单簇问题，不需要 CVRPEnv",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=defaults.model,
        choices=SUPPORTED_MODELS,
        help="RL4CO 路由模型",
    )
    parser.add_argument("--num-loc", type=int, default=defaults.num_loc)
    parser.add_argument("--embed-dim", type=int, default=defaults.embed_dim)
    parser.add_argument(
        "--num-encoder-layers", type=int, default=defaults.num_encoder_layers
    )
    parser.add_argument("--num-heads", type=int, default=defaults.num_heads)
    parser.add_argument(
        "--normalization",
        default=defaults.normalization,
        choices=("batch", "instance"),
    )
    parser.add_argument(
        "--baseline",
        default=defaults.baseline,
        choices=("auto", "rollout", "shared", "exponential", "no"),
        help="auto: POMO->shared，其余->rollout",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=defaults.weight_decay,
        help="Adam weight decay",
    )
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=defaults.gradient_clip_val,
        help="gradient clipping value；MDAM 论文使用 1.0，0 表示不裁剪",
    )
    parser.add_argument(
        "--lr-scheduler",
        default=defaults.lr_scheduler,
        choices=("none", "multi_step"),
        help="学习率调度器；POMO 论文使用 multi_step",
    )
    parser.add_argument(
        "--lr-milestones",
        type=int,
        nargs="+",
        default=None,
        help="MultiStepLR 的 milestone；不指定时使用配置默认值 [501]",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=defaults.lr_gamma,
        help="MultiStepLR 的 gamma",
    )
    parser.add_argument(
        "--logit-clipping",
        type=float,
        default=defaults.logit_clipping,
        help="AM/POMO decoder 的 tanh logit clipping",
    )
    parser.add_argument(
        "--feedforward-hidden",
        type=int,
        default=defaults.feedforward_hidden,
        help="AM/POMO feed-forward hidden dimension",
    )
    parser.add_argument(
        "--num-starts",
        type=int,
        default=defaults.num_starts,
        help="POMO 多起点数量；不指定时使用节点数，例如 TSP20 为 20",
    )
    parser.add_argument(
        "--mdam-num-paths",
        type=int,
        default=defaults.mdam_num_paths,
        help="MDAM decoder 路径数，论文默认 5",
    )
    parser.add_argument(
        "--mdam-eg-step-gap",
        type=int,
        default=defaults.mdam_eg_step_gap,
        help="MDAM embedding-glimpse 间隔 n_EG，论文 TSP 配置为 2",
    )
    parser.add_argument(
        "--mdam-kl-loss",
        type=float,
        default=defaults.mdam_kl_loss,
        help="MDAM 路径多样性 KL 项系数，论文 TSP 配置为 0.01",
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--train-data-size", type=int, default=defaults.train_data_size)
    parser.add_argument("--val-data-size", type=int, default=defaults.val_data_size)
    parser.add_argument(
        "--val-batch-size", type=int, default=defaults.val_batch_size
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=defaults.test_batch_size
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--accelerator", default=defaults.accelerator)
    parser.add_argument("--devices", type=int, default=defaults.devices)
    parser.add_argument("--precision", default=defaults.precision)
    parser.add_argument(
        "--logger",
        default=defaults.logger,
        choices=("csv", "none"),
    )
    parser.add_argument("--log-dir", default=defaults.log_dir)
    parser.add_argument("--data-dir", default=defaults.data_dir)
    parser.add_argument("--train-file", default=defaults.train_file)
    parser.add_argument("--val-file", default=defaults.val_file)
    parser.add_argument("--test-file", default=defaults.test_file)
    parser.add_argument(
        "-p",
        "--generator-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="额外 generator 参数；Stage1 的 TSP 一般只需要 num_loc",
    )
    return parser


def args_to_config(parser: ArgumentParser) -> Config:
    namespace: Namespace = parser.parse_args()
    raw = vars(namespace).copy()
    generator_params = {
        key: value
        for key, value in (
            parse_key_value(item) for item in raw.pop("generator_param", [])
        )
    }
    raw = {key: value for key, value in raw.items() if value is not None}
    raw["generator_params"] = generator_params
    return Config(**raw)


def build_logger(config: Config):
    if config.logger == "none":
        return None
    from lightning.pytorch.loggers import CSVLogger

    return CSVLogger(
        save_dir=config.log_dir, name=f"stage1/{config.env}-{config.model}"
    )


def build_callbacks(config: Config, logger):
    from lightning.pytorch.callbacks import ModelCheckpoint

    if logger is not None:
        ckpt_dir = Path(logger.log_dir) / "checkpoints"
    else:
        ckpt_dir = (
            Path(config.log_dir)
            / "stage1"
            / f"{config.env}-{config.model}"
            / "checkpoints"
        )
    return [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch_{epoch:03d}",
            auto_insert_metric_name=False,
            save_top_k=1,
            save_last=True,
            monitor="val/reward",
            mode="max",
        )
    ]


def main():
    config = args_to_config(build_parser())

    if config.env not in SUPPORTED_ENVS:
        raise SystemExit(
            "Stage1 训练的是单车单簇 TSP，不接受 CVRPEnv。\n"
            "完整 CVRP 实例（多车、容量）由 Stage2 的 CVRPEnv 负责。"
        )

    import numpy as np
    import torch

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = make_env(
        "TSPEnv",
        num_loc=config.num_loc,
        data_dir=config.data_dir,
        train_file=config.train_file,
        val_file=config.val_file,
        test_file=config.test_file,
        generator_params=config.generator_params,
    )
    model = get_model(env, config)

    print(
        f"开始 Stage1：{type(env).__name__} / {config.model}，"
        f"num_loc={config.num_loc}"
    )
    print(config.to_dict())

    from rl4co.utils.trainer import RL4COTrainer

    logger = build_logger(config)
    callbacks = build_callbacks(config, logger)
    if logger is not None:
        run_dir = Path(logger.log_dir)
    else:
        run_dir = Path(config.log_dir) / "stage1" / f"{config.env}-{config.model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_file = run_dir / "config.json"
    config_file.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, default=str)
    )
    print(f"日志目录: {run_dir.resolve()}")
    print(f"Checkpoint 目录: {Path(callbacks[0].dirpath).resolve()}")
    print(f"配置文件: {config_file.resolve()}")

    trainer = RL4COTrainer(
        max_epochs=config.epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val or None,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=config.log_dir,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
