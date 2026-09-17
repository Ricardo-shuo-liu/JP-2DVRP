"""Train end-to-end RL4CO CVRP baselines.

This is intentionally separate from Stage2:

- Stage2 = partition network + frozen Stage1 TSP evaluator.
- This script = one neural model solving CVRP end-to-end.

Examples:

    # Attention Model
    python -m scripts.baselines.run_cvrp_model \
        -m AttentionModel \
        --num-loc 100 --num-vehicles 12 --capacity 50 \
        --epochs 100 --batch-size 512 --train-data-size 1280000

    # POMO
    python -m scripts.baselines.run_cvrp_model \
        -m POMO \
        --num-loc 100 --num-vehicles 12 --capacity 50 \
        --epochs 100 --batch-size 64 --train-data-size 100000
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl4co.envs import CVRPEnv

from utils.config import parse_key_values
from utils.data import FeasibleCVRPGenerator


SUPPORTED_MODELS = ("AttentionModel", "POMO")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Train an end-to-end RL4CO CVRP baseline."
    )
    parser.add_argument(
        "-m",
        "--model",
        default="AttentionModel",
        choices=SUPPORTED_MODELS,
        help="End-to-end CVRP model from RL4CO.",
    )
    parser.add_argument("--num-loc", type=int, default=100)
    parser.add_argument("--num-vehicles", type=int, default=12)
    parser.add_argument("--capacity", type=float, default=50.0)

    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument(
        "--num-encoder-layers",
        type=int,
        default=None,
        help="Defaults: 3 for AttentionModel, 6 for POMO.",
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument(
        "--normalization",
        default=None,
        choices=("batch", "instance"),
        help="Defaults: batch for AttentionModel, instance for POMO.",
    )
    parser.add_argument("--num-starts", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-data-size", type=int, default=1_280_000)
    parser.add_argument("--val-data-size", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--baseline",
        default="auto",
        choices=("auto", "rollout", "exponential", "shared", "no"),
    )
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="32")
    parser.add_argument("--logger", default="csv", choices=("csv", "none"))
    parser.add_argument("--log-dir", default="logs")

    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument(
        "-p",
        "--generator-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    return parser


def build_model(args: Namespace):
    from rl4co import models

    generator_params = parse_key_values(args.generator_param)
    generator_params["capacity"] = args.capacity
    generator = FeasibleCVRPGenerator(
        num_loc=args.num_loc,
        num_vehicles=args.num_vehicles,
        **generator_params,
    )
    env = CVRPEnv(
        generator=generator,
        data_dir=args.data_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        check_solution=False,
    )

    if args.model == "POMO":
        if args.baseline not in ("auto", "shared"):
            raise SystemExit("POMO 只支持 --baseline shared")
        args.baseline = "shared"
    elif args.baseline == "auto":
        args.baseline = "rollout"

    if args.num_encoder_layers is None:
        args.num_encoder_layers = 6 if args.model == "POMO" else 3
    if args.normalization is None:
        args.normalization = "instance" if args.model == "POMO" else "batch"

    policy_kwargs = {
        "embed_dim": args.embed_dim,
        "num_encoder_layers": args.num_encoder_layers,
        "num_heads": args.num_heads,
        "normalization": args.normalization,
    }
    model_cls = getattr(models, args.model)
    model_kwargs = dict(
        policy_kwargs=policy_kwargs,
        baseline=args.baseline,
        batch_size=args.batch_size,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        test_data_size=args.val_data_size,
        optimizer_kwargs={"lr": args.lr},
    )
    if args.model == "POMO":
        model_kwargs["num_starts"] = args.num_starts or None

    return model_cls(env, **model_kwargs)


def build_logger(args: Namespace):
    if args.logger == "none":
        return None
    from lightning.pytorch.loggers import CSVLogger

    return CSVLogger(
        save_dir=args.log_dir,
        name=f"baselines/CVRPEnv-{args.model}",
    )


def build_callbacks(args: Namespace, logger):
    from lightning.pytorch.callbacks import ModelCheckpoint

    if logger is not None:
        ckpt_dir = Path(logger.log_dir) / "checkpoints"
    else:
        ckpt_dir = (
            Path(args.log_dir)
            / "baselines"
            / f"CVRPEnv-{args.model}"
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


def main() -> None:
    args = build_parser().parse_args()

    import numpy as np
    import torch
    from rl4co.utils.trainer import RL4COTrainer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(args)
    logger = build_logger(args)
    callbacks = build_callbacks(args, logger)
    run_dir = (
        Path(logger.log_dir)
        if logger is not None
        else Path(args.log_dir)
        / "baselines"
        / f"CVRPEnv-{args.model}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str)
    )

    print(
        f"开始 CVRP baseline：{args.model}, "
        f"num_loc={args.num_loc}, num_vehicles={args.num_vehicles}"
    )
    print(f"日志目录: {run_dir.resolve()}")
    print(f"Checkpoint 目录: {Path(callbacks[0].dirpath).resolve()}")

    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=args.log_dir,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
