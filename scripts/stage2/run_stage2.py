"""Stage 2：CVRP 分簇训练，使用冻结的 Stage1 TSP 模型作为 reward 评价器。

用法（项目根目录下）：
    python -m scripts.stage2.run_stage2 --num-loc 50 --num-vehicles 8 \
        --route-model AttentionModel \
        --route-checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nets.partition import PartitionNetwork
from utils.config import parse_key_value
from utils.config.stage2 import Config
from rl4co.envs import CVRPEnv

from utils.data import FeasibleCVRPGenerator


def build_policy(config: Config):
    from scripts.stage2.policy import Stage2Policy
    from scripts.stage2.route_evaluator import RouteEvaluator

    route_evaluator = RouteEvaluator(config)
    partition_network = PartitionNetwork(
        embed_dim=config.embed_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_heads=config.num_heads,
        temperature=config.temperature,
    )
    return Stage2Policy(
        partition_network=partition_network,
        route_evaluator=route_evaluator,
        num_vehicles=config.num_vehicles,
        objective=config.objective,
        penalty=config.penalty,
        capacity_mask_mode=config.capacity_mask_mode,
    )


def build_model(config: Config):
    from rl4co.models import REINFORCE

    if config.baseline == "shared":
        raise ValueError(
            "Stage2 policy 的 reward 形状是 [B]，RL4CO SharedBaseline "
            "需要额外的 reward 维度；请使用 rollout、exponential 或 no。"
        )
    generator_params = dict(config.generator_params)
    if config.capacity is not None:
        generator_params["capacity"] = config.capacity
    generator = FeasibleCVRPGenerator(
        num_loc=config.num_loc,
        num_vehicles=config.num_vehicles,
        **generator_params,
    )
    env = CVRPEnv(
        generator=generator,
        data_dir=config.data_dir,
        train_file=config.train_file,
        val_file=config.val_file,
        test_file=config.test_file,
        check_solution=False,
    )
    policy = build_policy(config)
    model = REINFORCE(
        env,
        policy,
        baseline=config.baseline,
        batch_size=config.batch_size,
        train_data_size=config.train_data_size,
        val_data_size=config.val_data_size,
        test_data_size=config.val_data_size,
        optimizer_kwargs={"lr": config.lr},
    )
    return model


def build_parser() -> ArgumentParser:
    defaults = Config()
    parser = ArgumentParser(
        description="Stage 2: CVRP 分簇 + 冻结 Stage1 TSP reward 评价器"
    )
    parser.add_argument("-e", "--env", default=defaults.env)
    parser.add_argument("--num-loc", type=int, default=defaults.num_loc)
    parser.add_argument("--capacity", type=float, default=defaults.capacity)
    parser.add_argument("--num-vehicles", type=int, default=defaults.num_vehicles)
    parser.add_argument(
        "--objective",
        default=defaults.objective,
        choices=("sum", "minmax"),
        help="sum: minimize total route length; minmax: minimize the longest route",
    )
    parser.add_argument("--penalty", type=float, default=defaults.penalty)
    parser.add_argument(
        "--capacity-mask-mode",
        default=defaults.capacity_mask_mode,
        choices=("none", "static"),
        help="capacity mask on the [vehicle, node] assignment matrix",
    )
    parser.add_argument(
        "--baseline",
        default=defaults.baseline,
        choices=("rollout", "exponential", "no"),
        help="Stage2 policy reward is [B]; shared baseline is not supported",
    )

    parser.add_argument("--embed-dim", type=int, default=defaults.embed_dim)
    parser.add_argument(
        "--num-encoder-layers", type=int, default=defaults.num_encoder_layers
    )
    parser.add_argument("--num-heads", type=int, default=defaults.num_heads)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument(
        "--route-pad-to",
        type=int,
        default=defaults.route_pad_to,
        help="minimum number of customer slots per padded cluster; 0 uses batch max",
    )
    parser.add_argument(
        "--route-num-starts",
        type=int,
        default=defaults.route_num_starts,
        help="POMO starts for the route evaluator; 0 uses the padded cluster length",
    )
    parser.add_argument(
        "--route-eval-batch-size",
        type=int,
        default=defaults.route_eval_batch_size,
        help="maximum clusters per TSP forward; 0 evaluates all clusters at once",
    )

    parser.add_argument(
        "--route-model",
        default=defaults.route_model,
        choices=("AttentionModel", "POMO", "MDAM", "PointerNetwork"),
    )
    parser.add_argument("--route-checkpoint", default=defaults.route_checkpoint)
    parser.add_argument("--route-embed-dim", type=int, default=defaults.route_embed_dim)
    parser.add_argument(
        "--route-num-encoder-layers",
        type=int,
        default=defaults.route_num_encoder_layers,
    )
    parser.add_argument(
        "--route-num-heads", type=int, default=defaults.route_num_heads
    )
    parser.add_argument(
        "--route-normalization",
        default=defaults.route_normalization,
        choices=("batch", "instance"),
    )

    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--train-data-size", type=int, default=defaults.train_data_size)
    parser.add_argument("--val-data-size", type=int, default=defaults.val_data_size)
    parser.add_argument("--accelerator", default=defaults.accelerator)
    parser.add_argument("--devices", type=int, default=defaults.devices)
    parser.add_argument("--precision", default=defaults.precision)
    parser.add_argument("--logger", default=defaults.logger, choices=("csv", "none"))
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

    return CSVLogger(save_dir=config.log_dir, name="stage2/CVRPEnv-cluster")


def build_callbacks(config: Config, logger):
    from lightning.pytorch.callbacks import ModelCheckpoint

    if logger is not None:
        ckpt_dir = Path(logger.log_dir) / "checkpoints"
    else:
        ckpt_dir = Path(config.log_dir) / "stage2" / "CVRPEnv-cluster" / "checkpoints"
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
    if config.env != "CVRPEnv":
        raise SystemExit("Stage2 目前只支持 RL4CO CVRPEnv")

    model = build_model(config)
    print(f"开始 Stage2：{config.env}, num_loc={config.num_loc}, "
          f"num_vehicles={config.num_vehicles}, objective={config.objective}")
    print(f"route model: {config.route_model}, route ckpt: {config.route_checkpoint}")
    print(config.to_dict())

    from rl4co.utils.trainer import RL4COTrainer

    logger = build_logger(config)
    callbacks = build_callbacks(config, logger)
    if logger is not None:
        run_dir = Path(logger.log_dir)
    else:
        run_dir = Path(config.log_dir) / "stage2" / "CVRPEnv-cluster"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_file = run_dir / "config.json"
    config_file.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, default=str)
    )
    print(f"日志目录: {run_dir.resolve()}")
    print(f"Checkpoint 目录: {Path(callbacks[0].dirpath).resolve()}")

    trainer = RL4COTrainer(
        max_epochs=config.epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        precision=config.precision,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=config.log_dir,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
