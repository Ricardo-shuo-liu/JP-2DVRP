"""Stage 1: 使用 RL4CO 选择环境/模型并训练。

默认按 test.ipynb 的方式，用 TSPEnv 环境训练 AttentionModel。
环境和模型均可通过命令行替换为 rl4co 中对应的类名。

用法（项目根目录下）：
    python scripts/stage1/run_stage1.py
    或
    python -m scripts.stage1.run_stage1
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

# 保证无论以脚本还是模块方式运行，都能找到项目根目录下的 utils/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config.stage1 import Config


class Getters:
    """根据配置获取并实例化 RL4CO 环境与可训练模型。"""

    def __init__(self, config: Config):
        self.config = config

    def get_env(self):
        from rl4co import envs

        env_cls = getattr(envs, self.config.env, None)
        if env_cls is None:
            raise ValueError(
                f"rl4co.envs 中不存在环境类 {self.config.env!r}；"
                "可使用 TSPEnv、CVRPEnv 等"
            )
        return env_cls(
            generator_params={"num_loc": self.config.num_loc},
            data_dir=self.config.data_dir,
            train_file=self.config.train_file,
            val_file=self.config.val_file,
            test_file=self.config.test_file,
        )

    def get_model(self, env=None):
        from rl4co import models

        model_cls = getattr(models, self.config.models, None)
        if model_cls is None:
            raise ValueError(
                f"rl4co.models 中不存在模型类 {self.config.models!r}；"
                "可使用 AttentionModel、POMO、MDAM 等"
            )
        if env is None:
            env = self.get_env()

        policy_kwargs = {
            "embed_dim": self.config.embed_dim,
            "num_encoder_layers": self.config.num_encoder_layers,
            "num_heads": self.config.num_heads,
        }
        return model_cls(
            env,
            policy_kwargs=policy_kwargs,
            batch_size=self.config.batch_size,
            train_data_size=self.config.train_data_size,
            val_data_size=self.config.val_data_size,
            optimizer_kwargs={"lr": self.config.lr},
        )


def build_parser() -> ArgumentParser:
    defaults = Config()
    parser = ArgumentParser(
        description="Stage 1: 选择 RL4CO 环境和模型进行训练"
    )

    # 环境 / 模型
    parser.add_argument(
        "-e",
        "--env",
        "--envs",
        dest="env",
        default=defaults.env,
        help="rl4co.envs 中的环境类名，如 TSPEnv",
    )
    parser.add_argument(
        "-m",
        "--model",
        "--models",
        dest="models",
        default=defaults.models,
        help="rl4co.models 中的模型类名，如 AttentionModel",
    )

    # 环境 / 网络结构
    parser.add_argument("--num-loc", type=int, default=defaults.num_loc)
    parser.add_argument("--embed-dim", type=int, default=defaults.embed_dim)
    parser.add_argument(
        "--num-encoder-layers",
        type=int,
        default=defaults.num_encoder_layers,
    )
    parser.add_argument("--num-heads", type=int, default=defaults.num_heads)

    # 训练参数
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument(
        "--train-data-size",
        type=int,
        default=defaults.train_data_size,
    )
    parser.add_argument(
        "--val-data-size",
        type=int,
        default=defaults.val_data_size,
    )

    # Trainer / 日志
    parser.add_argument(
        "--accelerator",
        default=defaults.accelerator,
        help="auto / cpu / gpu",
    )
    parser.add_argument("--devices", type=int, default=defaults.devices)
    parser.add_argument(
        "--precision",
        default=defaults.precision,
        help="如 16-mixed / 32 / bf16-mixed",
    )
    parser.add_argument(
        "--logger",
        default=defaults.logger,
        choices=("csv", "none"),
        help="csv 表示写入 CSV 日志，none 表示不记录",
    )
    parser.add_argument("--log-dir", default=defaults.log_dir)

    # 固定的数据文件（用 scripts/stage1/generate_data.py 生成后可在这里指定）
    parser.add_argument(
        "--data-dir",
        default=defaults.data_dir,
        help="环境数据目录，train/val/test 文件会拼接在此目录下",
    )
    parser.add_argument(
        "--train-file",
        default=defaults.train_file,
        help="固定的训练 npz 文件名；默认不设置，每 epoch 随机生成",
    )
    parser.add_argument(
        "--val-file",
        default=defaults.val_file,
        help="固定的验证 npz 文件名；默认不设置，验证时随机生成",
    )
    parser.add_argument(
        "--test-file",
        default=defaults.test_file,
        help="固定的测试 npz 文件名；默认不设置，测试时随机生成",
    )
    return parser


def get_args(parser: ArgumentParser) -> Dict:
    """解析命令行并过滤掉 None，便于直接构造 Config。"""
    namespace: Namespace = parser.parse_args()
    return {
        key: value
        for key, value in vars(namespace).items()
        if value is not None
    }


def _build_logger(config: Config):
    if config.logger == "none":
        return None
    from lightning.pytorch.loggers import CSVLogger

    # name 使用 stage1/<env>-<model>，CSVLogger 会自动追加 version_0/1/...
    name = f"stage1/{config.env}-{config.models}"
    return CSVLogger(save_dir=config.log_dir, name=name)


def _build_callbacks(config: Config, logger):
    """按官方 full-training 文档添加 checkpoint 回调。"""
    from lightning.pytorch.callbacks import ModelCheckpoint

    # 有 logger 时把 checkpoint 放到该次 run 的目录下；无 logger 时放在固定目录
    if logger is not None:
        ckpt_dir = Path(logger.log_dir) / "checkpoints"
    else:
        ckpt_dir = (
            Path(config.log_dir)
            / "stage1"
            / f"{config.env}-{config.models}"
            / "checkpoints"
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,
        monitor="val/reward",
        mode="max",
    )
    return [checkpoint_callback]


def main():
    config = Config(**get_args(build_parser()))

    from rl4co.utils.trainer import RL4COTrainer

    getters = Getters(config)

    env = getters.get_env()
    model = getters.get_model(env)
    print(
        f"开始训练：环境 {type(env).__name__}，"
        f"模型 {type(model).__name__}"
    )
    print(config.to_dict())

    logger = _build_logger(config)
    callbacks = _build_callbacks(config, logger)
    if logger is not None:
        run_dir = Path(logger.log_dir)
    else:
        print("logger=none：不记录 CSV 日志")
        run_dir = (
            Path(config.log_dir)
            / "stage1"
            / f"{config.env}-{config.models}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_file = run_dir / "config.json"
    config_file.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2)
    )
    if logger is not None:
        print(f"CSV 日志将写入: {run_dir.resolve()}")
    print(
        "Checkpoint 将写入: "
        f"{Path(callbacks[0].dirpath).resolve()}"
    )
    print(f"本次完整配置: {config_file.resolve()}")

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
