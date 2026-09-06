"""生成 RL4CO 指定环境的随机数据（保存为 .npz）。

用法示例：
    # 生成 TSP50 的 1 万个测试样本，用于比较不同模型
    python scripts/stage1/generate_data.py -e TSPEnv --num-loc 50 --num-samples 10000 \
        --split test --out-dir data/tsp50 --seed 1234

生成的 train.npz / val.npz / test.npz 可直接在 run_stage1.py 中指定：
    python scripts/stage1/run_stage1.py -e TSPEnv -m AttentionModel \
        --data-dir data/tsp50 --val-file val.npz --test-file test.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成 RL4CO 指定环境的随机数据集（npz）"
    )
    parser.add_argument(
        "-e",
        "--env",
        default="TSPEnv",
        help="rl4co.envs 中的环境类名，如 TSPEnv、CVRPEnv、PDPEnv",
    )
    parser.add_argument(
        "--num-loc",
        type=int,
        default=50,
        help="节点数，写入 generator 的 num_loc 参数（默认 50）",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=10_000,
        help="样本数（默认 10000）",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="数据集阶段，默认 test",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default=None,
        help=(
            "输出目录；不指定时默认 data/<env 小写名><num_loc>，"
            "例如 data/tsp50"
        ),
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="完整输出文件名；不指定时使用 <out-dir>/<split>.npz",
    )
    parser.add_argument(
        "-p",
        "--generator-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "额外 generator 参数，可重复传入，例如 "
            "-p vehicle_capacity=1.0 -p loc_distribution=gaussian_mixture"
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compress", action="store_true", help="保存为压缩 npz")
    parser.add_argument("-f", "--force", action="store_true", help="覆盖已存在文件")
    return parser


def parse_generator_params(items: list[str]) -> dict:
    params = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"generator 参数 {item!r} 格式应为 KEY=VALUE"
            )
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if raw in ("True", "true"):
            value = True
        elif raw in ("False", "false"):
            value = False
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        params[key] = value
    return params


def add_extension(path: Path) -> Path:
    return path if path.suffix == ".npz" else Path(f"{path}.npz")


def main():
    args = build_parser().parse_args()

    # 延迟导入，保证只看 --help 时不需要安装/初始化 rl4co
    import numpy as np
    import torch
    from rl4co import envs
    from rl4co.data.utils import save_tensordict_to_npz

    env_cls = getattr(envs, args.env, None)
    if env_cls is None:
        available = ", ".join(
            name
            for name in dir(envs)
            if name.endswith("Env") and not name.startswith("_")
        )
        raise SystemExit(f"rl4co.envs 中不存在 {args.env!r}。可用环境: {available}")

    generator_params = {"num_loc": args.num_loc}
    generator_params.update(parse_generator_params(args.generator_param))

    env = env_cls(generator_params=generator_params)

    if args.filename is not None:
        out_path = add_extension(Path(args.filename))
    else:
        out_dir = (
            Path(args.out_dir)
            if args.out_dir is not None
            else Path("data") / f"{env.name}{args.num_loc}"
        )
        out_path = add_extension(out_dir / f"{args.split}.npz")

    if out_path.exists() and not args.force:
        print(
            f"{out_path} 已存在，跳过。如需覆盖请加 --force。",
            file=sys.stderr,
        )
        return

    # 同时固定 torch 和 numpy 的随机种子，保证多次生成结果一致
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    td = env.generator(batch_size=args.num_samples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_tensordict_to_npz(td, out_path, compress=args.compress)

    print(f"环境: {args.env} ({env.name})")
    print(f"generator 参数: {generator_params}")
    print(f"样本数: {args.num_samples}, seed: {args.seed}")
    for key, value in td.items():
        print(f"  {key}: {tuple(value.shape)}")
    print(f"已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()
