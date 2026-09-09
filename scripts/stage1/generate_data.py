"""Stage 1 数据生成器：TSP 固定数据集。

用法：
    python -m scripts.stage1.generate_data --num-loc 50 \
        --num-samples 10000 --split val --out-dir data/tsp50 --seed 1234
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import parse_key_values
from utils.data import make_env, save_td


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 Stage1 TSP 固定数据集")
    parser.add_argument("-e", "--env", default="TSPEnv")
    parser.add_argument("--num-loc", type=int, default=50)
    parser.add_argument("--num-samples", "-n", type=int, default=10_000)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("-o", "--out-dir", default=None)
    parser.add_argument("--filename", default=None)
    parser.add_argument(
        "-p",
        "--generator-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("-f", "--force", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    if args.env != "TSPEnv":
        raise SystemExit("Stage1 数据生成器只支持 TSPEnv；CVRP 数据请使用 Stage2 生成器")

    import numpy as np
    import torch

    env = make_env(
        "TSPEnv",
        num_loc=args.num_loc,
        generator_params=parse_key_values(args.generator_param),
    )
    if args.filename:
        out_path = Path(args.filename)
    else:
        out_path = (
            Path(args.out_dir) / f"{args.split}.npz"
            if args.out_dir
            else Path("data") / f"tsp{args.num_loc}" / f"{args.split}.npz"
        )
    out_path = (
        out_path if out_path.suffix == ".npz" else Path(f"{out_path}.npz")
    )

    if out_path.exists() and not args.force:
        print(f"{out_path} 已存在，跳过；如需覆盖请加 --force", file=sys.stderr)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    td = env.generator(args.num_samples)
    saved = save_td(td, out_path, compress=args.compress)

    print(f"环境: TSPEnv, num_loc={args.num_loc}, samples={args.num_samples}")
    print(f"字段: {', '.join(f'{k}{tuple(v.shape)}' for k, v in td.items())}")
    print(f"已保存: {saved.resolve()}")


if __name__ == "__main__":
    main()
