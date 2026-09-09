"""Stage 2 数据生成器：RL4CO CVRPEnv 固定数据集。

用法：
    python -m scripts.stage2.generate_data --num-loc 50 --num-vehicles 8 \
        --num-samples 10000 --split train --out-dir data/cvrp50 --seed 1234

生成文件中的 demand 为未归一化整数，capacity 为原始容量；加载时会由
``CVRPEnv.load_data`` 归一化一次。样本均满足
``sum(demand) <= num_vehicles * capacity``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tensordict import TensorDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import canonical_cvrp_td, feasible_cvrp_mask, make_env, save_td


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 Stage2 CVRP 固定数据集")
    parser.add_argument("-e", "--env", default="CVRPEnv")
    parser.add_argument("--num-loc", type=int, default=50)
    parser.add_argument("--capacity", type=float, default=None)
    parser.add_argument("--num-vehicles", type=int, default=8)
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


def parse_key_value(item: str) -> tuple[str, object]:
    if "=" not in item:
        raise ValueError(f"参数 {item!r} 格式应为 KEY=VALUE")
    key, raw = item.split("=", 1)
    raw = raw.strip()
    if raw in ("True", "true"):
        return key.strip(), True
    if raw in ("False", "false"):
        return key.strip(), False
    if raw in ("None", "null"):
        return key.strip(), None
    try:
        return key.strip(), int(raw)
    except ValueError:
        try:
            return key.strip(), float(raw)
        except ValueError:
            return key.strip(), raw


def main():
    args = build_parser().parse_args()
    if args.env != "CVRPEnv":
        raise SystemExit("Stage2 数据生成器只支持 CVRPEnv")

    generator_params = dict(
        parse_key_value(item) for item in args.generator_param
    )
    if args.capacity is not None:
        generator_params["capacity"] = args.capacity

    env = make_env(
        "CVRPEnv",
        num_loc=args.num_loc,
        generator_params=generator_params,
        check_solution=False,
    )
    out_path = (
        Path(args.filename)
        if args.filename
        else Path(args.out_dir or f"data/cvrp{args.num_loc}") / f"{args.split}.npz"
    )
    out_path = (
        out_path if out_path.suffix == ".npz" else Path(f"{out_path}.npz")
    )
    if out_path.exists() and not args.force:
        print(f"{out_path} 已存在，跳过；如需覆盖请加 --force", file=sys.stderr)
        return

    torch.manual_seed(args.seed)
    accepted: list[TensorDict] = []
    accepted_count = 0
    batch_size = min(1024, max(args.num_samples, 1))

    while accepted_count < args.num_samples:
        td = env.generator(batch_size)
        mask = feasible_cvrp_mask(td, args.num_vehicles)
        if mask.any():
            keep = mask
            remaining = args.num_samples - accepted_count
            if int(keep.sum()) > remaining:
                # keep only the first `remaining` feasible samples
                idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)[:remaining]
                keep = torch.zeros_like(mask)
                keep[idx] = True
            part = canonical_cvrp_td(td[keep])
            part.set(
                "num_vehicles",
                torch.full(
                    (part.batch_size[0], 1),
                    args.num_vehicles,
                    dtype=torch.int64,
                ),
            )
            accepted.append(part)
            accepted_count += part.batch_size[0]

    td_all = torch.cat(accepted, dim=0)
    saved = save_td(td_all, out_path, compress=args.compress)
    print(
        f"CVRP: num_loc={args.num_loc}, capacity={env.generator.capacity}, "
        f"num_vehicles={args.num_vehicles}, samples={td_all.batch_size[0]}"
    )
    print(
        "字段: "
        + ", ".join(f"{key}{tuple(value.shape)}" for key, value in td_all.items())
    )
    print(f"已保存: {saved.resolve()}")


if __name__ == "__main__":
    main()
