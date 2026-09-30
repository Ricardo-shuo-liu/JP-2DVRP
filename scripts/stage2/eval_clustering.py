"""Stage2 分簇模型测试：只统计分簇成功率，不做任何路径规划。

用法（服务器项目根目录）：

    python -m scripts.stage2.eval_clustering \
        --data-file data/cvrp100_test/test.npz \
        --checkpoint logs/stage2/CVRPEnv-cluster/version_0/checkpoints/last.ckpt \
        --num-instances 1000 --num-vehicles 12 \
        --batch-size 256 --device cuda \
        --out assets/results/cvrp100_clustering_success.json

"分簇成功" 定义为：该实例的每个簇都满足容量约束（sum(demand) <= capacity）。
脚本只做一次前向传播，不加载 Stage1 路由模型，也不调用 LKH/OR-Tools，
所以 1000 个实例通常只需几秒到几十秒。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nets.partition import PartitionNetwork  # noqa: E402

try:  # rl4co 依赖里自带 tensordict
    from tensordict import TensorDict
except ImportError:  # pragma: no cover - 退化为普通 dict
    TensorDict = None


def load_cvrp_npz(path):
    """读取 Stage2 生成的 CVRP npz（locs 只含客户，depot 单独存放）。"""
    with np.load(path) as data:
        files = set(data.files)
        for key in ("locs", "depot", "demand"):
            if key not in files:
                raise SystemExit(f"{path} 缺少 {key} 字段：{sorted(files)}")
        locs = np.asarray(data["locs"], dtype=np.float64)
        depot = np.asarray(data["depot"], dtype=np.float64)
        demand = np.asarray(data["demand"], dtype=np.float64)
        capacity = (
            np.asarray(data["capacity"], dtype=np.float64).reshape(-1)
            if "capacity" in files
            else np.ones(locs.shape[0])
        )
        vehicles = (
            np.asarray(data["num_vehicles"]).reshape(-1).astype(int)
            if "num_vehicles" in files
            else None
        )
    if demand.ndim == 3:
        demand = demand.squeeze(-1)
    if demand.shape[-1] == locs.shape[1] + 1:  # 少数文件把仓库需求一起存了
        demand = demand[..., 1:]
    if depot.ndim == 2:
        depot = depot[:, None, :]
    if demand.shape[-1] != locs.shape[1]:
        raise SystemExit(f"demand 形状 {demand.shape} 与 locs {locs.shape} 不匹配")
    return {
        "locs": locs,
        "depot": depot,
        "demand": demand,
        "capacity": capacity,
        "num_vehicles": vehicles,
    }


def decode_assignments(logits, demand_norm, valid_vehicle, mask_mode, sample):
    """[B,M,N] logits -> [B,N] 车辆编号，镜像 Stage2Policy 的推理解码。"""
    if mask_mode == "static":
        # demand 已按 capacity 归一化，因此单车容量恒为 1
        capacity = torch.ones(
            demand_norm.shape[0], 1, dtype=logits.dtype, device=logits.device
        )
        feasible = demand_norm.unsqueeze(1) <= (capacity.unsqueeze(-1) + 1e-6)
        feasible = feasible & valid_vehicle[:, :, None]
        any_feasible = feasible.any(dim=1, keepdim=True)
        # 没有可行车辆时放开所有有效车辆，交给后验惩罚（与训练一致）
        feasible = torch.where(any_feasible, feasible, valid_vehicle[:, :, None])
        logits = logits.masked_fill(~feasible, float("-inf"))
    if not sample:
        return logits.argmax(dim=1)
    probs = torch.softmax(logits, dim=1)
    flat = probs.permute(0, 2, 1).reshape(-1, probs.shape[1])
    return torch.multinomial(flat, 1).reshape(logits.shape[0], logits.shape[2])


def batch_cluster_stats(assign, demand, capacity, num_vehicles):
    """assign [R,N] -> 每行的载重 / 簇大小 / 超载统计。"""
    rows = assign.shape[0]
    loads = np.zeros((rows, num_vehicles))
    sizes = np.zeros((rows, num_vehicles), dtype=int)
    for vehicle in range(num_vehicles):
        mask = assign == vehicle
        sizes[:, vehicle] = mask.sum(axis=1)
        loads[:, vehicle] = np.where(mask, demand, 0.0).sum(axis=1)
    overload = np.clip(loads - capacity[:, None], 0.0, None)
    return {
        "sizes": sizes,
        "overload_total": overload.sum(axis=1),
        "invalid_clusters": (overload > 1e-6).sum(axis=1),
        "vehicles_used": (sizes > 0).sum(axis=1),
        "max_load_ratio": (loads / capacity[:, None]).max(axis=1),
    }


def evaluate(net, data, indices, num_vehicles, args, device):
    records = []
    started = time.perf_counter()
    demand_all = np.asarray(data["demand"])
    capacity_all = np.asarray(data["capacity"])
    depot_all = np.asarray(data["depot"])
    locs_all = np.asarray(data["locs"])

    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            chunk = indices[start : start + args.batch_size]
            demand_raw = demand_all[chunk]
            capacity_raw = capacity_all[chunk]
            # 按构造就不可行的实例（总需求超过车队容量）单独统计，不混进成功率
            structurally_ok = (
                demand_raw.sum(axis=1) <= capacity_raw * num_vehicles + 1e-9
            )

            depot = torch.as_tensor(depot_all[chunk], dtype=torch.float32, device=device)
            customers = torch.as_tensor(locs_all[chunk], dtype=torch.float32, device=device)
            demand_norm = torch.as_tensor(
                demand_raw / capacity_raw[:, None], dtype=torch.float32, device=device
            )
            td = {
                "locs": torch.cat([depot, customers], dim=1),
                "demand": demand_norm,
                "vehicle_capacity": torch.ones(
                    len(chunk), 1, dtype=torch.float32, device=device
                ),
            }
            if TensorDict is not None:
                td = TensorDict(td, batch_size=[len(chunk)], device=device)

            valid = torch.ones(len(chunk), num_vehicles, dtype=torch.bool, device=device)
            logits = net(td, num_vehicles)

            if args.decode == "greedy":
                assign = decode_assignments(
                    logits, demand_norm, valid, args.capacity_mask_mode, False
                )
                stats = batch_cluster_stats(
                    assign.detach().cpu().numpy(), demand_raw, capacity_raw, num_vehicles
                )
                for row, index in enumerate(chunk):
                    records.append(
                        {
                            "index": int(index),
                            "structurally_infeasible": bool(not structurally_ok[row]),
                            "feasible": bool(stats["overload_total"][row] <= 1e-6),
                            "overload_total": float(stats["overload_total"][row]),
                            "invalid_clusters": int(stats["invalid_clusters"][row]),
                            "vehicles_used": int(stats["vehicles_used"][row]),
                            "max_load_ratio": float(stats["max_load_ratio"][row]),
                            "cluster_sizes": [int(x) for x in stats["sizes"][row]],
                        }
                    )
            else:
                samples = [
                    decode_assignments(
                        logits, demand_norm, valid, args.capacity_mask_mode, True
                    )
                    for _ in range(args.num_samples)
                ]
                stack = torch.stack(samples, dim=0)  # [S,B,N]
                stats = batch_cluster_stats(
                    stack.reshape(-1, stack.shape[-1]).detach().cpu().numpy(),
                    np.tile(demand_raw, (args.num_samples, 1)),
                    np.tile(capacity_raw, args.num_samples),
                    num_vehicles,
                )
                stats = {
                    key: (
                        value.reshape(args.num_samples, len(chunk), num_vehicles)
                        if value.ndim == 2
                        else value.reshape(args.num_samples, len(chunk))
                    )
                    for key, value in stats.items()
                }
                for row, index in enumerate(chunk):
                    overloads = stats["overload_total"][:, row]
                    flags = overloads <= 1e-6
                    best = int(np.argmin(overloads))
                    records.append(
                        {
                            "index": int(index),
                            "structurally_infeasible": bool(not structurally_ok[row]),
                            "feasible": bool(flags.any()),
                            "feasible_samples": int(flags.sum()),
                            "num_samples": int(args.num_samples),
                            "overload_total": float(overloads[best]),
                            "invalid_clusters": int(stats["invalid_clusters"][best, row]),
                            "vehicles_used": int(stats["vehicles_used"][best, row]),
                            "max_load_ratio": float(stats["max_load_ratio"][best, row]),
                            "cluster_sizes": [int(x) for x in stats["sizes"][best, row]],
                        }
                    )

            if args.progress_interval and (
                (start + len(chunk)) % args.progress_interval < len(chunk)
                or start + len(chunk) >= len(indices)
            ):
                done = start + len(chunk)
                ok = sum(1 for r in records if r["feasible"])
                print(
                    f"  {done}/{len(indices)} 成功 {ok} "
                    f"({100 * ok / max(done, 1):.1f}%)",
                    flush=True,
                )
    return records, time.perf_counter() - started


def summarize(records, num_vehicles, args):
    n = len(records)
    feasible = np.array([r["feasible"] for r in records], dtype=bool)
    valid = np.array(
        [not r.get("structurally_infeasible", False) for r in records], dtype=bool
    )
    overload = np.array([r["overload_total"] for r in records], dtype=float)
    invalid = np.array([r["invalid_clusters"] for r in records], dtype=float)
    used = np.array([r["vehicles_used"] for r in records], dtype=float)
    ratio = np.array([r["max_load_ratio"] for r in records], dtype=float)
    sizes = (
        np.concatenate([np.asarray(r["cluster_sizes"]) for r in records])
        if records
        else np.array([], dtype=float)
    )
    sizes = sizes[sizes > 0]
    failed = feasible == False  # noqa: E712 - 明确表达“失败”的含义
    summary = {
        "num_instances": n,
        "num_valid": int(valid.sum()),
        "num_structurally_infeasible": int((~valid).sum()),
        "success": int((feasible & valid).sum()),
        "success_rate": float((feasible & valid).mean() / valid.mean()) if valid.any() else float("nan"),
        "success_rate_all": float(feasible.mean()) if n else float("nan"),
        "num_clusters": int(used.sum()),
        "invalid_clusters": int(invalid.sum()),
        "invalid_cluster_rate": float(invalid.sum() / used.sum()) if used.sum() else 0.0,
        "vehicles_used_mean": float(used.mean()) if n else float("nan"),
        "vehicles_total": int(num_vehicles),
        "max_load_ratio_mean": float(ratio.mean()) if n else float("nan"),
        "max_load_ratio_max": float(ratio.max()) if n else float("nan"),
        "overload_failed_mean": float(overload[failed].mean()) if failed.any() else 0.0,
        "overload_all_mean": float(overload.mean()) if n else 0.0,
        "cluster_size_min": int(sizes.min()) if sizes.size else 0,
        "cluster_size_mean": float(sizes.mean()) if sizes.size else 0.0,
        "cluster_size_max": int(sizes.max()) if sizes.size else 0,
    }
    if args.decode == "sample":
        counts = np.array([r.get("feasible_samples", 0) for r in records], dtype=float)
        summary["sample_success_rate_mean"] = (
            float((counts / args.num_samples).mean()) if n else float("nan")
        )
    return summary


def print_summary(num_vehicles, summary, args, seconds):
    line = "=" * 62
    print(line)
    print(
        f"num_vehicles = {num_vehicles}   decode = {args.decode}"
        + (f" x{args.num_samples}" if args.decode == "sample" else "")
    )
    print(line)
    print(f"实例数                : {summary['num_instances']}")
    if summary["num_structurally_infeasible"]:
        print(
            f"按构造不可行(已剔除)  : {summary['num_structurally_infeasible']}"
        )
    if summary["num_valid"] == 0:
        print("分簇成功率            : 不适用（该车队规模下所有实例的总需求都超容量）")
    else:
        print(
            f"分簇成功率            : {summary['success_rate'] * 100:.2f}%  "
            f"({summary['success']}/{summary['num_valid']})"
        )
    if args.decode == "sample":
        print(
            f"单次采样平均成功率    : "
            f"{summary['sample_success_rate_mean'] * 100:.2f}%"
        )
    print(
        f"不可行簇比例          : {summary['invalid_cluster_rate'] * 100:.3f}%  "
        f"({summary['invalid_clusters']}/{summary['num_clusters']})"
    )
    print(
        f"平均使用车辆          : {summary['vehicles_used_mean']:.2f} / "
        f"{summary['vehicles_total']}"
    )
    print(
        f"最大装载率 mean / max : {summary['max_load_ratio_mean']:.3f} / "
        f"{summary['max_load_ratio_max']:.3f}"
    )
    print(f"失败实例平均超载      : {summary['overload_failed_mean']:.3f}")
    print(
        f"簇大小 min/mean/max   : {summary['cluster_size_min']} / "
        f"{summary['cluster_size_mean']:.2f} / {summary['cluster_size_max']}"
    )
    print(
        f"总耗时 / 每实例       : {seconds:.2f}s / "
        f"{1000 * seconds / max(summary['num_instances'], 1):.2f} ms"
    )


def load_partition_weights(net, checkpoint):
    """从 Lightning checkpoint 里取出分簇网络权重，兼容各种键名前缀。

    支持的写法：
      ``partition_network.node_encoder...``
      ``policy.partition_network.node_encoder...``
      ``model.policy.partition_network.node_encoder...``
      ``litmodule.policy.partition_network.node_encoder...``
    最后兜底用「以本网络参数名结尾」做后缀匹配。
    """
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict) or not state:
        raise SystemExit(f"{checkpoint} 里没有可用的 state_dict")

    own = net.state_dict()
    prefixes = ("policy.", "model.", "litmodule.", "module.", "_orig_mod.")
    matched = {}
    for key, value in state.items():
        if key in own:
            matched[key] = value
            continue
        stripped = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
                    changed = True
        if stripped in own:
            matched[stripped] = value
            continue
        for own_key in own:
            if key.endswith("." + own_key):
                matched[own_key] = value
                break

    if not matched:
        preview = list(state)[:4]
        raise SystemExit(
            f"{checkpoint} 中找不到分簇网络权重。\n"
            f"  checkpoint 里的 key 示例: {preview}\n"
            f"  本脚本期望的 key 示例: {list(own)[:4]}\n"
            f"  如果两边结构完全不同，请确认这是 Stage2 的 checkpoint。"
        )
    missing, _unexpected = net.load_state_dict(matched, strict=False)
    return matched, missing


def build_parser():
    parser = argparse.ArgumentParser(description="Stage2 分簇成功率测试（不做路径规划）")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-instances", type=int, default=1000)
    parser.add_argument(
        "--num-vehicles",
        type=int,
        nargs="+",
        default=None,
        help="可给多个值，例如 --num-vehicles 8 10 12 16；默认用数据里的 num_vehicles",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode", default="greedy", choices=("greedy", "sample"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--capacity-mask-mode", default="static", choices=("static", "none")
    )
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--progress-interval", type=int, default=200)
    parser.add_argument("--out", default=None, help="结果 JSON 路径")
    parser.add_argument("--csv", default=None, help="每实例明细 CSV 路径")
    return parser


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_cvrp_npz(args.data_file)
    total = int(data["locs"].shape[0])
    indices = list(
        range(total if args.num_instances <= 0 else min(args.num_instances, total))
    )

    if args.num_vehicles:
        vehicle_options = [int(value) for value in args.num_vehicles]
    elif data["num_vehicles"] is not None:
        vehicle_options = sorted({int(value) for value in data["num_vehicles"]})
    else:
        raise SystemExit("数据里没有 num_vehicles，请用 --num-vehicles 指定车队规模")

    device = torch.device(args.device)
    net = PartitionNetwork(
        embed_dim=args.embed_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        temperature=args.temperature,
    ).to(device)

    matched, missing = load_partition_weights(net, args.checkpoint)
    net.eval()
    print(
        f"[eval-clustering] 数据={args.data_file} 实例={len(indices)}/{total} "
        f"客户数={data['locs'].shape[1]} device={device} "
        f"载入参数={len(matched)} (missing={len(missing)})",
        flush=True,
    )

    results = {}
    for num_vehicles in vehicle_options:
        records, seconds = evaluate(net, data, indices, num_vehicles, args, device)
        summary = summarize(records, num_vehicles, args)
        print_summary(num_vehicles, summary, args, seconds)
        results[num_vehicles] = {"summary": summary, "instances": records}

    if args.out:
        payload = {
            "data_file": args.data_file,
            "checkpoint": args.checkpoint,
            "num_instances": len(indices),
            "decode": args.decode,
            "num_samples": args.num_samples if args.decode == "sample" else None,
            "results": {str(key): value for key, value in results.items()},
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果已写入 {out_path}")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "num_vehicles",
                    "index",
                    "feasible",
                    "structurally_infeasible",
                    "vehicles_used",
                    "overload_total",
                    "invalid_clusters",
                    "max_load_ratio",
                ]
            )
            for num_vehicles, payload in results.items():
                for record in payload["instances"]:
                    writer.writerow(
                        [
                            num_vehicles,
                            record["index"],
                            int(record["feasible"]),
                            int(record.get("structurally_infeasible", False)),
                            record["vehicles_used"],
                            f"{record['overload_total']:.4f}",
                            record["invalid_clusters"],
                            f"{record['max_load_ratio']:.4f}",
                        ]
                    )
        print(f"明细已写入 {csv_path}")


if __name__ == "__main__":
    main()
