"""Stage2 CVRP 验证脚本：传统求解器基线 +（可选）已训练分簇模型。

最小用法（只看 LKH-3 / OR-Tools 在 CVRP100-12 上的表现）：

    python -m scripts.stage2.eval_cvrp \
        --data-file data/cvrp100/test.npz \
        --num-instances 100 \
        --solvers lkh ortools \
        --objectives minsum minmax \
        --solver-time-limit 5 \
        --lkh-path assets/solver/LKH-3.0.14/LKH \
        --output assets/results/cvrp100_baseline.json

加入已训练的 Stage2 分簇模型一起对比：

    python -m scripts.stage2.eval_cvrp \
        --data-file data/cvrp100/test.npz \
        --num-instances 100 \
        --solvers lkh \
        --objectives minsum minmax \
        --stage2-checkpoint logs/stage2/.../checkpoints/last.ckpt \
        --route-model AttentionModel \
        --route-checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt \
        --model-cluster-solver lkh \
        --output assets/results/cvrp100_model_vs_lkh.json

分簇质量诊断（模型 vs 经典分簇启发式 + LKH 重算路径代价）：

    python -m scripts.stage2.eval_cvrp \
        --data-file data/cvrp100/test.npz \
        --stage2-checkpoint logs/stage2/.../checkpoints/last.ckpt \
        --route-checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt \
        --partition-baselines random loadbalance sweep kmeans \
        --model-cluster-solver lkh \
        --output assets/results/cvrp100_partition_diag.json

`--partition-baselines` 里的 ``sweep``（按极角扫描）和 ``kmeans``（空间聚类）是
判断“网络是否学到东西”的基准线：如果模型的最长路径不比它们好，问题在分簇而不在
二阶求解器。

指标口径（全部在**原始坐标**上重新计算，与 Stage2 的奖励一致）：

- ``max_route``：最长单条路径长度（Stage2 的 min-max 目标）；
- ``total``：全部路径总长度；
- ``overflow``：超出单车容量的装载量之和（模型解可能出现）；
- ``seconds``：单个实例的求解耗时。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.solvers.cvrp import (  # noqa: E402
    CVRPSolution,
    evaluate_routes,
    solve_cvrp_lkh,
    solve_cvrp_lkh_minmax,
    solve_cvrp_ortools,
)
from utils.solvers.tsp import solve_tsp_lkh  # noqa: E402


# --------------------------------------------------------------------------
# 数据读取
# --------------------------------------------------------------------------
def load_cvrp_npz(path: str | Path) -> dict[str, np.ndarray]:
    """读取 Stage2 生成的 CVRP npz（locs 为客户，depot 单独存放）。"""
    with np.load(path) as data:
        files = set(data.files)
        if "locs" not in files or "depot" not in files:
            raise ValueError(f"{path} 缺少 locs/depot 字段，当前字段: {sorted(files)}")
        locs = np.asarray(data["locs"], dtype=np.float64)
        depot = np.asarray(data["depot"], dtype=np.float64)
        demand = np.asarray(data["demand"], dtype=np.float64)
        if "capacity" in files:
            capacity = np.asarray(data["capacity"], dtype=np.float64).reshape(-1)
        else:
            capacity = np.ones(locs.shape[0], dtype=np.float64)
        num_vehicles = (
            np.asarray(data["num_vehicles"], dtype=np.int64).reshape(-1)
            if "num_vehicles" in files
            else None
        )

    if demand.ndim == 3:
        demand = demand.squeeze(-1)
    if demand.shape[-1] == locs.shape[1] + 1:
        # 少数文件会把仓库需求一起存进来
        demand = demand[..., 1:]
    if depot.ndim == 2:
        depot = depot[:, None, :]
    return {
        "locs": locs,
        "depot": depot,
        "demand": demand,
        "capacity": capacity,
        "num_vehicles": num_vehicles,
    }


def build_instance(
    data: dict[str, np.ndarray], index: int, num_vehicles: int | None
) -> tuple[np.ndarray, np.ndarray, float, int]:
    depot = data["depot"][index].reshape(2)
    customers = data["locs"][index]
    locs = np.vstack([depot[None, :], customers])
    demand = data["demand"][index].reshape(-1)
    capacity = float(data["capacity"][index])
    if num_vehicles is not None:
        vehicles = int(num_vehicles)
    elif data["num_vehicles"] is not None:
        vehicles = int(data["num_vehicles"][index])
    else:
        raise SystemExit("数据中没有 num_vehicles，请用 --num-vehicles 指定车队规模")
    return locs, demand, capacity, vehicles


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------
def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _metric(record: dict[str, Any], names: tuple[str, ...]) -> float:
    """按优先级取第一个存在的指标（lkh 精确值优先于 AM 估计值）。"""
    for name in names:
        if name in record:
            return float(record[name])
    return float("nan")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "success"]
    if not ok:
        return {"num_instances": len(records), "num_success": 0}
    max_route = [_metric(r, ("lkh_max_route", "max_route", "am_max_route")) for r in ok]
    total = [_metric(r, ("lkh_total", "total", "am_total")) for r in ok]
    seconds = [float(r["seconds"]) for r in ok]
    overflow = [float(r.get("overflow", 0.0)) for r in ok]
    return {
        "num_instances": len(records),
        "num_success": len(ok),
        "max_route_mean": float(statistics.fmean(max_route)),
        "max_route_median": float(statistics.median(max_route)),
        "max_route_p95": _percentile(max_route, 95),
        "max_route_min": float(min(max_route)),
        "max_route_max": float(max(max_route)),
        "total_mean": float(statistics.fmean(total)),
        "total_median": float(statistics.median(total)),
        "seconds_mean": float(statistics.fmean(seconds)),
        "seconds_total": float(sum(seconds)),
        "instances_per_second": (
            float(len(ok) / sum(seconds)) if sum(seconds) > 0 else float("inf")
        ),
        "overflow_mean": float(statistics.fmean(overflow)),
        "overflow_instances": int(sum(1 for value in overflow if value > 1e-9)),
    }


# --------------------------------------------------------------------------
# 传统求解器
# --------------------------------------------------------------------------
def evaluate_classical(
    data: dict[str, np.ndarray],
    *,
    indices: list[int],
    solvers: list[str],
    objectives: list[str],
    num_vehicles: int | None,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    if "ortools" in solvers:
        try:
            import ortools  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "请求了 OR-Tools 基线，但当前环境没有安装 ortools；"
                "请先 `pip install ortools`，或改用 `--solvers lkh`。"
            ) from exc

    methods: dict[str, dict[str, Any]] = {}
    for solver in solvers:
        for objective in objectives:
            name = f"{solver}-{objective}"
            records: list[dict[str, Any]] = []
            methods[name] = {"solver": solver, "objective": objective, "instances": records}
            for count, index in enumerate(indices, start=1):
                locs, demand, capacity, vehicles = build_instance(
                    data, index, num_vehicles
                )
                if demand.sum() > capacity * vehicles + 1e-9:
                    records.append(
                        {
                            "index": index,
                            "status": "infeasible_instance",
                            "error": "总需求超过车队容量",
                        }
                    )
                    continue
                started = time.perf_counter()
                try:
                    if solver == "lkh":
                        if objective == "minsum":
                            solution = solve_cvrp_lkh(
                                locs,
                                demand,
                                capacity,
                                vehicles,
                                lkh_path=args.lkh_path,
                                time_limit=args.solver_time_limit,
                                runs=args.lkh_runs,
                                seed=args.seed + index,
                            )
                        else:
                            solution = solve_cvrp_lkh_minmax(
                                locs,
                                demand,
                                capacity,
                                vehicles,
                                lkh_path=args.lkh_path,
                                time_limit=args.solver_time_limit,
                                runs=args.lkh_runs,
                                seed=args.seed + index,
                                iterations=args.minmax_iterations,
                            )
                    elif solver == "ortools":
                        solution = solve_cvrp_ortools(
                            locs,
                            demand,
                            capacity,
                            vehicles,
                            objective=objective,
                            time_limit=args.solver_time_limit,
                            seed=args.seed + index,
                        )
                    else:
                        raise ValueError(f"未知求解器 {solver}")
                except Exception as exc:  # noqa: BLE001
                    solution = CVRPSolution(
                        solver=solver,
                        objective=objective,
                        routes=[],
                        route_costs=[],
                        route_demands=[],
                        total_cost=float("nan"),
                        max_route_cost=float("nan"),
                        mean_route_cost=float("nan"),
                        num_routes=0,
                        capacity_violation=float("nan"),
                        seconds=time.perf_counter() - started,
                        status="failed",
                        error=str(exc),
                    )
                records.append(_solution_record(index, solution, vehicles))
                if count % args.progress_interval == 0 or count == len(indices):
                    print(
                        f"[{name}] {count}/{len(indices)} "
                        f"max_route={records[-1].get('max_route', float('nan')):.4f} "
                        f"total={records[-1].get('total', float('nan')):.4f} "
                        f"sec={records[-1].get('seconds', float('nan')):.2f}",
                        flush=True,
                    )
                    if args.output:
                        write_output(
                            args.output, data, indices, methods, args, partial=True
                        )
    return methods


def _solution_record(
    index: int, solution: CVRPSolution, vehicles: int
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "index": int(index),
        "status": solution.status,
        "seconds": float(solution.seconds),
        "num_routes": int(solution.num_routes),
        "num_vehicles": int(vehicles),
    }
    if solution.status == "success":
        record.update(
            {
                "max_route": float(solution.max_route_cost),
                "total": float(solution.total_cost),
                "mean_route": float(solution.mean_route_cost),
                "route_costs": [float(x) for x in solution.route_costs],
                "route_loads": [float(x) for x in solution.route_demands],
                "overflow": float(solution.capacity_violation),
            }
        )
    else:
        record["error"] = solution.error
    return record


def _resolve_vehicles_label(
    args: argparse.Namespace, data: dict[str, np.ndarray]
) -> str:
    if args.num_vehicles is not None:
        return str(args.num_vehicles)
    if data["num_vehicles"] is not None:
        values = np.asarray(data["num_vehicles"]).reshape(-1)
        unique = sorted({int(v) for v in values})
        return str(unique[0]) if len(unique) == 1 else f"{unique[0]}..{unique[-1]}"
    return "?"


# --------------------------------------------------------------------------
# Stage2 模型 + 对照分区
# --------------------------------------------------------------------------
def baseline_assignments(
    name: str,
    locs: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    vehicles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """经典分簇启发式，返回每个客户（0..n-1）的车辆编号。

    - ``random``：均匀随机分配；
    - ``loadbalance``：需求降序 + 装到当前最空的车上（LPT，只保证载重均衡）；
    - ``sweep``：按绕仓库的极角排序后扫描切分（经典 CVRP sweep 起点）；
    - ``kmeans``：对客户坐标做 k-means，得到空间紧凑的簇。
    """
    points = locs[1:]
    num_nodes = len(points)
    if name == "random":
        return rng.integers(0, vehicles, size=num_nodes)

    if name == "loadbalance":
        loads = np.zeros(vehicles)
        assignment = np.zeros(num_nodes, dtype=int)
        for node in np.argsort(-demand):
            vehicle = int(np.argmin(loads))
            assignment[node] = vehicle
            loads[vehicle] += demand[node]
        return assignment

    if name == "sweep":
        angle = np.arctan2(
            points[:, 1] - locs[0, 1], points[:, 0] - locs[0, 0]
        )
        order = np.argsort(np.mod(angle, 2 * np.pi))
        suffix = np.concatenate([demand[order][::-1].cumsum()[::-1], [0.0]])
        target = float(demand.sum()) / max(vehicles, 1)
        assignment = np.zeros(num_nodes, dtype=int)
        vehicle = 0
        load = 0.0
        for position, node in enumerate(order):
            remaining_vehicle = vehicles - vehicle - 1
            remaining_demand = suffix[position + 1]
            must_open = load + demand[node] > capacity + 1e-9
            can_cut = (
                remaining_vehicle > 0
                and load >= target
                and np.ceil(remaining_demand / capacity) <= remaining_vehicle
            )
            if remaining_vehicle > 0 and (must_open or can_cut):
                vehicle += 1
                load = 0.0
            assignment[node] = vehicle
            load += demand[node]
        return assignment

    if name == "kmeans":
        centers = [points[rng.integers(num_nodes)]]
        for _ in range(1, vehicles):
            distance = np.min(
                [
                    np.sum((points - center) ** 2, axis=1)
                    for center in centers
                ],
                axis=0,
            )
            probability = (
                distance / distance.sum()
                if distance.sum() > 0
                else np.full(num_nodes, 1.0 / num_nodes)
            )
            centers.append(points[rng.choice(num_nodes, p=probability)])
        centers = np.asarray(centers)
        labels = np.zeros(num_nodes, dtype=int)
        for _ in range(64):
            distance = np.linalg.norm(
                points[:, None, :] - centers[None, :, :], axis=-1
            )
            labels = distance.argmin(axis=1)
            updated = np.asarray(
                [
                    points[labels == index].mean(axis=0)
                    if np.any(labels == index)
                    else centers[index]
                    for index in range(len(centers))
                ]
            )
            if np.allclose(updated, centers):
                break
            centers = updated
        return labels

    raise ValueError(f"未知对照分区 {name!r}")


def evaluate_partitions(
    data: dict[str, np.ndarray],
    *,
    indices: list[int],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """在同一批实例上评估 Stage2 模型与各类对照分区的分簇质量。"""
    import torch
    from tensordict import TensorDict

    from nets.partition import PartitionNetwork
    from scripts.stage2.policy import Stage2Policy
    from scripts.stage2.route_evaluator import RouteEvaluator
    from utils.config.stage2 import Config

    device = torch.device(args.device)
    num_loc = int(data["locs"].shape[1])
    config = Config(
        num_loc=num_loc,
        num_vehicles=args.num_vehicles,
        objective=args.model_objective,
        penalty=args.penalty,
        capacity_mask_mode=args.capacity_mask_mode,
        embed_dim=args.embed_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        temperature=args.temperature,
        route_model=args.route_model,
        route_checkpoint=args.route_checkpoint,
        route_embed_dim=args.route_embed_dim,
        route_num_encoder_layers=args.route_num_encoder_layers,
        route_num_heads=args.route_num_heads,
        route_normalization=args.route_normalization,
        route_pad_to=args.route_pad_to,
        route_num_starts=args.route_num_starts,
        route_eval_batch_size=args.route_eval_batch_size,
    )
    if not args.route_checkpoint:
        raise SystemExit(
            "评估分簇质量需要冻结的 Stage1 路由模型，请提供 --route-checkpoint"
        )
    route_evaluator = RouteEvaluator(config)

    policy = None
    if args.stage2_checkpoint:
        policy = Stage2Policy(
            partition_network=PartitionNetwork(
                embed_dim=config.embed_dim,
                num_encoder_layers=config.num_encoder_layers,
                num_heads=config.num_heads,
                temperature=config.temperature,
            ),
            route_evaluator=route_evaluator,
            num_vehicles=config.num_vehicles,
            objective=config.objective,
            penalty=config.penalty,
            capacity_mask_mode=config.capacity_mask_mode,
        ).to(device)
        raw = torch.load(
            args.stage2_checkpoint, map_location="cpu", weights_only=False
        )
        state = raw.get("state_dict", raw)
        state = {
            key[len("policy.") :] if key.startswith("policy.") else key: value
            for key, value in state.items()
        }
        policy_keys = set(policy.state_dict())
        matched = {key: value for key, value in state.items() if key in policy_keys}
        if not matched:
            raise SystemExit(
                f"{args.stage2_checkpoint} 中没有可用的分簇网络权重，请检查路径"
            )
        missing, unexpected = policy.load_state_dict(matched, strict=False)
        print(
            f"[stage2] 载入 {len(matched)} 个参数；missing={len(missing)} "
            f"unexpected={len(unexpected)}",
            flush=True,
        )
        policy.eval()

    sources = (["stage2-model"] if policy is not None else []) + [
        f"baseline-{name}" for name in args.partition_baselines
    ]
    methods: dict[str, dict[str, Any]] = {
        name: {"solver": name, "objective": args.model_objective, "instances": []}
        for name in sources
    }
    rng = np.random.default_rng(args.seed)

    for count, index in enumerate(indices, start=1):
        locs, demand, capacity, vehicles = build_instance(data, index, args.num_vehicles)
        if demand.sum() > capacity * vehicles + 1e-9:
            for name in sources:
                methods[name]["instances"].append(
                    {
                        "index": index,
                        "status": "infeasible_instance",
                        "error": "总需求超过车队容量",
                    }
                )
            continue
        td = TensorDict(
            {
                "locs": torch.as_tensor(locs[None], dtype=torch.float32, device=device),
                "demand": torch.as_tensor(
                    (demand / capacity)[None], dtype=torch.float32, device=device
                ),
                "vehicle_capacity": torch.ones(
                    (1, 1), dtype=torch.float32, device=device
                ),
                "num_vehicles": torch.tensor(
                    [[vehicles]], dtype=torch.long, device=device
                ),
            },
            batch_size=[1],
            device=device,
        )
        for name in sources:
            started = time.perf_counter()
            if name == "stage2-model":
                with torch.no_grad():
                    out = policy(td, phase="test")
                assignments = out["assignments"][0].detach().cpu().numpy()
            else:
                with torch.no_grad():
                    assignments = baseline_assignments(
                        name[len("baseline-") :],
                        locs,
                        demand,
                        capacity,
                        vehicles,
                        rng,
                    )
            with torch.no_grad():
                assign_tensor = torch.as_tensor(
                    assignments[None], dtype=torch.long, device=device
                )
                costs_t, overflow_t, valid_t = route_evaluator.evaluate(
                    td, assign_tensor, vehicle_counts=vehicles
                )
                costs_t = costs_t[0].detach().cpu().numpy()
                overflow_t = overflow_t[0].detach().cpu().numpy()
                valid_t = valid_t[0].detach().cpu().numpy().astype(bool)
            penalized = costs_t + args.penalty * overflow_t
            objective = float(penalized[valid_t].max()) if valid_t.any() else 0.0
            seconds = time.perf_counter() - started

            clusters: list[list[int]] = []
            vehicle_ids: list[int] = []
            for vehicle in range(vehicles):
                members = [
                    node + 1
                    for node, value in enumerate(assignments.tolist())
                    if value == vehicle
                ]
                if members:
                    clusters.append(members)
                    vehicle_ids.append(vehicle)

            record: dict[str, Any] = {
                "index": int(index),
                "status": "success",
                "seconds": seconds,
                "num_routes": len(clusters),
                "num_vehicles": int(vehicles),
                "objective": objective,
                "cluster_sizes": [len(cluster) for cluster in clusters],
                # 注意：分簇结果没有“路径顺序”，因此这里给出的是**簇内 TSP 代价**，
                # 由冻结 Stage1 模型估计（与训练口径一致），不是客户下标的顺序长度。
                "am_route_costs": [
                    float(costs_t[vehicle]) for vehicle in vehicle_ids
                ],
                "am_max_route": (
                    float(max(costs_t[vehicle] for vehicle in vehicle_ids))
                    if vehicle_ids
                    else 0.0
                ),
                "am_total": float(sum(costs_t[vehicle] for vehicle in vehicle_ids)),
            }
            _costs, loads, violation = evaluate_routes(locs, demand, clusters, capacity)
            record["route_loads"] = loads
            # overflow 统一用原始需求单位，和传统求解器记录保持一致
            record["overflow"] = violation
            if args.model_cluster_solver == "lkh":
                lkh_costs: list[float] = []
                for cluster in clusters:
                    cluster_locs = np.vstack([locs[0][None, :], locs[cluster]])
                    tsp = solve_tsp_lkh(
                        cluster_locs,
                        lkh_path=args.lkh_path,
                        time_limit=args.cluster_time_limit,
                        runs=1,
                        seed=args.seed + index,
                    )
                    lkh_costs.append(float(tsp.cost))
                record.update(
                    {
                        "lkh_route_costs": lkh_costs,
                        "lkh_max_route": float(max(lkh_costs)) if lkh_costs else 0.0,
                        "lkh_total": float(sum(lkh_costs)),
                    }
                )
            methods[name]["instances"].append(record)

        if count % args.progress_interval == 0 or count == len(indices):
            for name in sources:
                record = methods[name]["instances"][-1]
                print(
                    f"[{name}] {count}/{len(indices)} "
                    f"max_route={_metric(record, ('lkh_max_route', 'max_route', 'am_max_route')):.4f} "
                    f"total={_metric(record, ('lkh_total', 'total', 'am_total')):.4f} "
                    f"overflow={record['overflow']:.3f} "
                    f"sec={record['seconds']:.2f}",
                    flush=True,
                )
            if args.output:
                write_output(args.output, data, indices, methods, args, partial=True)

    return methods


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def write_output(
    path: str | Path,
    data: dict[str, np.ndarray],
    indices: list[int],
    methods: dict[str, Any],
    args: argparse.Namespace,
    *,
    partial: bool,
) -> None:
    payload: dict[str, Any] = {
        "data_file": str(args.data_file),
        "num_instances": len(indices),
        "num_loc": int(data["locs"].shape[1]),
        "capacity": float(data["capacity"][0]) if len(data["capacity"]) else None,
        "num_vehicles": args.num_vehicles,
        "lkh_path": args.lkh_path,
        "solver_time_limit": args.solver_time_limit,
        "partial": partial,
        "methods": {},
    }
    for name, method in methods.items():
        if "instances" in method:
            payload["methods"][name] = {
                "solver": method.get("solver"),
                "objective": method.get("objective"),
                "summary": summarize(method["instances"]),
                "instances": method["instances"],
            }
        elif "records" in method:
            records = method["records"]
            summary = summarize(records)
            if records and all("lkh_max_route" in r for r in records if r["status"] == "success"):
                lkh_max = [r["lkh_max_route"] for r in records if r["status"] == "success"]
                summary["lkh_max_route_mean"] = float(statistics.fmean(lkh_max))
                summary["lkh_total_mean"] = float(
                    statistics.fmean(
                        [r["lkh_total"] for r in records if r["status"] == "success"]
                    )
                )
            payload["methods"][name] = {
                "solver": "stage2-model",
                "objective": args.model_objective,
                "summary": summary,
                "instances": records,
            }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage2 CVRP 验证：传统求解器基线 + Stage2 分簇模型"
    )
    parser.add_argument("--data-file", required=True, help="CVRP npz 数据文件")
    parser.add_argument(
        "--num-instances",
        type=int,
        default=100,
        help="评估前 N 个实例（<=0 表示全部）",
    )
    parser.add_argument(
        "--num-vehicles",
        type=int,
        default=None,
        help="覆盖车队规模；默认使用数据中的 num_vehicles",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["lkh"],
        choices=("lkh", "ortools", "none"),
        help="none 表示只评估 --stage2-checkpoint 的模型",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["minsum", "minmax"],
        choices=("minsum", "minmax"),
        help="minsum=总长度最短；minmax=最长路径最短（Stage2 口径）",
    )
    parser.add_argument("--solver-time-limit", type=float, default=None)
    parser.add_argument("--cluster-time-limit", type=float, default=None)
    parser.add_argument(
        "--lkh-runs", type=int, default=1, help="每个实例的 LKH 重复次数"
    )
    parser.add_argument(
        "--minmax-iterations",
        type=int,
        default=8,
        help="min-max 二分搜索的 LKH 调用次数上限",
    )
    parser.add_argument("--lkh-path", default="assets/solver/LKH-3.0.14/LKH")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("-o", "--output", default=None)

    # Stage2 模型
    parser.add_argument("--stage2-checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-objective", default="minmax", choices=("sum", "minmax"))
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument(
        "--capacity-mask-mode", default="static", choices=("none", "static")
    )
    parser.add_argument(
        "--model-cluster-solver",
        default="model",
        choices=("model", "lkh"),
        help="模型分簇的路径代价：model=冻结的 Stage1 模型，lkh=用 LKH 重算",
    )
    parser.add_argument(
        "--partition-baselines",
        nargs="+",
        default=[],
        choices=("random", "loadbalance", "sweep", "kmeans"),
        help="对照分区：random / loadbalance / sweep / kmeans",
    )
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--route-model",
        default="AttentionModel",
        choices=("AttentionModel", "POMO", "MDAM", "PointerNetwork"),
    )
    parser.add_argument("--route-checkpoint", default=None)
    parser.add_argument("--route-embed-dim", type=int, default=128)
    parser.add_argument("--route-num-encoder-layers", type=int, default=3)
    parser.add_argument("--route-num-heads", type=int, default=8)
    parser.add_argument(
        "--route-normalization", default="batch", choices=("batch", "instance")
    )
    parser.add_argument("--route-pad-to", type=int, default=0)
    parser.add_argument("--route-num-starts", type=int, default=0)
    parser.add_argument("--route-eval-batch-size", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_cvrp_npz(args.data_file)
    total = int(data["locs"].shape[0])
    if args.num_instances <= 0 or args.num_instances > total:
        indices = list(range(total))
    else:
        indices = list(range(args.num_instances))
    print(
        f"[eval-cvrp] 数据={args.data_file} 实例数={len(indices)}/{total} "
        f"客户数={data['locs'].shape[1]} "
        f"容量={float(data['capacity'][0]):g} "
        f"车辆数={_resolve_vehicles_label(args, data)}",
        flush=True,
    )

    methods: dict[str, Any] = {}
    solvers = [s for s in args.solvers if s != "none"]
    if solvers:
        methods.update(
            evaluate_classical(
                data,
                indices=indices,
                solvers=solvers,
                objectives=list(args.objectives),
                num_vehicles=args.num_vehicles,
                args=args,
            )
        )
    if args.stage2_checkpoint or args.partition_baselines:
        if not args.route_checkpoint:
            raise SystemExit(
                "评估分簇质量需要冻结的 Stage1 路由模型，请提供 --route-checkpoint"
            )
        methods.update(evaluate_partitions(data, indices=indices, args=args))

    if args.output:
        write_output(args.output, data, indices, methods, args, partial=False)
        print(f"[eval-cvrp] 结果已写入 {args.output}", flush=True)

    for name, method in methods.items():
        records = method.get("instances") or method.get("records") or []
        summary = summarize(records)
        if not summary.get("num_success"):
            print(f"{name}: 无成功实例")
            continue
        print(
            f"{name}: n_ok={summary['num_success']}/{summary['num_instances']} "
            f"max_route_mean={summary['max_route_mean']:.4f} "
            f"total_mean={summary['total_mean']:.4f} "
            f"sec/instance={summary['seconds_mean']:.3f} "
            f"overflow_mean={summary['overflow_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
