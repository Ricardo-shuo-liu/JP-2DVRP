"""Classical CVRP solvers (LKH-3 / OR-Tools) used as evaluation baselines.

输入约定（与 :mod:`utils.solvers.tsp` 保持一致）：

- ``locs``: ``[n, 2]``，**索引 0 是仓库**，1..n-1 是客户；
- ``demand``: ``[n]``，与 ``capacity`` 同一量纲（仓库需求可以填 0）；
- ``capacity``: 单车容量（标量）；
- ``num_vehicles``: 车队规模 K。

两种目标：

- ``objective="minsum"``：最小化全部路径总长度（LKH-3 原生 CVRP 目标）；
- ``objective="minmax"``：最小化**最长**单条路径长度，与 Stage2 的奖励口径一致。

实现说明（重要）：

1. LKH-3 的原生 ``TYPE: CVRP`` 只支持 min-sum 目标；``MTSP_OBJECTIVE=MINMAX``
   对 CVRP 会被忽略（源码 ``MTSP2TSP.c`` 中 CVRP 固定使用 ``Penalty_CVRP``），
   而 ``TYPE: CCVRP`` + ``MINMAX`` 优化的是另一套内部量，并不是本项目的
   min-max 口径。因此 min-max 基线通过**二分路径长度上限**实现：反复用
   ``DISTANCE = λ`` 调用 LKH-3，寻找“最长路径 ≤ λ 仍可行”的最小 λ。
2. 所有路径代价都在调用方这一侧用欧氏距离重新计算，不依赖 LKH 输出的
   ``Cost``（它包含内部变换后的量纲）。
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .tsp import find_lkh_executable

__all__ = [
    "CVRPSolution",
    "evaluate_routes",
    "solve_cvrp_lkh",
    "solve_cvrp_lkh_minmax",
    "solve_cvrp_ortools",
]


@dataclass
class CVRPSolution:
    """一条 CVRP 完整解及可复现的度量信息。"""

    solver: str
    objective: str
    routes: list[list[int]]  # 客户下标（0 表示仓库，不出现）
    route_costs: list[float]
    route_demands: list[float]
    total_cost: float
    max_route_cost: float
    mean_route_cost: float
    num_routes: int
    capacity_violation: float
    seconds: float
    status: str
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def feasible(self) -> bool:
        return self.capacity_violation <= 1e-6 and self.error is None

    def to_dict(self, with_routes: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "solver": self.solver,
            "objective": self.objective,
            "num_routes": self.num_routes,
            "total_cost": self.total_cost,
            "max_route_cost": self.max_route_cost,
            "mean_route_cost": self.mean_route_cost,
            "capacity_violation": self.capacity_violation,
            "seconds": self.seconds,
            "status": self.status,
            "route_costs": list(self.route_costs),
            "meta": dict(self.meta),
        }
        if self.error is not None:
            payload["error"] = self.error
        if with_routes:
            payload["routes"] = [list(route) for route in self.routes]
        return payload


def _as_locs(locs) -> np.ndarray:
    array = np.asarray(locs, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"locs 应为 [n, 2]，当前 {array.shape}")
    return array


def _as_demand(demand, n: int) -> np.ndarray:
    array = np.asarray(demand, dtype=np.float64).reshape(-1)
    if array.shape[0] == n:
        # 允许把仓库需求一起传进来（与 locs 同长时首元素视为仓库需求）
        array = array[1:]
    if array.shape[0] != n - 1:
        raise ValueError(
            f"demand 长度应为客户数 {n - 1}（或含仓库的 {n}），当前 {array.shape[0]}"
        )
    return array


def _distance_matrix(locs: np.ndarray, scale: int) -> np.ndarray:
    delta = locs[:, None, :] - locs[None, :, :]
    matrix = np.rint(np.linalg.norm(delta, axis=-1) * scale).astype(np.int64)
    np.fill_diagonal(matrix, 0)
    return matrix


def _route_cost(locs: np.ndarray, nodes: Sequence[int]) -> float:
    """仓库 → nodes → 仓库 的欧氏闭路长度。"""
    if not nodes:
        return 0.0
    cost = float(np.linalg.norm(locs[0] - locs[nodes[0]]))
    for a, b in zip(nodes, nodes[1:]):
        cost += float(np.linalg.norm(locs[a] - locs[b]))
    cost += float(np.linalg.norm(locs[nodes[-1]] - locs[0]))
    return cost


def evaluate_routes(
    locs,
    demand,
    routes: Iterable[Sequence[int]],
    capacity: float,
) -> tuple[list[float], list[float], float]:
    """重新计算路径长度 / 载重 / 容量超出量（不依赖任何求解器输出）。

    ``routes`` 中的下标是 **locs 的下标**（0 是仓库）；仓库即使出现也会被忽略。
    ``demand`` 可以是客户需求 ``[n-1]``（推荐）或含仓库的 ``[n]``。
    """
    point_set = _as_locs(locs)
    demand_array = _as_demand(demand, len(point_set))
    route_list: list[list[int]] = []
    for route in routes:
        nodes = [int(node) for node in route if int(node) != 0]
        for node in nodes:
            if node < 1 or node >= len(point_set):
                raise ValueError(f"路径包含非法客户下标 {node}")
        route_list.append(nodes)
    costs = [_route_cost(point_set, route) for route in route_list]
    loads = [
        float(sum(demand_array[node - 1] for node in route)) if route else 0.0
        for route in route_list
    ]
    violation = float(
        sum(max(0.0, load - float(capacity)) for load in loads)
    )
    return costs, loads, violation


def _integerize(
    demand: np.ndarray, capacity: float, scale: int
) -> tuple[np.ndarray, int, float]:
    """把需求/容量转成整数，返回 ``(demand_int, capacity_int, 1/放大倍数)``。

    已经满足整数条件的实例原样使用；小数实例（例如 RL4CO 归一化后的
    ``demand/capacity``）放大后分别向上/向下取整，保证不会出现“虚假可行”。
    """
    if np.allclose(demand, np.rint(demand)) and float(capacity).is_integer():
        return (
            np.rint(demand).astype(np.int64),
            max(1, int(round(float(capacity)))),
            1.0,
        )
    demand_int = np.ceil(demand * scale).astype(np.int64)
    capacity_int = max(1, int(math.floor(float(capacity) * scale)))
    return demand_int, capacity_int, 1.0 / scale


def _build_cvrp_problem(
    locs: np.ndarray,
    matrix: np.ndarray,
    demand_int: np.ndarray,
    capacity_int: int,
    *,
    name: str = "stage2_cvrp",
    scale: int = 1_000_000,
) -> str:
    """按 TSPLIB 格式生成 CVRP 实例文本。

    注意：使用 ``EXPLICIT`` 距离矩阵时，LKH 仍要求存在
    ``NODE_COORD_TYPE`` / ``NODE_COORD_SECTION``，否则会返回 -11（段错误）。
    """
    n = len(locs)
    coords = np.rint(locs * scale).astype(np.int64)
    lines = [
        f"NAME: {name}",
        "TYPE: CVRP",
        f"DIMENSION: {n}",
        f"CAPACITY: {capacity_int}",
        "EDGE_WEIGHT_TYPE: EXPLICIT",
        "EDGE_WEIGHT_FORMAT: FULL_MATRIX",
        "NODE_COORD_TYPE: TWOD_COORDS",
        "NODE_COORD_SECTION",
        *[f"{i + 1} {coords[i, 0]} {coords[i, 1]}" for i in range(n)],
        "DEMAND_SECTION",
        *[f"{i + 1} {int(demand_int[i - 1]) if i else 0}" for i in range(n)],
        "DEPOT_SECTION",
        "1",
        "-1",
        "EDGE_WEIGHT_SECTION",
        *[" ".join(map(str, row)) for row in matrix.tolist()],
        "EOF",
        "",
    ]
    return "\n".join(lines)


def _parse_lkh_tour(tour_text: str, n: int) -> list[list[int]]:
    """把 LKH 的 CVRP tour 按仓库标记切成多条路径（返回客户下标）。"""
    if "TOUR_SECTION" not in tour_text:
        return []
    body = tour_text.split("TOUR_SECTION", 1)[1]
    sequence: list[int] = []
    for token in body.split():
        if token in {"-1", "EOF"}:
            break
        if token.lstrip("-").isdigit():
            sequence.append(int(token))
    if not sequence:
        return []

    # 仓库副本的编号 > n；节点 1 是仓库本身。
    markers = [i for i, value in enumerate(sequence) if value == 1 or value > n]
    if not markers:
        return [[value - 1 for value in sequence if 1 < value <= n]]

    routes: list[list[int]] = []
    for index, start in enumerate(markers):
        end = markers[(index + 1) % len(markers)]
        if end > start:
            segment = sequence[start + 1 : end]
        else:  # 环形：绕回序列开头
            segment = sequence[start + 1 :] + sequence[:end]
        customers = [value - 1 for value in segment if 1 < value <= n]
        if customers:
            routes.append(customers)
    return routes


def _run_lkh(
    executable: str,
    problem_text: str,
    *,
    num_vehicles: int,
    time_limit: float | None,
    runs: int,
    seed: int,
    max_route_limit: float | None,
    extra_params: dict[str, Any] | None,
    n: int,
) -> tuple[list[list[int]], float, dict[str, Any], str, str | None]:
    """执行一次 LKH；返回 ``(routes, seconds, meta, status, error)``。

    ``status`` 取值：

    - ``success``：拿到 tour；
    - ``infeasible``：LKH 正常结束但没写出 tour（例如 ``DISTANCE`` 上限太紧）；
    - ``failed``：进程返回码异常或抛出异常。
    """
    started = time.perf_counter()
    workdir = Path(tempfile.mkdtemp(prefix="jp2dvrp_cvrp_"))
    problem_file = workdir / "problem.cvrp"
    tour_file = workdir / "tour.tour"
    parameter_file = workdir / "problem.par"
    problem_file.write_text(problem_text)

    parameter_lines = [
        "SPECIAL",
        f"PROBLEM_FILE = {problem_file}",
        f"TOUR_FILE = {tour_file}",
        f"RUNS = {int(runs)}",
        f"SEED = {int(seed)}",
        f"SALESMEN = {int(num_vehicles)}",
        # LKH 默认 MTSP_MIN_SIZE = 1（每辆车至少 1 个客户），当车队规模接近
        # 下界时会人为制造“不可行”（Penalty=1）；这里允许空路径。
        "MTSP_MIN_SIZE = 0",
    ]
    if time_limit is not None and time_limit > 0:
        parameter_lines.append(f"TIME_LIMIT = {float(time_limit):g}")
    if max_route_limit is not None:
        # DISTANCE = 单车最大路径长度（LKH 内部惩罚项），用于 min-max 二分。
        parameter_lines.append(f"DISTANCE = {float(max_route_limit):g}")
    for key, value in (extra_params or {}).items():
        parameter_lines.append(f"{key} = {value}")
    parameter_file.write_text("\n".join(parameter_lines) + "\n")

    try:
        completed = subprocess.run(
            [executable, str(parameter_file)],
            cwd=str(workdir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - 求解失败也当成数据返回
        elapsed = time.perf_counter() - started
        return [], elapsed, {"returncode": None}, "failed", str(exc)

    elapsed = time.perf_counter() - started
    stdout = completed.stdout or ""
    penalty = re.search(r"Penalty\.min = (\d+)", stdout)
    if completed.returncode != 0:
        return (
            [],
            elapsed,
            {"returncode": completed.returncode},
            "failed",
            stdout.strip()[-1000:],
        )
    if not tour_file.is_file():
        return (
            [],
            elapsed,
            {
                "returncode": completed.returncode,
                "penalty": int(penalty.group(1)) if penalty else None,
            },
            "infeasible",
            None,
        )

    routes = _parse_lkh_tour(tour_file.read_text(), n)
    meta = {
        "returncode": completed.returncode,
        "lkh_log_tail": stdout.strip().splitlines()[-5:],
    }
    return routes, elapsed, meta, "success", None


def solve_cvrp_lkh(
    locs,
    demand,
    capacity: float,
    num_vehicles: int,
    *,
    lkh_path: str | None = None,
    time_limit: float | None = None,
    runs: int = 1,
    seed: int = 1234,
    scale: int = 1_000_000,
    extra_params: dict[str, Any] | None = None,
) -> CVRPSolution:
    """用 LKH-3 求 CVRP 的 min-sum 解（可复现、自算代价）。"""
    point_set = _as_locs(locs)
    n = len(point_set)
    if n < 2:
        raise ValueError("CVRP 至少需要 1 个仓库 + 1 个客户")
    demand_array = _as_demand(demand, n)
    num_vehicles = int(num_vehicles)
    if num_vehicles < 1:
        raise ValueError("num_vehicles 必须 >= 1")

    demand_int, capacity_int, demand_scale = _integerize(
        demand_array, capacity, scale
    )
    matrix = _distance_matrix(point_set, scale)
    problem_text = _build_cvrp_problem(
        point_set, matrix, demand_int, capacity_int, scale=scale
    )
    executable = find_lkh_executable(lkh_path)

    routes, seconds, meta, status, error = _run_lkh(
        executable,
        problem_text,
        num_vehicles=num_vehicles,
        time_limit=time_limit,
        runs=runs,
        seed=seed,
        max_route_limit=None,
        extra_params=extra_params,
        n=n,
    )
    meta.update(
        {
            "time_limit": time_limit,
            "runs": runs,
            "seed": seed,
            "num_vehicles": num_vehicles,
            "executable": executable,
            "demand_scale": demand_scale,
        }
    )
    if status != "success" or not routes:
        return CVRPSolution(
            solver="lkh3",
            objective="minsum",
            routes=[],
            route_costs=[],
            route_demands=[],
            total_cost=float("nan"),
            max_route_cost=float("nan"),
            mean_route_cost=float("nan"),
            num_routes=0,
            capacity_violation=float("nan"),
            seconds=seconds,
            status=status,
            meta=meta,
            error=error
            or ("LKH 没有找到可行解" if status == "infeasible" else "LKH 没有返回 tour"),
        )

    costs, loads, violation = evaluate_routes(point_set, demand_array, routes, capacity)
    return CVRPSolution(
        solver="lkh3",
        objective="minsum",
        routes=routes,
        route_costs=costs,
        route_demands=loads,
        total_cost=float(sum(costs)),
        max_route_cost=float(max(costs)),
        mean_route_cost=float(sum(costs) / len(costs)),
        num_routes=len(costs),
        capacity_violation=violation,
        seconds=seconds,
        status="success",
        meta=meta,
    )


def solve_cvrp_lkh_minmax(
    locs,
    demand,
    capacity: float,
    num_vehicles: int,
    *,
    lkh_path: str | None = None,
    time_limit: float | None = None,
    runs: int = 1,
    seed: int = 1234,
    scale: int = 1_000_000,
    iterations: int = 8,
    tolerance: float = 1e-3,
    upper_bound: float | None = None,
    extra_params: dict[str, Any] | None = None,
) -> CVRPSolution:
    """min-max CVRP 基线：对单车路径长度上限做二分搜索。

    LKH-3 原生不支持 min-max CVRP（见模块 docstring），但它支持
    ``DISTANCE = λ``（每条路径长度上限）。于是：

    1. 先用 min-sum 解拿到一个可行上界 ``hi = 该解的最长路径``；
    2. 下界 ``lo = max_i 2 * dist(depot, i)``（任何服务客户 i 的路径都不短于往返）；
    3. 二分 ``λ``：若 LKH 在 ``DISTANCE = λ`` 下找到最长路径 ≤ λ 的解，
       把 ``hi`` 更新为该解的最长路径；否则 ``lo = λ``。

    返回的是**近似最优的可行解**（启发式），不是可证明最优。
    """
    point_set = _as_locs(locs)
    n = len(point_set)
    demand_array = _as_demand(demand, n)
    num_vehicles = int(num_vehicles)

    demand_int, capacity_int, demand_scale = _integerize(
        demand_array, capacity, scale
    )
    matrix = _distance_matrix(point_set, scale)
    problem_text = _build_cvrp_problem(
        point_set, matrix, demand_int, capacity_int, scale=scale
    )
    executable = find_lkh_executable(lkh_path)

    started = time.perf_counter()
    total_lkh_seconds = 0.0
    history: list[dict[str, Any]] = []

    if upper_bound is not None:
        best_routes: list[list[int]] = []
        best_max = float(upper_bound)
    else:
        base = solve_cvrp_lkh(
            point_set,
            demand_array,
            capacity,
            num_vehicles,
            lkh_path=lkh_path,
            time_limit=time_limit,
            runs=runs,
            seed=seed,
            scale=scale,
            extra_params=extra_params,
        )
        total_lkh_seconds += base.seconds
        if base.error is not None or not base.routes:
            return CVRPSolution(
                solver="lkh3",
                objective="minmax",
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
                meta={"history": history},
                error=base.error or "min-sum 初始解失败，无法开始二分",
            )
        best_routes = base.routes
        best_max = base.max_route_cost

    # 下界：任何路径服务客户 i 至少要往返仓库。
    detour = 2.0 * np.linalg.norm(point_set[1:] - point_set[0], axis=-1)
    lo = float(detour.max()) if len(detour) else 0.0
    hi = float(best_max)

    for _ in range(max(0, int(iterations))):
        if hi - lo <= tolerance * max(1.0, hi):
            break
        mid = 0.5 * (lo + hi)
        # 放大到 LKH 的整数距离尺度后向下取整，避免由于取整放宽约束。
        limit = math.floor(mid * scale)
        routes, seconds, meta, status, error = _run_lkh(
            executable,
            problem_text,
            num_vehicles=num_vehicles,
            time_limit=time_limit,
            runs=runs,
            seed=seed + len(history) + 1,
            max_route_limit=limit,
            extra_params=extra_params,
            n=n,
        )
        total_lkh_seconds += seconds
        record: dict[str, Any] = {
            "limit": mid,
            "seconds": seconds,
            "status": status,
            "error": error,
        }
        if status == "success" and routes:
            costs, _loads, violation = evaluate_routes(
                point_set, demand_array, routes, capacity
            )
            record["max_route_cost"] = max(costs)
            record["capacity_violation"] = violation
            if violation <= 1e-6 and max(costs) <= mid * (1 + 1e-6):
                best_routes = routes
                best_max = max(costs)
                hi = min(hi, best_max)
            else:
                lo = mid
        else:
            # 没有 tour（或进程失败）都视为该 λ 不可行
            lo = mid
        history.append(record)

    costs, loads, violation = evaluate_routes(
        point_set, demand_array, best_routes, capacity
    )
    return CVRPSolution(
        solver="lkh3",
        objective="minmax",
        routes=best_routes,
        route_costs=costs,
        route_demands=loads,
        total_cost=float(sum(costs)),
        max_route_cost=float(max(costs)),
        mean_route_cost=float(sum(costs) / len(costs)),
        num_routes=len(costs),
        capacity_violation=violation,
        seconds=time.perf_counter() - started,
        status="success",
        meta={
            "iterations": len(history),
            "lkh_seconds": total_lkh_seconds,
            "num_vehicles": num_vehicles,
            "lower_bound": lo,
            "upper_bound": hi,
            "history": history,
            "seed": seed,
            "executable": executable,
            "demand_scale": demand_scale,
            "note": "binary search on DISTANCE limit; heuristic, not proven optimal",
        },
    )


def solve_cvrp_ortools(
    locs,
    demand,
    capacity: float,
    num_vehicles: int,
    *,
    objective: str = "minsum",
    time_limit: float = 1.0,
    seed: int = 1234,
    scale: int = 1_000_000,
    span_coefficient: int = 1_000,
    first_solution_strategy: str = "PATH_CHEAPEST_ARC",
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH",
) -> CVRPSolution:
    """用 OR-Tools 求解 CVRP。

    ``objective="minmax"`` 时使用 ``SetGlobalSpanCostCoefficient``，即
    目标变为 ``sum(路径长度) + span_coefficient * max(路径长度)``，
    ``span_coefficient`` 足够大时等价于优先压缩最长路径。
    """
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:  # pragma: no cover - 取决于可选依赖
        raise ImportError("使用 OR-Tools 前请先安装：pip install ortools") from exc

    if objective not in ("minsum", "minmax"):
        raise ValueError(f"未知 objective: {objective}")

    point_set = _as_locs(locs)
    n = len(point_set)
    demand_array = _as_demand(demand, n)
    demand_int, capacity_int, _demand_scale = _integerize(
        demand_array, capacity, scale
    )
    matrix = _distance_matrix(point_set, scale)
    num_vehicles = int(num_vehicles)

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(matrix[from_node, to_node])

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return 0 if from_node == 0 else int(demand_int[from_node - 1])

    transit_index = routing.RegisterTransitCallback(distance_callback)
    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [capacity_int] * num_vehicles,
        True,
        "Capacity",
    )
    if objective == "minmax":
        routing.SetGlobalSpanCostCoefficient(int(span_coefficient))

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, first_solution_strategy
    )
    search_parameters.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, local_search_metaheuristic
    )
    if hasattr(search_parameters.sat_parameters, "random_seed"):
        search_parameters.sat_parameters.random_seed = int(seed)
    elif hasattr(search_parameters, "random_seed"):
        search_parameters.random_seed = int(seed)
    if time_limit is not None and time_limit > 0:
        search_parameters.time_limit.seconds = int(time_limit)
        search_parameters.time_limit.nanos = int(
            (float(time_limit) - int(time_limit)) * 1_000_000_000
        )

    started = time.perf_counter()
    solution = routing.SolveWithParameters(search_parameters)
    seconds = time.perf_counter() - started

    if solution is None:
        return CVRPSolution(
            solver="ortools",
            objective=objective,
            routes=[],
            route_costs=[],
            route_demands=[],
            total_cost=float("nan"),
            max_route_cost=float("nan"),
            mean_route_cost=float("nan"),
            num_routes=0,
            capacity_violation=float("nan"),
            seconds=seconds,
            status="failed",
            meta={"time_limit": time_limit, "seed": seed},
            error="OR-Tools 没有返回可行解",
        )

    routes: list[list[int]] = []
    for vehicle in range(num_vehicles):
        index = routing.Start(vehicle)
        route: list[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                route.append(int(node))
            index = solution.Value(routing.NextVar(index))
        if route:
            routes.append(route)

    costs, loads, violation = evaluate_routes(point_set, demand_array, routes, capacity)
    return CVRPSolution(
        solver="ortools",
        objective=objective,
        routes=routes,
        route_costs=costs,
        route_demands=loads,
        total_cost=float(sum(costs)),
        max_route_cost=float(max(costs)),
        mean_route_cost=float(sum(costs) / len(costs)),
        num_routes=len(costs),
        capacity_violation=violation,
        seconds=seconds,
        status="success",
        meta={
            "time_limit": time_limit,
            "seed": seed,
            "num_vehicles": num_vehicles,
            "span_coefficient": span_coefficient if objective == "minmax" else None,
            "objective_scaled": int(solution.ObjectiveValue()),
        },
    )
