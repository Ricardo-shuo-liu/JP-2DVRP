"""Classical TSP solvers for Stage1 evaluation.

OR-Tools is imported lazily. LKH is invoked directly as an external
executable, so no Python LKH wrapper is required.
"""

from __future__ import annotations

import itertools
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _as_locs(locs) -> np.ndarray:
    array = np.asarray(locs, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"locs 应为 [n, 2]，当前 {array.shape}")
    return array


def _distance_matrix(locs: np.ndarray) -> np.ndarray:
    delta = locs[:, None, :] - locs[None, :, :]
    return np.linalg.norm(delta, axis=-1)


def _canonical_tour(actions: list[int]) -> list[int]:
    """Rotate a cyclic tour so that node 0 is first."""
    if not actions:
        return []
    zero = actions.index(0)
    return actions[zero:] + actions[:zero]


def _tour_cost(locs: np.ndarray, actions: list[int]) -> float:
    if len(actions) != len(locs):
        raise ValueError(f"tour 长度 {len(actions)} != 节点数 {len(locs)}")
    if sorted(actions) != list(range(len(locs))):
        raise ValueError("tour 不是所有节点的排列")
    if len(actions) <= 1:
        return 0.0
    return float(
        sum(
            math.dist(locs[actions[i]], locs[actions[(i + 1) % len(actions)]])
            for i in range(len(actions))
        )
    )


@dataclass
class TSPClassicalResult:
    """One classical TSP solve result."""

    solver: str
    cost: float
    actions: list[int]
    seconds: float
    status: str
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def solve_tsp_ortools(
    locs,
    *,
    time_limit: float = 0.1,
    seed: int = 1234,
    first_solution_strategy: str = "PATH_CHEAPEST_ARC",
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH",
) -> TSPClassicalResult:
    """Solve one Euclidean TSP with OR-Tools RoutingModel.

    OR-Tools routing is a heuristic solver by default; ``status == "success"``
    means a feasible solution was returned, not that optimality was proved.
    """
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "使用 OR-Tools 前请先安装：pip install ortools"
        ) from exc

    point_set = _as_locs(locs)
    n = len(point_set)
    if n == 0:
        return TSPClassicalResult(
            solver="ortools",
            cost=0.0,
            actions=[],
            seconds=0.0,
            status="empty",
        )
    if n == 1:
        return TSPClassicalResult(
            solver="ortools",
            cost=0.0,
            actions=[0],
            seconds=0.0,
            status="trivial",
        )

    scale = 1_000_000
    matrix = np.rint(_distance_matrix(point_set) * scale).astype(np.int64)
    np.fill_diagonal(matrix, 0)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(matrix[from_node, to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    first_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy,
        first_solution_strategy,
    )
    search_parameters.first_solution_strategy = first_strategy
    local_search = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic,
        local_search_metaheuristic,
    )
    search_parameters.local_search_metaheuristic = local_search
    # ``random_seed`` moved under SAT parameters in newer OR-Tools releases.
    if hasattr(search_parameters.sat_parameters, "random_seed"):
        search_parameters.sat_parameters.random_seed = int(seed)
    elif hasattr(search_parameters, "random_seed"):
        search_parameters.random_seed = int(seed)
    search_parameters.log_search = False
    if time_limit is not None and time_limit > 0:
        search_parameters.time_limit.seconds = int(time_limit)
        search_parameters.time_limit.nanos = int(
            (float(time_limit) - int(time_limit)) * 1_000_000_000
        )

    started = time.perf_counter()
    solution = routing.SolveWithParameters(search_parameters)
    elapsed = time.perf_counter() - started
    status_value = routing.status()
    status_enum = getattr(routing_enums_pb2, "RoutingSearchStatus", None)
    if status_enum is not None and hasattr(status_enum, "Value"):
        status_name = status_enum.Value.Name(status_value)
    else:
        status_name = str(status_value)
    if solution is None:
        return TSPClassicalResult(
            solver="ortools",
            cost=float("nan"),
            actions=[],
            seconds=elapsed,
            status=status_name,
            meta={"time_limit": time_limit, "seed": seed},
            error="OR-Tools 没有返回可行解",
        )

    actions: list[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        actions.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    actions = _canonical_tour(actions)
    return TSPClassicalResult(
        solver="ortools",
        cost=_tour_cost(point_set, actions),
        actions=actions,
        seconds=elapsed,
        status=status_name,
        meta={
            "time_limit": time_limit,
            "seed": seed,
            "first_solution_strategy": first_solution_strategy,
            "local_search_metaheuristic": local_search_metaheuristic,
            "objective_scaled": int(solution.ObjectiveValue()),
        },
    )


def find_lkh_executable(lkh_path: str | None = None) -> str:
    """定位 LKH-3 可执行文件，始终返回绝对路径。

    返回绝对路径很重要：CVRP 求解时会在临时目录里启动子进程，相对路径会失效。
    """
    if lkh_path:
        path = Path(lkh_path)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        found = shutil.which(lkh_path)
        if found:
            return str(Path(found).resolve())
        raise FileNotFoundError(
            f"找不到可执行的 LKH-3 文件: {lkh_path}"
        )

    # Prefer the executable bundled under this repository.
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in sorted(
        (repo_root / "assets" / "solver").glob("LKH-*/LKH")
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    for name in ("LKH", "LKH-3", "lkh"):
        found = shutil.which(name)
        if found:
            return str(Path(found).resolve())
    raise FileNotFoundError(
        "找不到 LKH-3 可执行文件。请用 --lkh-path 指定，或把 LKH 加入 PATH。"
    )


def solve_tsp_lkh(
    locs,
    *,
    lkh_path: str | None = None,
    time_limit: float | None = None,
    runs: int = 1,
    seed: int = 1234,
) -> TSPClassicalResult:
    """Solve one Euclidean TSP by invoking the LKH-3 executable directly."""
    point_set = _as_locs(locs)
    n = len(point_set)
    if n == 0:
        return TSPClassicalResult(
            solver="lkh3",
            cost=0.0,
            actions=[],
            seconds=0.0,
            status="empty",
        )
    if n == 1:
        return TSPClassicalResult(
            solver="lkh3",
            cost=0.0,
            actions=[0],
            seconds=0.0,
            status="trivial",
        )
    if n <= 3:
        # LKH 要求 DIMENSION >= 3，两节点实例（仓库 + 单个客户）会直接报错；
        # 这类超小规模直接精确枚举即可。
        best_cost = math.inf
        best_actions: list[int] = []
        for perm in itertools.permutations(range(1, n)):
            tour = [0, *perm]
            cost = _tour_cost(point_set, tour)
            if cost < best_cost:
                best_cost = cost
                best_actions = tour
        return TSPClassicalResult(
            solver="lkh3",
            cost=float(best_cost),
            actions=best_actions,
            seconds=0.0,
            status="trivial",
            meta={"note": "n<=3 精确枚举（LKH 要求 DIMENSION>=3）"},
        )

    executable = find_lkh_executable(lkh_path)
    scale = 1_000_000
    matrix = np.rint(_distance_matrix(point_set) * scale).astype(np.int64)
    np.fill_diagonal(matrix, 0)

    started = time.perf_counter()
    workdir = Path(tempfile.mkdtemp(prefix="jp2dvrp_lkh_"))
    problem_file = workdir / "problem.tsp"
    tour_file = workdir / "tour.tour"
    parameter_file = workdir / "problem.par"

    problem_file.write_text(
        "\n".join(
            [
                "NAME: stage1_tsp",
                "TYPE: TSP",
                f"DIMENSION: {n}",
                "EDGE_WEIGHT_TYPE: EXPLICIT",
                "EDGE_WEIGHT_FORMAT: FULL_MATRIX",
                "EDGE_WEIGHT_SECTION",
                "\n".join(" ".join(map(str, row)) for row in matrix.tolist()),
                "EOF",
                "",
            ]
        )
    )
    parameter_lines = [
        "SPECIAL",
        f"PROBLEM_FILE = {problem_file}",
        f"TOUR_FILE = {tour_file}",
        f"RUNS = {int(runs)}",
        f"SEED = {int(seed)}",
    ]
    if time_limit is not None and time_limit > 0:
        parameter_lines.append(f"TIME_LIMIT = {float(time_limit):g}")
    parameter_file.write_text("\n".join(parameter_lines) + "\n")

    try:
        completed = subprocess.run(
            [executable, str(parameter_file)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - convert solver failures into data
        elapsed = time.perf_counter() - started
        return TSPClassicalResult(
            solver="lkh3",
            cost=float("nan"),
            actions=[],
            seconds=elapsed,
            status="failed",
            meta={
                "time_limit": time_limit,
                "runs": runs,
                "seed": seed,
                "executable": executable,
            },
            error=str(exc),
        )

    elapsed = time.perf_counter() - started
    if completed.returncode != 0 or not tour_file.is_file():
        return TSPClassicalResult(
            solver="lkh3",
            cost=float("nan"),
            actions=[],
            seconds=elapsed,
            status="failed",
            meta={
                "time_limit": time_limit,
                "runs": runs,
                "seed": seed,
                "executable": executable,
                "returncode": completed.returncode,
            },
            error=(completed.stdout or "").strip()[-1000:],
        )

    tour_values: list[int] = []
    in_tour = False
    for line in tour_file.read_text().splitlines():
        text = line.strip()
        if text == "TOUR_SECTION":
            in_tour = True
            continue
        if not in_tour:
            continue
        if text in {"-1", "EOF"}:
            break
        try:
            tour_values.append(int(text))
        except ValueError:
            continue
    shutil.rmtree(workdir, ignore_errors=True)

    if not tour_values:
        return TSPClassicalResult(
            solver="lkh3",
            cost=float("nan"),
            actions=[],
            seconds=elapsed,
            status="no_tour",
            meta={
                "time_limit": time_limit,
                "runs": runs,
                "seed": seed,
                "executable": executable,
            },
            error="LKH 没有返回 tour",
        )

    # LKH uses 1-indexed nodes.
    actions = [int(node) - 1 for node in tour_values]
    actions = _canonical_tour(actions)
    return TSPClassicalResult(
        solver="lkh3",
        cost=_tour_cost(point_set, actions),
        actions=actions,
        seconds=elapsed,
        status="success",
        meta={
            "time_limit": time_limit,
            "runs": runs,
            "seed": seed,
            "executable": executable,
            "returncode": completed.returncode,
        },
    )
