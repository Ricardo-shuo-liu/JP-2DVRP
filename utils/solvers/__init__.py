"""Optional classical solvers used as evaluation baselines."""

from .cvrp import (
    CVRPSolution,
    evaluate_routes,
    solve_cvrp_lkh,
    solve_cvrp_lkh_minmax,
    solve_cvrp_ortools,
)
from .tsp import (
    TSPClassicalResult,
    find_lkh_executable,
    solve_tsp_lkh,
    solve_tsp_ortools,
)

__all__ = [
    "CVRPSolution",
    "TSPClassicalResult",
    "evaluate_routes",
    "find_lkh_executable",
    "solve_cvrp_lkh",
    "solve_cvrp_lkh_minmax",
    "solve_cvrp_ortools",
    "solve_tsp_lkh",
    "solve_tsp_ortools",
]
