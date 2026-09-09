"""Shared dataset helpers used by Stage1/Stage2.

The canonical npz formats match what RL4CO's environment dataset loaders expect:

- TSP:  ``locs`` with shape ``[num_samples, num_loc, 2]``.
- CVRP: raw integer ``demand`` together with raw ``capacity``, because
  ``CVRPEnv.load_data`` normalizes ``demand`` by ``capacity`` on load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from tensordict import TensorDict

from rl4co.data.utils import save_tensordict_to_npz
from rl4co.envs import CVRPEnv, TSPEnv
from rl4co.envs.routing.cvrp.generator import CVRPGenerator


class FeasibleCVRPGenerator(CVRPGenerator):
    """CVRP generator rejecting instances that cannot fit ``num_vehicles``."""

    def __init__(self, num_vehicles: int, max_attempts: int = 10_000, **kwargs):
        super().__init__(**kwargs)
        self.num_vehicles = num_vehicles
        self.max_attempts = max_attempts

    def _generate(self, batch_size) -> TensorDict:
        target = int(batch_size[0])
        accepted: list[TensorDict] = []
        accepted_count = 0
        for _ in range(self.max_attempts):
            td = super()._generate(batch_size)
            # demand is already normalized to capacity
            feasible = td["demand"].sum(-1) <= self.num_vehicles + 1e-5
            if not bool(feasible.any()):
                continue
            remaining = target - accepted_count
            idx = torch.nonzero(feasible, as_tuple=False).squeeze(-1)[:remaining]
            keep = torch.zeros_like(feasible)
            keep[idx] = True
            accepted.append(td[keep])
            accepted_count += int(keep.sum())
            if accepted_count >= target:
                return torch.cat(accepted, dim=0)
        raise RuntimeError(
            f"无法生成满足 {self.num_vehicles} 辆车容量的 CVRP batch；"
            "请增大 --num-vehicles 或调小需求"
        )


def add_npz_extension(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".npz" else Path(f"{path}.npz")


def make_env(
    env_name: str,
    *,
    num_loc: int,
    data_dir: str = "data",
    train_file: Optional[str] = None,
    val_file: Optional[str] = None,
    test_file: Optional[str] = None,
    generator_params: Optional[dict] = None,
    check_solution: bool = False,
):
    """Build an RL4CO environment from its public class name."""
    if env_name == "TSPEnv":
        env_cls = TSPEnv
    elif env_name == "CVRPEnv":
        env_cls = CVRPEnv
    else:
        from rl4co import envs

        env_cls = getattr(envs, env_name, None)
        if env_cls is None:
            raise ValueError(f"rl4co.envs 中不存在环境类 {env_name!r}")

    params = {"num_loc": num_loc}
    params.update(generator_params or {})
    return env_cls(
        generator_params=params,
        data_dir=data_dir,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        check_solution=check_solution,
    )


def canonical_cvrp_td(td: TensorDict) -> TensorDict:
    """Convert a raw RL4CO CVRP TensorDict into a loadable canonical npz format.

    RL4CO's CVRPGenerator returns already normalized demand plus the raw
    ``capacity``. For files we store the raw demand so that
    ``CVRPEnv.load_data`` performs exactly one normalization step.
    """
    capacity = td["capacity"]  # [B,1] from RL4CO CVRPGenerator
    return TensorDict(
        {
            "locs": td["locs"],
            "depot": td["depot"],
            "demand": td["demand"] * capacity,
            "capacity": capacity.squeeze(-1),  # official npz format: [B]
        },
        batch_size=td.batch_size,
    )


def save_td(td: TensorDict, path: str | Path, compress: bool = False) -> Path:
    path = add_npz_extension(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_tensordict_to_npz(td, path, compress=compress)
    return path


def feasible_cvrp_mask(td: TensorDict, num_vehicles: int) -> torch.Tensor:
    """Boolean mask of instances that can fit into ``num_vehicles`` vehicles."""
    capacity = td["capacity"].squeeze(-1)
    total_demand = td["demand"].sum(-1)
    return total_demand <= capacity * num_vehicles


def select_by_mask(td: TensorDict, mask: torch.Tensor) -> TensorDict:
    return td[mask]
