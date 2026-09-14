"""Stage2 partition policy.

The policy samples one cluster assignment per customer (independent over the
node dimension), evaluates all clusters with the frozen Stage1 route model and
returns the REINFORCE quantities ``reward`` and ``log_likelihood`` for the
assignment action.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions import Categorical

from nets.partition import PartitionNetwork

from scripts.stage2.route_evaluator import RouteEvaluator


class Stage2Policy(nn.Module):
    def __init__(
        self,
        partition_network: PartitionNetwork,
        route_evaluator: RouteEvaluator,
        num_vehicles: int,
        objective: str = "minmax",
        penalty: float = 10.0,
        capacity_mask_mode: str = "static",
    ):
        super().__init__()
        self.partition_network = partition_network
        self.route_evaluator = route_evaluator
        self.num_vehicles = num_vehicles
        if objective not in ("sum", "minmax"):
            raise ValueError(f"未知 objective: {objective}")
        if capacity_mask_mode not in ("none", "static"):
            raise ValueError(f"未知 capacity_mask_mode: {capacity_mask_mode}")
        self.objective = objective
        self.penalty = penalty
        self.capacity_mask_mode = capacity_mask_mode

    @staticmethod
    def _sample_independent(logits, phase, decode_type):
        """Sample/argmax one vehicle per node from [B, M, N] logits."""
        logp = torch.log_softmax(logits, dim=1)
        if decode_type == "greedy" or phase != "train":
            assignments = logits.argmax(dim=1)
        else:
            dist = Categorical(logits=logits.transpose(1, 2))
            assignments = dist.sample()
        assign_logp = logp.gather(
            1, assignments.unsqueeze(1)
        ).squeeze(1).sum(dim=-1)
        return assignments, assign_logp

    def _capacity_aware_assignments(
        self,
        logits: torch.Tensor,
        demand: torch.Tensor,
        capacity: torch.Tensor,
        phase: str,
        decode_type: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign nodes while disabling vehicle choices that exceed capacity.

        ``sequential`` uses a deterministic demand-descending order and masks
        each vehicle whose current load plus the current demand is infeasible.
        The resulting log-likelihood is the sum of the masked per-node log
        probabilities, so REINFORCE remains well-defined.
        """
        if self.capacity_mask_mode == "none":
            return self._sample_independent(logits, phase, decode_type)

        # Static capacity mask: for each node, disable vehicles whose capacity
        # is smaller than that node's demand.  This is independent over nodes.
        feasible = demand.unsqueeze(1) <= (
            capacity.unsqueeze(-1) + 1e-6
        )
        any_feasible = feasible.any(dim=1, keepdim=True)
        # If a node cannot fit any vehicle, keep choices enabled as a fallback;
        # the post-assignment overflow and total-capacity penalties handle it.
        feasible = torch.where(
            any_feasible, feasible, torch.ones_like(feasible)
        )
        masked_logits = logits.masked_fill(~feasible, float("-inf"))
        return self._sample_independent(masked_logits, phase, decode_type)

    def forward(
        self,
        td: TensorDict,
        env=None,
        phase: str = "train",
        decode_type: str | None = None,
        **kwargs,
    ) -> dict:
        # td is already reset by REINFORCE.shared_step
        locs = td["locs"]
        demand = td["demand"]
        if demand.dim() == 3:
            demand = demand.squeeze(-1)
        batch_size, num_nodes = demand.shape

        capacity = td.get("vehicle_capacity", None)
        if capacity is None:
            capacity = torch.ones(batch_size, 1, device=demand.device)
        capacity = capacity.reshape(batch_size, 1)

        logits = self.partition_network(td, self.num_vehicles)  # [B,M,N]
        assignments, assign_logp = self._capacity_aware_assignments(
            logits, demand, capacity, phase, decode_type
        )

        costs, overflow = self.route_evaluator.evaluate(
            td, assignments
        )
        penalized_costs = costs + self.penalty * overflow
        if self.objective == "minmax":
            objective_cost = penalized_costs.max(dim=1).values
        else:
            objective_cost = penalized_costs.sum(dim=1)
        reward = -objective_cost

        return {
            "reward": reward,
            "log_likelihood": assign_logp,
            "assignments": assignments,
            "cost": costs,
            "objective_cost": objective_cost,
            "overflow": overflow.sum(dim=1),
            "overflow_per_vehicle": overflow,
        }
