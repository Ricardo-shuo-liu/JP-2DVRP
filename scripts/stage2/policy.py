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
        penalty: float = 10.0,
    ):
        super().__init__()
        self.partition_network = partition_network
        self.route_evaluator = route_evaluator
        self.num_vehicles = num_vehicles
        self.penalty = penalty

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

        logits = self.partition_network(td, self.num_vehicles)  # [B,M,N]
        logp = torch.log_softmax(logits, dim=1)  # per-node over vehicles

        if decode_type == "greedy" or phase != "train":
            assignments = logits.argmax(dim=1)  # [B,N]
        else:
            dist = Categorical(logits=logits.transpose(1, 2))  # [B,N,M]
            assignments = dist.sample()  # [B,N]

        assign_logp = (
            logp.gather(1, assignments.unsqueeze(1)).squeeze(1).sum(dim=-1)
        )

        costs, overflow = self.route_evaluator.evaluate(td, assignments)
        reward = -(costs + self.penalty * overflow)

        return {
            "reward": reward,
            "log_likelihood": assign_logp,
            "assignments": assignments,
            "cost": costs,
            "overflow": overflow,
        }
