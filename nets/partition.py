"""One-shot partition network for Stage2.

The partition head returns logits ``[batch, num_vehicles, num_nodes]``. A
softmax over the vehicle dimension turns them into the assignment policy used
by ``scripts/stage2``. Node features follow the original 2D-Ptr input
(x, y, demand) and homogeneous vehicles are distinguished by DPN rotary PE.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from nets.blocks import AttentionBlock
from nets.vehicle_encoder import VehicleEncoder


class NodeEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depot_embed = nn.Linear(2, embed_dim)
        self.node_embed = nn.Linear(3, embed_dim)
        self.layers = nn.ModuleList(
            [
                AttentionBlock(embed_dim, num_heads=num_heads, ff_dim=ff_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        depot: torch.Tensor,
        customer_locs: torch.Tensor,
        demand: torch.Tensor,
    ) -> torch.Tensor:
        depot_emb = self.depot_embed(depot)  # [B,1,d]
        node_feats = torch.cat((customer_locs, demand.unsqueeze(-1)), dim=-1)
        node_emb = self.node_embed(node_feats)  # [B,N,d]
        x = torch.cat((depot_emb, node_emb), dim=1)  # [B,N+1,d]
        for layer in self.layers:
            x = layer(x)
        return x


class PartitionNetwork(nn.Module):
    """Outputs ``[B, M, N]`` unnormalized assignment logits for customer nodes."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.node_encoder = NodeEncoder(
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_encoder_layers
        )
        self.vehicle_encoder = VehicleEncoder(
            embed_dim=embed_dim, num_heads=num_heads, num_layers=max(2, num_encoder_layers)
        )
        self.logit_scale = embed_dim ** 0.5

    def forward(self, td: TensorDict, num_vehicles: int) -> torch.Tensor:
        locs = td["locs"]
        if locs.dim() != 3 or locs.size(-1) != 2:
            raise ValueError(f"locs 应为 [B, N, 2]，当前 {tuple(locs.shape)}")
        depot = locs[..., :1, :]  # [B,1,2]
        customer_locs = locs[..., 1:, :]  # [B,N,2]
        demand = td["demand"]
        if demand.dim() == 3 and demand.size(-1) == 1:
            demand = demand.squeeze(-1)

        x = self.node_encoder(depot, customer_locs, demand)  # [B,N+1,d]
        node_emb = x[:, 1:, :]  # customer-only embeddings

        capacity = td.get("vehicle_capacity", None)
        if capacity is None:
            capacity = 1.0
        veh_emb = self.vehicle_encoder(
            node_emb, depot, num_vehicles, vehicle_capacity=capacity
        )
        logits = torch.matmul(veh_emb, node_emb.transpose(-1, -2))
        logits = logits / self.logit_scale
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return logits  # [B,M,N]
