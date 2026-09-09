"""2D-Ptr style vehicle encoder migrated from the RANS HCVRP implementation.

The original code embedded heterogeneous vehicle states (time, capacity, used
capacity, speed), then applied vehicle self-attention and vehicle-to-node
cross-attention. Here the state machine (`HcvrpEnv`) is removed; the encoder
receives one-shot CVRP tensors and homogeneous vehicle states, and uses the
DPN rotary PE to distinguish identical vehicles.
"""

from __future__ import annotations

import torch
from torch import nn

from nets.blocks import AttentionBlock, CrossAttentionBlock
from nets.rotary_pe import DepotAwareRotaryPE


class VehicleEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int | None = None,
        use_rotary_pe: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_rotary_pe = use_rotary_pe

        # Migrated from old 2D-Ptr veh_encoder_mlp: Linear(4) -> GELU -> Linear
        self.state_mlp = nn.Sequential(
            nn.Linear(4, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.rotary_pe = DepotAwareRotaryPE(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.self_layers = nn.ModuleList(
            [
                AttentionBlock(embed_dim, num_heads=num_heads, ff_dim=ff_dim)
                for _ in range(num_layers)
            ]
        )
        self.cross = CrossAttentionBlock(embed_dim, num_heads=num_heads)
        self.fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        depot: torch.Tensor,
        num_vehicles: int,
        vehicle_capacity: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Return vehicle embeddings ``[B, M, embed_dim]``.

        Args:
            node_embeddings: ``[B, N, embed_dim]`` customer embeddings.
            depot: ``[B, 1, 2]`` or ``[B, 2]`` depot coordinates.
            num_vehicles: number of homogeneous vehicles ``M``.
            vehicle_capacity: normalized capacity (all vehicles share one value).
        """
        batch_size = node_embeddings.size(0)
        device = node_embeddings.device

        if isinstance(vehicle_capacity, torch.Tensor):
            capacity = vehicle_capacity.float().reshape(batch_size, 1).expand(
                batch_size, num_vehicles
            )
        else:
            capacity = torch.full(
                (batch_size, num_vehicles), float(vehicle_capacity), device=device
            )

        # Old 2D-Ptr state features: time, capacity, used capacity, speed
        time = torch.zeros_like(capacity)
        used = torch.zeros_like(capacity)
        speed = torch.ones_like(capacity)
        state = torch.stack((time, capacity, used, speed), dim=-1)  # [B, M, 4]
        embeddings = self.state_mlp(state)

        if self.use_rotary_pe:
            pe = self.rotary_pe(depot, num_vehicles)
            embeddings = embeddings + pe

        embeddings = self.norm(embeddings)

        # vehicle-to-node cross attention (migrated 2D-Ptr block)
        cross = self.cross(embeddings, node_embeddings)

        # vehicle self attention (migrated 2D-Ptr block)
        veh = embeddings
        for layer in self.self_layers:
            veh = layer(veh)

        fused = self.fusion(torch.cat((veh, cross), dim=-1))
        return fused + embeddings
