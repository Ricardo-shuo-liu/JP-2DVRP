"""DPN-style rotation based positional encoding for depot-aware vehicle identity."""

from __future__ import annotations

import torch
from torch import nn


class DepotAwareRotaryPE(nn.Module):
    """Rotation-based PE from DPN.

    ``PE(m, q) = Re[(x_d W_a + b_a) exp(i m / 1000^{floor(q/2)/d})]``

    The depot coordinates are first projected to ``[B, 1, d]``. Treating each
    consecutive pair as one complex coordinate and rotating it by an
    angle that depends on vehicle index ``m`` gives ``[B, M, d]``.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        assert embed_dim % 2 == 0, "rotary PE requires even embed_dim"
        self.embed_dim = embed_dim
        self.depot_proj = nn.Linear(2, embed_dim, bias=True)

    def forward(self, depot: torch.Tensor, num_vehicles: int) -> torch.Tensor:
        # depot: [B, 1, 2] or [B, 2]
        if depot.dim() == 2:
            depot = depot.unsqueeze(1)
        z = self.depot_proj(depot)  # [B, 1, d]
        batch_size = z.size(0)
        d = self.embed_dim

        device = z.device
        m = torch.arange(num_vehicles, device=device).float()  # [M]
        half_idx = torch.arange(d // 2, device=device).float()
        # 1000 ** (idx / d), exactly following the DPN exponent
        frequency = 1000.0 ** (half_idx / d)  # [d/2]
        angle = m[:, None] / frequency[None, :]  # [M, d/2]

        cos_a = torch.cos(angle)  # [M, d/2]
        sin_a = torch.sin(angle)

        # z: [B, 1, d/2, 2] -> real, imag
        z_pairs = z.view(batch_size, 1, d // 2, 2)
        zr = z_pairs[..., 0]
        zi = z_pairs[..., 1]
        # complex multiply by e^{i angle}
        real = zr * cos_a[None] - zi * sin_a[None]
        imag = zr * sin_a[None] + zi * cos_a[None]
        out = torch.stack((real, imag), dim=-1).view(batch_size, num_vehicles, d)
        return out
