"""Small transformer building blocks shared by Stage2 partition networks."""

from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionBlock(nn.Module):
    """Pre-LN transformer block with multi-head self-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            attn_mask=mask,
            need_weights=False,
        )
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Query attends over key/value items."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query)
        kv = self.kv_proj(key_value)
        k, v = torch.chunk(kv, 2, dim=-1)
        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.embed_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return self.out_proj(out)
