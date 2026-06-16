from __future__ import annotations

import torch
import torch.nn as nn


class StreamEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureFusion(nn.Module):
    def __init__(self, stream_dim: int, n_streams: int, out_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(stream_dim * n_streams, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.cat(streams, dim=-1))