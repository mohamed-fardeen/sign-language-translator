from __future__ import annotations

import torch
import torch.nn as nn


class CTCHead(nn.Module):
    def __init__(self, in_dim: int, vocab_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(x))