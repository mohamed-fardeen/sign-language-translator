"""Classification head for single-label isolated sign recognition.

Replaces the CTC head (which is kept for reference under
``signlang.models.heads.ctc``). Pools the temporal dimension to a single
vector before applying the linear projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Linear projection to ``num_classes`` logits (per clip, after pooling)."""

    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)