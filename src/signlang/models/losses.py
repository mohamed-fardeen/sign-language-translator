from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLossWrapper(nn.Module):
    def __init__(self, blank: int = 0, zero_infinity: bool = True, weight_decay: float = 1e-5) -> None:
        super().__init__()
        self.blank = blank
        self.zero_infinity = zero_infinity
        self.weight_decay = weight_decay
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        params: list[torch.nn.Parameter] | None = None,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        if self.weight_decay > 0 and params is not None:
            reg = sum((p * p).sum() for p in params)
            loss = loss + self.weight_decay * reg
        return loss