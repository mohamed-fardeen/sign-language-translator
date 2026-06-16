from __future__ import annotations

import math
from typing import Any

from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def build_scheduler(optimizer, cfg, trainer: Any | None = None) -> LRScheduler:
    name = str(cfg.get("name", "cosine")).lower()
    warmup_ratio = float(cfg.get("warmup_ratio", 0.05))

    total_steps: int | None = None
    if trainer is not None:
        try:
            total_steps = int(trainer.estimated_stepping_batches)
        except AttributeError:
            total_steps = None
    if total_steps is None or total_steps <= 0:
        total_steps = int(cfg.get("total_steps", 10_000))

    warmup_steps = max(1, round(warmup_ratio * total_steps))

    if name == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    if name == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    raise ValueError(f"Unknown scheduler: {name}")
