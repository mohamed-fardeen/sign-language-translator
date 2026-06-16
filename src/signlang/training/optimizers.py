from __future__ import annotations

import torch


def build_optimizer(params, cfg) -> torch.optim.Optimizer:
    name = str(cfg.get("name", "adamw")).lower()
    lr = float(cfg.get("lr", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    betas = tuple(cfg.get("betas", (0.9, 0.98)))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")
