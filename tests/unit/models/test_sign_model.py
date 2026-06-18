from __future__ import annotations

import torch
from omegaconf import OmegaConf

from signlang.models.sign_model import SignModel


def _tiny_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "vocab_size": 50,
            "blank_id": 0,
            "clip_frames": 16,
            "in_dim_pose": 99,
            "in_dim_hand": 63,
            "n_streams": 3,
            "encoder_dim": 32,
            "fusion_dim": 64,
            "backbone": {
                "_target_": "signlang.models.backbones.transformer.TransformerBackbone",
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 2,
                "dim_feedforward": 128,
                "dropout": 0.0,
                "activation": "gelu",
                "norm_first": True,
            },
            "head": {"_target_": "signlang.models.heads.ctc.CTCHead", "in_dim": 64, "vocab_size": 51},
            "loss": {"blank": 0, "zero_infinity": True, "weight_decay": 0.0},
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.98]},
            "scheduler": {"name": "cosine", "warmup_ratio": 0.0},
        }
    )


def test_sign_model_forward_shapes() -> None:
    model = SignModel(_tiny_cfg())
    B, T = 2, 16
    pose = torch.randn(B, T, 99)
    lh = torch.randn(B, T, 63)
    rh = torch.randn(B, T, 63)
    out = model(pose, lh, rh)
    assert out.logits.shape == (B, T, 51)


def test_sign_model_training_step_runs() -> None:
    model = SignModel(_tiny_cfg())
    B, T = 2, 16
    batch = {
        "pose": torch.randn(B, T, 99),
        "lh": torch.randn(B, T, 63),
        "rh": torch.randn(B, T, 63),
        "mask": torch.ones(B, T, dtype=torch.bool),
        "label": torch.tensor([1, 2]),
    }
    loss = model.training_step(batch, 0)
    assert loss.requires_grad
    assert float(loss) > 0
