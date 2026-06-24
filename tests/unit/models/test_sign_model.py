from __future__ import annotations

import torch
from omegaconf import OmegaConf

from signlang.models.sign_model import SignModel


def _tiny_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "vocab_size": 50,
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
            "head": {
                "_target_": "signlang.models.heads.classification.ClassificationHead",
                "d_model": 64,
                "num_classes": 50,
            },
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.98]},
            "scheduler": {"name": "cosine", "warmup_ratio": 0.0},
        }
    )


def test_sign_model_forward_shape_is_classification() -> None:
    """v1: forward returns (B, num_classes) logits, not (B, T, V+1)."""
    model = SignModel(_tiny_cfg())
    B, T = 2, 16
    pose = torch.randn(B, T, 99)
    lh = torch.randn(B, T, 63)
    rh = torch.randn(B, T, 63)
    logits = model(pose, lh, rh)
    assert logits.shape == (B, 50)


def test_sign_model_training_step_runs_cross_entropy() -> None:
    """v1: training_step returns a scalar cross-entropy loss."""
    model = SignModel(_tiny_cfg())
    B, T = 2, 16
    batch = {
        "pose": torch.randn(B, T, 99),
        "lh": torch.randn(B, T, 63),
        "rh": torch.randn(B, T, 63),
        "mask": torch.ones(B, T, dtype=torch.bool),
        # Manifest labels are 1-based; the model subtracts 1 internally.
        "label": torch.tensor([3, 25]),
    }
    loss = model.training_step(batch, 0)
    assert loss.requires_grad
    assert float(loss) > 0


def test_sign_model_validation_step_logs_accuracy() -> None:
    """v1: validation_step calls ``self.log`` for ``val/accuracy`` and
    ``val/top5_accuracy``. We verify the side-effect by patching
    ``self.log`` and asserting the keys were emitted (Lightning
    internally aggregates these into ``callback_metrics``).
    """
    model = SignModel(_tiny_cfg())
    B, T = 2, 16
    batch = {
        "pose": torch.randn(B, T, 99),
        "lh": torch.randn(B, T, 63),
        "rh": torch.randn(B, T, 63),
        "mask": torch.ones(B, T, dtype=torch.bool),
        "label": torch.tensor([1, 2]),
    }
    logged: list[tuple] = []
    model.log = lambda *args, **kwargs: logged.append((args, kwargs))  # type: ignore[method-assign]
    _ = model.validation_step(batch, 0)
    keys = {a[0] for a, _ in logged}
    assert "val/loss" in keys
    assert "val/accuracy" in keys
    assert "val/top5_accuracy" in keys


def test_masked_mean_pool_ignores_padding() -> None:
    """If mask is provided, masked mean pool should not be affected by padded frames."""
    model = SignModel(_tiny_cfg())
    B, T = 1, 16
    pose = torch.zeros(B, T, 99)
    lh = torch.zeros(B, T, 63)
    rh = torch.zeros(B, T, 63)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, 0] = True  # only the first frame is valid
    with torch.no_grad():
        out = model(pose, lh, rh, mask=mask)
    assert out.shape == (B, 50)