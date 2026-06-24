from __future__ import annotations

import torch

from signlang.models.backbones.transformer import TransformerBackbone
from signlang.models.encoders.stream_encoder import FeatureFusion, StreamEncoder
from signlang.models.heads.classification import ClassificationHead
from signlang.models.heads.ctc import CTCHead  # CTC: kept for reference
from signlang.models.losses import CTCLossWrapper  # CTC: kept for reference


def test_stream_encoder_output_shape() -> None:
    enc = StreamEncoder(in_dim=99, out_dim=128)
    x = torch.randn(2, 64, 99)
    y = enc(x)
    assert y.shape == (2, 64, 128)


def test_feature_fusion_concat() -> None:
    fus = FeatureFusion(stream_dim=128, n_streams=4, out_dim=512)
    streams = [torch.randn(2, 64, 128) for _ in range(4)]
    y = fus(streams)
    assert y.shape == (2, 64, 512)


def test_transformer_backbone() -> None:
    bb = TransformerBackbone(d_model=64, n_heads=4, n_layers=2, dim_feedforward=128)
    x = torch.randn(2, 64, 64)
    mask = torch.ones(2, 64, dtype=torch.bool)
    y = bb(x, mask=mask)
    assert y.shape == (2, 64, 64)


def test_ctc_head_output_size() -> None:
    # CTC kept for reference.
    head = CTCHead(in_dim=512, vocab_size=501)
    x = torch.randn(2, 64, 512)
    y = head(x)
    assert y.shape == (2, 64, 501)


def test_ctc_loss_runs() -> None:
    # CTC kept for reference.
    loss_fn = CTCLossWrapper(blank=0, zero_infinity=True, weight_decay=0.0)
    logits = torch.randn(2, 64, 501).log_softmax(dim=-1).requires_grad_(True)
    targets = torch.tensor([3, 5])
    input_lengths = torch.tensor([64, 64])
    target_lengths = torch.tensor([1, 1])
    loss = loss_fn(logits, targets, input_lengths, target_lengths)
    assert loss.item() > 0
    loss.backward()


# ---------------------------------------------------------------------------
# v1 classification additions.
# ---------------------------------------------------------------------------


def test_classification_head_output_shape() -> None:
    head = ClassificationHead(d_model=512, num_classes=500)
    x = torch.randn(2, 512)
    y = head(x)
    assert y.shape == (2, 500)


def test_classification_head_after_pooling_shape() -> None:
    """The ClassificationHead expects a pooled (B, d_model) tensor."""
    head = ClassificationHead(d_model=64, num_classes=10)
    x = torch.randn(4, 64)
    y = head(x)
    assert y.shape == (4, 10)