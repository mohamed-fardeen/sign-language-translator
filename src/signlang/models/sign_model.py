"""SignModel — single-label isolated sign classifier (post-CTC v1).

CTC code is preserved as comments throughout per the v1 architecture
unfreeze rules. The model no longer uses CTC loss, beam search, or
greedy decode; it uses masked mean-pool + CrossEntropyLoss + argmax.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import DictConfig

from signlang.models.encoders.stream_encoder import FeatureFusion, StreamEncoder

# CTC kept for reference:
# from signlang.models.heads.ctc import CTCHead


@dataclass
class SignModelOutput:
    """Legacy container kept for backward compatibility.

    CTC mode returned ``logits`` of shape ``(B, T, V+1)`` together with
    a precomputed ``log_probs``. Classification mode returns logits of
    shape ``(B, num_classes)``; ``log_probs`` is exposed only for
    backwards-compat callers and is computed as softmax (not
    log_softmax).
    """

    logits: torch.Tensor
    log_probs: torch.Tensor


class SignModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})

        in_pose = int(cfg.in_dim_pose)
        in_hand = int(cfg.in_dim_hand)
        enc_dim = int(cfg.encoder_dim)
        n_streams = int(cfg.get("n_streams", 3))
        fusion_dim = int(cfg.fusion_dim)
        # Classification: num_classes == vocab_size (manifest labels are
        # 1..num_classes; the model internally uses 0..num_classes-1 and we
        # subtract 1 from targets at the loss call).
        num_classes = int(cfg.vocab_size)

        self.encoder_pose = StreamEncoder(in_pose, enc_dim)
        self.encoder_lh = StreamEncoder(in_hand, enc_dim)
        self.encoder_rh = StreamEncoder(in_hand, enc_dim)
        if n_streams == 4:
            in_face = int(cfg.in_dim_face)
            self.encoder_face = StreamEncoder(in_face, enc_dim)
        self.n_streams = n_streams
        self.fusion = FeatureFusion(enc_dim, n_streams=n_streams, out_dim=fusion_dim)

        backbone_cls = cfg.backbone["_target_"].rsplit(".", 1)[-1]
        from signlang.models.backbones.transformer import TransformerBackbone

        backbone: nn.Module
        if backbone_cls == "TransformerBackbone":
            backbone = TransformerBackbone(
                d_model=int(cfg.backbone.d_model),
                n_heads=int(cfg.backbone.n_heads),
                n_layers=int(cfg.backbone.n_layers),
                dim_feedforward=int(cfg.backbone.dim_feedforward),
                dropout=float(cfg.backbone.dropout),
                activation=str(cfg.backbone.activation),
                norm_first=bool(cfg.backbone.norm_first),
            )
        else:  # pragma: no cover - guarded by config
            raise ValueError(f"Unknown backbone: {backbone_cls}")
        self.backbone = backbone

        # CTC kept for reference:
        # self.head = CTCHead(in_dim=fusion_dim, vocab_size=vocab_size)
        from signlang.models.heads.classification import ClassificationHead

        self.head = ClassificationHead(d_model=fusion_dim, num_classes=num_classes)

        # CTC kept for reference:
        # from signlang.models.losses import CTCLossWrapper
        # self.loss_fn = CTCLossWrapper(
        #     blank=0,
        #     zero_infinity=True,
        #     weight_decay=float(cfg.loss.weight_decay),
        # )
        self.loss_fn = nn.CrossEntropyLoss()

        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.val_top5 = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=min(5, num_classes)
        )

    def forward(
        self,
        pose: torch.Tensor,
        lh: torch.Tensor,
        rh: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode -> fuse -> transformer -> masked mean pool -> classify.

        CTC: previously returned ``SignModelOutput(logits, log_probs)`` of
        shape ``(B, T, V+1)``. Classification returns raw logits of
        shape ``(B, num_classes)``.
        """
        p = self.encoder_pose(pose)
        l = self.encoder_lh(lh)
        r = self.encoder_rh(rh)
        streams = [p, l, r]
        # CTC face stream kept for reference (unused in v1 single-label):
        # if self.n_streams == 4:
        #     f = self.encoder_face(face)
        #     streams.append(f)
        fused = self.fusion(streams)
        x = self.backbone(fused, mask=mask)
        # Mean-pool over time. If a frame mask is provided, use masked
        # averaging so padded/zero frames don't contaminate the result.
        if mask is not None:
            m = mask.float().unsqueeze(-1)
            denom = m.sum(dim=1).clamp(min=1.0)
            features = (x * m).sum(dim=1) / denom
        else:
            features = x.mean(dim=1)
        return self.head(features)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        logits = self.forward(
            batch["pose"],
            batch["lh"],
            batch["rh"],
            mask=batch.get("mask"),
        )
        labels = batch["label"].long()  # (B,), manifest labels 1..num_classes

        # CTC kept for reference (v1: single-label cross-entropy):
        # targets = labels
        # target_lengths = torch.ones(
        #     logits.size(0), dtype=torch.long, device=labels.device
        # )
        # mask_tensor = batch.get("mask")
        # if isinstance(mask_tensor, torch.Tensor):
        #     input_lengths = mask_tensor.sum(dim=1).long().clamp(min=1)
        # else:
        #     input_lengths = torch.full(
        #         (logits.size(0),), logits.size(1),
        #         dtype=torch.long, device=labels.device,
        #     )
        # loss = self.loss_fn(
        #     out.logits,
        #     targets,
        #     input_lengths,
        #     target_lengths,
        #     params=list(self.parameters()),
        # )
        # v1: cross-entropy on (B, num_classes) logits. Manifest labels are
        # 1-based; the model uses 0..num_classes-1 internally, so subtract
        # 1 before passing targets.
        target = labels - 1
        loss = self.loss_fn(logits, target)

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        if stage in {"val", "test"}:
            pred = logits.argmax(dim=1)
            self.val_acc(pred, target)
            self.val_top5(logits, target)
            self.log(f"{stage}/accuracy", self.val_acc, prog_bar=True, on_epoch=True)
            self.log(f"{stage}/top5_accuracy", self.val_top5, on_epoch=True)
        return loss

    def training_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        from signlang.training.optimizers import build_optimizer
        from signlang.training.schedulers import build_scheduler

        opt_cfg = self.hparams.cfg.get("optimizer") or {}
        sch_cfg = self.hparams.cfg.get("scheduler") or {}
        opt = build_optimizer(self.parameters(), opt_cfg)
        sch = build_scheduler(opt, sch_cfg, trainer=self.trainer)
        return {"optimizer": opt, "lr_scheduler": sch}