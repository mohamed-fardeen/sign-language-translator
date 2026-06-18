from __future__ import annotations

from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import DictConfig

from signlang.models.encoders.stream_encoder import FeatureFusion, StreamEncoder


@dataclass
class SignModelOutput:
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
        vocab_size = int(cfg.vocab_size) + 1

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

        from signlang.models.heads.ctc import CTCHead

        self.head = CTCHead(in_dim=fusion_dim, vocab_size=vocab_size)

        from signlang.models.losses import CTCLossWrapper

        self.loss_fn = CTCLossWrapper(
            blank=int(cfg.head.vocab_size - vocab_size) if False else 0,
            zero_infinity=bool(cfg.loss.zero_infinity),
            weight_decay=float(cfg.loss.weight_decay),
        )

        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=vocab_size, top_k=1
        )
        self.val_top5 = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=vocab_size, top_k=5
        )

    def forward(self, pose, lh, rh, face=None, mask=None) -> SignModelOutput:
        p = self.encoder_pose(pose)
        l = self.encoder_lh(lh)
        r = self.encoder_rh(rh)
        streams = [p, l, r]
        if self.n_streams == 4 and face is not None:
            f = self.encoder_face(face)
            streams.append(f)
        fused = self.fusion(streams)
        x = self.backbone(fused, mask=mask)
        logits = self.head(x)
        return SignModelOutput(logits=logits, log_probs=torch.log_softmax(logits, dim=-1))

    def _step(self, batch, stage: str) -> torch.Tensor:
        out = self.forward(
            batch["pose"],
            batch["lh"],
            batch["rh"],
            face=batch.get("face"),
            mask=batch.get("mask"),
        )
        B, T, _V = out.logits.shape
        labels = batch["label"]
        targets = labels
        target_lengths = torch.ones(B, dtype=torch.long, device=labels.device)
        mask_tensor = batch.get("mask")
        if isinstance(mask_tensor, torch.Tensor):
            input_lengths = mask_tensor.sum(dim=1).long().clamp(min=1)
        else:
            input_lengths = torch.full((B,), T, dtype=torch.long, device=labels.device)

        loss = self.loss_fn(
            out.logits,
            targets,
            input_lengths,
            target_lengths,
            params=list(self.parameters()),
        )
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        if stage in {"val", "test"}:
            preds = out.logits.argmax(dim=-1)
            self.val_acc(preds, labels)
            self.val_top5(out.logits, labels)
            self.log(f"{stage}/accuracy", self.val_acc, prog_bar=True, on_epoch=True)
            self.log(f"{stage}/top5_accuracy", self.val_top5, on_epoch=True)
        return loss

    def training_step(self, batch, _batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, _batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch, _batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        from signlang.training.optimizers import build_optimizer
        from signlang.training.schedulers import build_scheduler

        opt_cfg = self.hparams.cfg.get("optimizer") or {}
        sch_cfg = self.hparams.cfg.get("scheduler") or {}
        opt = build_optimizer(self.parameters(), opt_cfg)
        sch = build_scheduler(opt, sch_cfg, trainer=self.trainer)
        return {"optimizer": opt, "lr_scheduler": sch}