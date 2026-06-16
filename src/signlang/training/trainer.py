from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from signlang.utils.logging import get_logger
from signlang.utils.seeding import seed_everything

log = get_logger(__name__)


def build_callbacks(
    checkpoint_dir: str | Path,
    early_stopping_cfg: dict | None = None,
    checkpoint_cfg: dict | None = None,
) -> list[pl.Callback]:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cfg = checkpoint_cfg or {}
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=str(ckpt_cfg.get("monitor", "val/accuracy")),
            mode=str(ckpt_cfg.get("mode", "max")),
            save_top_k=int(ckpt_cfg.get("save_top_k", 3)),
            filename=str(ckpt_cfg.get("filename", "sign-{epoch:02d}-{val/accuracy:.4f}")),
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    es = early_stopping_cfg or {}
    if bool(es.get("enabled", True)):
        callbacks.append(
            EarlyStopping(
                monitor=str(es.get("monitor", "val/accuracy")),
                mode=str(es.get("mode", "max")),
                patience=int(es.get("patience", 10)),
            )
        )
    return callbacks


def build_trainer(
    trainer_cfg: dict,
    callbacks: list[pl.Callback],
    logger: pl.loggers.Logger | None = None,
) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 80)),
        accelerator=str(trainer_cfg.get("accelerator", "auto")),
        devices=int(trainer_cfg.get("devices", 1)),
        precision=str(trainer_cfg.get("precision", "bf16-mixed")),
        accumulate_grad_batches=int(trainer_cfg.get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 20)),
        enable_progress_bar=bool(trainer_cfg.get("enable_progress_bar", True)),
        deterministic=bool(trainer_cfg.get("deterministic", False)),
        callbacks=callbacks,
        logger=logger,
    )


def make_reproducible(seed: int) -> None:
    seed_everything(seed)
    log.info("seed.set", seed=seed)


def attach_mlflow_logger(
    trainer: pl.Trainer,
    tracking_uri: str,
    experiment: str,
    run_name: str | None = None,
) -> pl.loggers.Logger:
    import mlflow
    from pytorch_lightning.loggers import MLFlowLogger

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    logger = MLFlowLogger(
        experiment_name=experiment,
        tracking_uri=tracking_uri,
        run_name=run_name or "run",
    )
    trainer.logger = logger
    return logger


def log_git_metadata(logger: Any) -> None:
    try:
        import subprocess

        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        logger.log_hyperparams({"git_sha": sha})
    except Exception:
        pass
