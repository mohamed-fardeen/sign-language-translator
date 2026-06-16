from __future__ import annotations

import argparse
from pathlib import Path

from signlang.config import load_config, to_dict
from signlang.data.datamodules import SignDataModule
from signlang.models.sign_model import SignModel
from signlang.tracking.mlflow_utils import log_params, setup_mlflow
from signlang.training.callbacks.metrics_json import MetricsJSONCallback
from signlang.training.trainer import (
    attach_mlflow_logger,
    build_callbacks,
    build_trainer,
    log_git_metadata,
    make_reproducible,
)
from signlang.utils.io import write_json
from signlang.utils.logging import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Hydra data config name (e.g., wlasl)")
    parser.add_argument("--model", default=None, help="Hydra model config name")
    parser.add_argument("--train", default=None, help="Hydra train config name")
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO", json_logs=False)
    log = get_logger("train")

    overrides: list[str] = []
    if args.data:
        overrides.append(f"data={args.data}")
    if args.model:
        overrides.append(f"model={args.model}")
    if args.train:
        overrides.append(f"train={args.train}")
    if args.max_epochs is not None:
        overrides.append(f"train.trainer.max_epochs={args.max_epochs}")
    if args.batch_size is not None:
        overrides.append(f"train.trainer.batch_size={args.batch_size}")
    if args.lr is not None:
        overrides.append(f"train.optimizer.lr={args.lr}")
    cfg = load_config(overrides)
    cfg_dict = to_dict(cfg)
    log.info("train.config.loaded", keys=list(cfg_dict.keys()))

    make_reproducible(int(cfg_dict.get("seed", 42)))
    paths = cfg_dict["paths"]
    features_dir = Path(args.features_dir) if args.features_dir else Path(paths["features_dir"])

    dm = SignDataModule(
        features_dir=features_dir,
        batch_size=int(cfg_dict["train"]["trainer"].get("batch_size", 32)),
        num_workers=int(cfg_dict["train"]["trainer"].get("num_workers", 4)),
    )
    model = SignModel(cfg_dict["model"])
    log.info("train.model.built", params=sum(p.numel() for p in model.parameters()))

    callbacks = build_callbacks(
        checkpoint_dir=paths["checkpoint_dir"],
        early_stopping_cfg=cfg_dict["train"].get("early_stopping"),
        checkpoint_cfg=cfg_dict["train"].get("checkpoint"),
    )
    callbacks.append(MetricsJSONCallback(str(Path(paths["log_dir"]) / "metrics.json")))

    trainer = build_trainer(cfg_dict["train"]["trainer"], callbacks=callbacks)

    if not args.no_mlflow:
        setup_mlflow(
            tracking_uri=cfg_dict["mlflow"]["tracking_uri"],
            experiment=cfg_dict["mlflow"]["experiment"],
        )
        attach_mlflow_logger(
            trainer,
            tracking_uri=cfg_dict["mlflow"]["tracking_uri"],
            experiment=cfg_dict["mlflow"]["experiment"],
            run_name=cfg_dict["mlflow"]["run_name"],
        )
        log_git_metadata(trainer.logger)
        log_params(cfg_dict)

    trainer.fit(model, datamodule=dm)
    log.info("train.done", best_ckpt=trainer.checkpoint_callback.best_model_path)
    if trainer.checkpoint_callback.best_model_path:
        write_json(
            {"best_ckpt": trainer.checkpoint_callback.best_model_path},
            Path(paths["log_dir"]) / "best_ckpt.json",
        )


if __name__ == "__main__":
    main()
