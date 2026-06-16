from __future__ import annotations

from pathlib import Path

import mlflow

from signlang.utils.logging import get_logger

log = get_logger(__name__)


def setup_mlflow(tracking_uri: str, experiment: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


def log_params(params: dict) -> None:
    if not mlflow.active_run():
        return
    mlflow.log_params({k: v for k, v in params.items() if _is_loggable(v)})


def log_metrics(metrics: dict, step: int | None = None) -> None:
    if not mlflow.active_run():
        return
    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if _is_loggable(v)}, step=step)


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    if not mlflow.active_run():
        return
    mlflow.log_artifact(str(path), artifact_path)


def start_run(run_name: str | None = None):
    return mlflow.start_run(run_name=run_name)


def end_run() -> None:
    if mlflow.active_run():
        mlflow.end_run()


def _is_loggable(v: object) -> bool:
    return isinstance(v, (int, float, str, bool))
