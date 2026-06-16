from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from signlang.utils.io import write_json
from signlang.utils.logging import get_logger

log = get_logger(__name__)


class MetricsJSONCallback(Callback):
    def __init__(self, out_path: str) -> None:
        super().__init__()
        self.out_path = out_path
        self.history: list[dict] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, _pl_module: pl.LightningModule) -> None:
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if hasattr(v, "item")}
        self.history.append({"epoch": trainer.current_epoch, **metrics})
        write_json(self.history, self.out_path)
