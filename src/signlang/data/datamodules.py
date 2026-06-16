from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from signlang.data.datasets.clip_dataset import SignClipDataset


class SignDataModule(pl.LightningDataModule):
    def __init__(
        self,
        features_dir: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        augment_train: bool = True,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.features_dir = Path(features_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_train = augment_train
        self.pin_memory = pin_memory
        self.train_ds: SignClipDataset | None = None
        self.val_ds: SignClipDataset | None = None
        self.test_ds: SignClipDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        clip_dir = self.features_dir / "clips"
        if stage in (None, "fit"):
            self.train_ds = SignClipDataset(
                self.features_dir / "train.json",
                clip_dir,
                augment=self.augment_train,
            )
            self.val_ds = SignClipDataset(
                self.features_dir / "val.json",
                clip_dir,
                augment=False,
            )
        if stage in (None, "test"):
            self.test_ds = SignClipDataset(
                self.features_dir / "test.json",
                clip_dir,
                augment=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )