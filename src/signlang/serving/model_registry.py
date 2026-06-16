from __future__ import annotations

from pathlib import Path

import numpy as np

from signlang.inference.predictor import TorchScriptPredictor
from signlang.utils.logging import get_logger

log = get_logger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, TorchScriptPredictor] = {}
        self._active: str | None = None

    def load(self, name: str, version: str, path: str | Path, device: str = "cpu") -> None:
        key = f"{name}@{version}"
        if key in self._models:
            return
        log.info("model.load", key=key, path=str(path), device=device)
        self._models[key] = TorchScriptPredictor(path, device=device)
        if self._active is None:
            self._active = key

    def set_active(self, key: str) -> None:
        if key not in self._models:
            raise KeyError(f"Unknown model: {key}")
        self._active = key

    @property
    def active(self) -> TorchScriptPredictor:
        if self._active is None:
            raise RuntimeError("No model loaded")
        return self._models[self._active]

    @property
    def active_key(self) -> str:
        if self._active is None:
            raise RuntimeError("No model loaded")
        return self._active

    def list(self) -> list[str]:
        return list(self._models.keys())


def pad_or_truncate(arr: np.ndarray, t: int) -> np.ndarray:
    n = arr.shape[0]
    if n == t:
        return arr
    if n > t:
        start = (n - t) // 2
        return arr[start : start + t]
    pad_shape = (t - n, *arr.shape[1:])
    pad = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)
