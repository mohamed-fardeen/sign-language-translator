from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import numpy as np
import torch

from signlang.inference.postprocess import beam_search_decode, greedy_decode
from signlang.utils.logging import get_logger

log = get_logger(__name__)


class TorchScriptPredictor:
    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self.model = torch.jit.load(str(model_path), map_location=device)
        self.model.eval()
        self.device = torch.device(device)

    @torch.inference_mode()
    def predict(
        self,
        clip: dict[str, np.ndarray],
        beam_size: int = 1,
        blank: int = 0,
    ) -> list[int]:
        pose = torch.from_numpy(clip["pose"]).unsqueeze(0).to(self.device).float()
        lh = torch.from_numpy(clip["lh"]).unsqueeze(0).to(self.device).float()
        rh = torch.from_numpy(clip["rh"]).unsqueeze(0).to(self.device).float()
        face = torch.from_numpy(clip["face"]).unsqueeze(0).to(self.device).float()
        out = self.model(pose, lh, rh, face)
        if isinstance(out, tuple):
            logits = out[0]
        elif isinstance(out, dict):
            logits = out["logits"]
        else:
            logits = out
        if beam_size <= 1:
            return greedy_decode(logits, blank=blank)[0]
        log_probs = torch.log_softmax(logits, dim=-1)[0].cpu().numpy()
        return beam_search_decode(log_probs, beam_size=beam_size, blank=blank)


class AsyncBatcher:
    def __init__(self, predictor: TorchScriptPredictor, window_ms: int = 30, max_batch: int = 32) -> None:
        self.predictor = predictor
        self.window_ms = window_ms
        self.max_batch = max_batch
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def submit(self, clip: dict[str, np.ndarray]) -> list[int]:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put((clip, fut))
        return await fut

    async def _loop(self) -> None:
        while True:
            clip, fut = await self._queue.get()
            try:
                preds = await asyncio.get_event_loop().run_in_executor(
                    None, self.predictor.predict, clip, 1, 0
                )
                fut.set_result(preds)
            except Exception as exc:
                fut.set_exception(exc)