from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Classify a single clip. Returns a dict with:

        - ``label``: int, 0-indexed class index (add 1 to get the
          manifest label)
        - ``probs``: ndarray of shape ``(num_classes,)``
        - ``top_k``: list of ``(class_idx, prob)`` tuples
        """
        pose = torch.from_numpy(clip["pose"]).unsqueeze(0).to(self.device).float()
        lh = torch.from_numpy(clip["lh"]).unsqueeze(0).to(self.device).float()
        rh = torch.from_numpy(clip["rh"]).unsqueeze(0).to(self.device).float()
        out = self.model(pose, lh, rh)
        if isinstance(out, tuple):
            logits = out[0]
        elif isinstance(out, dict):
            logits = out["logits"]
        else:
            logits = out

        # CTC kept for reference (removed in v1 single-label mode):
        # if beam_size <= 1:
        #     return greedy_decode(logits, blank=blank)[0]
        # log_probs = torch.log_softmax(logits, dim=-1)[0].cpu().numpy()
        # return beam_search_decode(log_probs, beam_size=beam_size, blank=blank)

        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        k = min(top_k, probs.shape[0])
        top_idx = probs.argsort()[::-1][:k]
        return {
            "label": int(probs.argmax()),
            "probs": probs,
            "top_k": [(int(i), float(probs[i])) for i in top_idx],
        }


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

    async def submit(self, clip: dict[str, np.ndarray]) -> dict[str, Any]:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put((clip, fut))
        return await fut

    async def _loop(self) -> None:
        while True:
            clip, fut = await self._queue.get()
            try:
                preds = await asyncio.get_event_loop().run_in_executor(
                    None, self.predictor.predict, clip, 5
                )
                fut.set_result(preds)
            except Exception as exc:
                fut.set_exception(exc)


# Local import for the AsyncBatcher; placed at the bottom so the rest of the
# module's public surface stays small.
import contextlib  # noqa: E402