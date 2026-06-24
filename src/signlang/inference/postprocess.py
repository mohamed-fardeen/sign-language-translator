"""Post-processing helpers.

CTC decoders (``collapse_ctc``, ``greedy_decode``, ``beam_search_decode``)
are kept for reference and continue to work, but the v1 single-label
classification pipeline does not use them. New classification helpers
(``classify_argmax``, ``classify_topk``) are used by the predictor and
the FastAPI route.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

# ---------------------------------------------------------------------------
# CTC kept for reference (v1 single-label classification does not use these).
# ---------------------------------------------------------------------------


def collapse_ctc(
    indices: Iterable[int],
    blank: int = 0,
) -> list[int]:
    """Standard CTC collapse: drop blanks, deduplicate consecutive runs."""
    out: list[int] = []
    prev = -1
    for i in indices:
        i = int(i)
        if i != blank and i != prev:
            out.append(i)
        prev = i
    return out


def greedy_decode(
    logits: torch.Tensor | np.ndarray,
    blank: int = 0,
) -> list[list[int]]:
    """Greedy CTC decoder. Returns one collapsed sequence per batch row."""
    if isinstance(logits, np.ndarray):
        idx = logits.argmax(axis=-1)
    else:
        idx = logits.argmax(dim=-1).cpu().numpy()
    return [collapse_ctc(seq, blank=blank) for seq in idx]


def beam_search_decode(
    log_probs_np: np.ndarray,
    beam_size: int = 8,
    blank: int = 0,
) -> list[int]:
    """Vocabulary-constrained CTC beam search (kept for reference)."""
    T, V = log_probs_np.shape
    beam: dict[tuple[int, ...], float] = {(): 0.0}
    for t in range(T):
        new_beam: dict[tuple[int, ...], float] = {}
        for seq, score in beam.items():
            for v in range(V):
                lp = float(log_probs_np[t, v])
                if v == blank:
                    new_seq = seq
                elif not seq or seq[-1] != v:
                    new_seq = (*seq, v)
                else:
                    new_seq = seq
                new_score = score + lp
                prev = new_beam.get(new_seq, float("-inf"))
                if new_score > prev:
                    new_beam[new_seq] = new_score
        beam = dict(sorted(new_beam.items(), key=lambda kv: kv[1], reverse=True)[:beam_size])
    best_seq = max(beam.items(), key=lambda kv: kv[1])[0]
    return collapse_ctc(best_seq, blank=blank)


# ---------------------------------------------------------------------------
# Classification (v1): argmax + softmax top-k.
# ---------------------------------------------------------------------------


def classify_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Return per-batch class indices (0-indexed) from classification logits."""
    return logits.argmax(dim=-1)


def classify_topk(
    logits: torch.Tensor,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k class probabilities and indices from classification logits.

    Returns ``(topk_probs, topk_indices)`` each of shape ``(B, k)`` when
    ``logits`` is 2D, or ``(k,)`` when 1D.
    """
    probs = torch.softmax(logits, dim=-1)
    k = min(k, probs.size(-1))
    return torch.topk(probs, k=k, dim=-1)