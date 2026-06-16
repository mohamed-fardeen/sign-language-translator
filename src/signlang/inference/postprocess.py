from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def collapse_ctc(
    indices: Iterable[int],
    blank: int = 0,
) -> list[int]:
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