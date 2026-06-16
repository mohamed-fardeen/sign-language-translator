from __future__ import annotations

import numpy as np

from signlang.inference.postprocess import beam_search_decode, collapse_ctc, greedy_decode


def test_collapse_ctc_basic() -> None:
    out = collapse_ctc([0, 1, 1, 0, 2, 2, 2, 0, 3], blank=0)
    assert out == [1, 2, 3]


def test_greedy_decode_returns_per_sample() -> None:
    logits = np.zeros((1, 8, 5), dtype=np.float32)
    logits[0, :, 2] = 10.0
    decoded = greedy_decode(logits, blank=0)
    assert decoded[0] == [2]


def test_beam_search_decode_finds_target() -> None:
    rng = np.random.default_rng(0)
    T, V = 16, 8
    log_probs = rng.normal(0, 1, (T, V)).astype(np.float32)
    for t in range(T):
        log_probs[t, 3] += 5.0
    out = beam_search_decode(log_probs, beam_size=4, blank=0)
    assert 3 in out
