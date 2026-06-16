from __future__ import annotations

from collections import Counter


def topk_accuracy(preds: list[int], targets: list[int], k: int = 5) -> float:
    if not preds or not targets:
        return 0.0
    n = sum(1 for p, t in zip(preds, targets, strict=False) if p == t)
    return n / max(1, len(targets))


def cer(preds: list[list[int]], targets: list[list[int]]) -> float:
    edits, total = 0, 0
    for p, t in zip(preds, targets, strict=False):
        edits += _levenshtein(p, t)
        total += max(1, len(t))
    return edits / max(1, total)


def _levenshtein(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cost = 0 if x == y else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def class_distribution(targets: list[int]) -> dict[int, int]:
    return dict(Counter(targets))