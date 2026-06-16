from __future__ import annotations

from pathlib import Path

from signlang.evaluation.metrics import class_distribution
from signlang.utils.io import write_json


def classification_report(
    preds: list[int],
    targets: list[int],
    label_names: dict[int, str] | None = None,
) -> dict:
    correct = sum(1 for p, t in zip(preds, targets, strict=False) if p == t)
    total = max(1, len(targets))
    per_class: dict[int, dict[str, int]] = {}
    for p, t in zip(preds, targets, strict=False):
        per_class.setdefault(t, {"tp": 0, "fn": 0, "fp": 0})
        per_class[t]["tp" if p == t else "fn"] += 1
        if p != t:
            per_class.setdefault(p, {"tp": 0, "fn": 0, "fp": 0})
            per_class[p]["fp"] += 1
    rows = []
    for label, counts in sorted(per_class.items()):
        tp = counts["tp"]
        fn = counts["fn"]
        fp = counts["fp"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append(
            {
                "label": label,
                "name": (label_names or {}).get(label, str(label)),
                "support": tp + fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "accuracy": correct / total,
        "n": total,
        "class_distribution": class_distribution(targets),
        "per_class": rows,
    }


def write_report(report: dict, path: str | Path) -> None:
    write_json(report, path)