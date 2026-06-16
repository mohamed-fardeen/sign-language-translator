"""Drift detection for the signlang inference service.

Reads a JSON-lines log file of recent predictions and computes simple
drift metrics on the input feature statistics and output distribution.

Run:
    python monitoring/drift_detection.py --logs artifacts/logs/predictions.jsonl

The script is intentionally lightweight: pure stdlib + numpy. It is
meant to be invoked on a schedule (cron / Render cron / GitHub Actions).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

WINDOW = 5_000


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def feature_stats(records: list[dict]) -> dict[str, float]:
    if not records:
        return {}
    hands_detected = []
    confidences = []
    for r in records:
        feat = r.get("feature_stats", {})
        hands_detected.append(float(feat.get("hands_detected", 0)))
        confidences.append(float(r.get("confidence", 0.0)))
    return {
        "mean_hands_detected": float(np.mean(hands_detected)),
        "mean_confidence": float(np.mean(confidences)),
        "p10_confidence": float(np.percentile(confidences, 10)),
    }


def topk_drift(recent: list[dict], baseline: list[dict], k: int = 10) -> float:
    def dist(rs: list[dict]) -> Counter:
        return Counter(r.get("gloss_id", 0) for r in rs)
    a = dist(recent)
    b = dist(baseline)
    def total(c):
        return sum(c.values()) or 1
    keys = set(a) | set(b)
    return sum(abs(a.get(kk, 0) / total(a) - b.get(kk, 0) / total(b)) for kk in keys) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--out", default=None, help="Path to write drift report JSON")
    args = parser.parse_args()

    records = load_records(Path(args.logs))
    if not records:
        report = {"status": "no data"}
    else:
        recent = records[-WINDOW:]
        baseline = records[:WINDOW] if len(records) > WINDOW else records
        report = {
            "n": len(records),
            "recent_n": len(recent),
            "feature_stats": feature_stats(recent),
            "topk_drift": topk_drift(recent, baseline),
        }
    out_path = Path(args.out) if args.out else Path("monitoring/last_drift.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
