from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from signlang.config import load_config, to_dict
from signlang.data.datasets.clip_dataset import make_manifest, stratified_split
from signlang.utils.io import write_json
from signlang.utils.logging import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vocab + train/val/test splits")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl from extract_landmarks")
    parser.add_argument("--out-dir", required=True, help="Output features directory")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    configure_logging("INFO", json_logs=False)
    log = get_logger("vocab")

    cfg = to_dict(load_config())
    vocab_size = args.vocab_size or int(cfg.get("vocab_size", 500))
    clip_frames = int(cfg["features"]["preprocess"]["clip_frames"])

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clips").mkdir(exist_ok=True)

    records: list[dict] = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    log.info("vocab.records", count=len(records))

    label_counter = Counter(int(r["label"]) for r in records)
    keep_labels = {label for label, _ in label_counter.most_common(vocab_size)}
    filtered = [r for r in records if int(r["label"]) in keep_labels]

    label_to_id = {label: i + 1 for i, label in enumerate(sorted(keep_labels))}
    for r in filtered:
        r["label"] = label_to_id[int(r["label"])]
    vocab = {label: i + 1 for i, label in enumerate(sorted(keep_labels))}
    {i + 1: label for label, i in vocab.items()}

    write_json(
        {
            "vocab_size": len(vocab),
            "blank_id": 0,
            "id_to_gloss": {str(i): l for l, i in vocab.items()},
            "gloss_to_id": {l: i for l, i in vocab.items()},
        },
        out_dir.parent.parent / "vocab" / "vocab.json",
    )

    train, val, test = stratified_split(
        filtered, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=42
    )
    make_manifest(train, out_dir / "train.json", clip_frames=clip_frames)
    make_manifest(val, out_dir / "val.json", clip_frames=clip_frames)
    make_manifest(test, out_dir / "test.json", clip_frames=clip_frames)
    log.info(
        "vocab.done",
        vocab=len(vocab),
        train=len(train),
        val=len(val),
        test=len(test),
    )


if __name__ == "__main__":
    main()
