from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from signlang.config import load_config, to_dict
from signlang.data.datasets.clip_dataset import make_manifest, stratified_split
from signlang.data.ingestion.annotations import build_gloss_to_id, build_id_to_gloss
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

    if not records:
        log.error("vocab.empty_manifest", path=str(manifest))
        sys.exit(1)

    gloss_counter = Counter(str(r["gloss"]) for r in records)
    kept_glosses = {g for g, _ in gloss_counter.most_common(vocab_size)}
    filtered = [r for r in records if str(r["gloss"]) in kept_glosses]

    gloss_to_id = build_gloss_to_id(kept_glosses)
    id_to_gloss = build_id_to_gloss(gloss_to_id)
    for r in filtered:
        r["label"] = gloss_to_id[str(r["gloss"])]

    vocab_path = out_dir.parent.parent / "vocab" / "vocab.json"
    write_json(
        {
            "vocab_size": len(gloss_to_id),
            "blank_id": 0,
            "id_to_gloss": id_to_gloss,
            "gloss_to_id": gloss_to_id,
        },
        vocab_path,
    )
    log.info("vocab.written", path=str(vocab_path), vocab_size=len(gloss_to_id))

    if len(gloss_to_id) < 2:
        log.error(
            "vocab.too_small",
            vocab_size=len(gloss_to_id),
            unique_glosses_in_manifest=len(gloss_counter),
        )
        sys.exit(1)

    train, val, test = stratified_split(
        filtered, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=42
    )
    make_manifest(train, out_dir / "train.json", clip_frames=clip_frames)
    make_manifest(val, out_dir / "val.json", clip_frames=clip_frames)
    make_manifest(test, out_dir / "test.json", clip_frames=clip_frames)
    log.info(
        "vocab.done",
        vocab=len(gloss_to_id),
        train=len(train),
        val=len(val),
        test=len(test),
    )


if __name__ == "__main__":
    main()
