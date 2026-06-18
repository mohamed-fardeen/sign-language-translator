from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from signlang.config import load_config, to_dict
from signlang.data.ingestion.annotations import (
    build_gloss_to_id,
    load_wlasl_annotations,
)
from signlang.data.landmarks.holistic import process_video
from signlang.utils.logging import configure_logging, get_logger

PROGRESS_EVERY_N = 50
PROGRESS_EVERY_S = 10.0


def _default_workers() -> int:
    n = os.cpu_count() or 1
    return min(6, max(n - 1, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmarks")
    parser.add_argument("--interim-dir", required=True, help="Directory with preprocessed .mp4")
    parser.add_argument("--out-dir", required=True, help="Output directory for .npz clips")
    parser.add_argument(
        "--annotations",
        required=True,
        help=(
            "Path to the dataset annotations file. For WLASL, point this "
            "to data/raw/wlasl/WLASL_v0.3.json."
        ),
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Override max_videos from config (positive int, or 0 to disable cap).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes. Default: "
            "min(6, max(os.cpu_count() - 1, 1))."
        ),
    )
    args = parser.parse_args()

    configure_logging("INFO", json_logs=False)
    log = get_logger("landmarks")

    cfg = to_dict(load_config())
    target_fps = int(cfg["features"]["preprocess"]["target_fps"])
    model_complexity = int(cfg["features"]["mediapipe"]["model_complexity"])

    cfg_max_videos = cfg["features"]["preprocess"].get("max_videos", None)
    if args.max_videos is not None:
        max_videos: int | None = args.max_videos if args.max_videos > 0 else None
    else:
        max_videos = cfg_max_videos

    workers = args.workers if args.workers is not None else _default_workers()
    workers = max(1, workers)

    try:
        video_to_gloss = load_wlasl_annotations(args.annotations)
    except (FileNotFoundError, ValueError) as exc:
        log.error(
            "annotations.load_failed",
            path=args.annotations,
            error=str(exc),
        )
        sys.exit(1)
    gloss_to_id = build_gloss_to_id(video_to_gloss.values())
    log.info(
        "annotations.loaded",
        path=args.annotations,
        videos_in_annotations=len(video_to_gloss),
        unique_glosses=len(gloss_to_id),
    )

    interim = Path(args.interim_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(interim.rglob("*.mp4"))
    if max_videos is not None:
        paths = paths[:max_videos]

    records: list[dict] = []
    pending: list[tuple[Path, Path, Path, int, str]] = []
    skipped_no_annotation = 0

    for video in paths:
        rel = video.relative_to(interim).with_suffix(".npz")
        out_path = out_dir / rel
        gloss = video_to_gloss.get(video.stem)
        if gloss is None:
            skipped_no_annotation += 1
            log.warning(
                "landmarks.skip",
                reason="no annotation",
                path=str(video),
                video_id=video.stem,
            )
            continue
        label = gloss_to_id[gloss]
        record = {
            "clip": str(rel).replace("\\", "/"),
            "label": label,
            "gloss": gloss,
            "video": video.name,
        }
        if out_path.exists():
            records.append(record)
            continue
        pending.append((video, out_path, rel, label, gloss))

    total = len(paths)
    skipped_existing = total - len(pending) - skipped_no_annotation
    log.info(
        "landmarks.start",
        count=total,
        pending=len(pending),
        skipped_existing=skipped_existing,
        skipped_no_annotation=skipped_no_annotation,
        workers=workers,
        src=str(interim),
        dst=str(out_dir),
        max_videos=max_videos,
    )

    completed = 0
    failed = 0
    pending_records: list[dict] = []
    start = time.perf_counter()
    last_log = start

    if pending:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    process_video,
                    str(video),
                    str(out_path),
                    target_fps,
                    model_complexity,
                ): (video, out_path, rel, label, gloss)
                for video, out_path, rel, label, gloss in pending
            }
            for fut in as_completed(futures):
                video, out_path, rel, label, gloss = futures[fut]
                try:
                    success, err = fut.result()
                except Exception as exc:
                    success, err = False, str(exc)
                if success:
                    pending_records.append(
                        {
                            "clip": str(rel).replace("\\", "/"),
                            "label": label,
                            "gloss": gloss,
                            "video": video.name,
                        }
                    )
                else:
                    failed += 1
                    log.warning(
                        "landmarks.skip",
                        reason="extraction failed",
                        path=str(video),
                        out=str(out_path),
                        error=err or "unknown error",
                    )
                completed += 1
                now = time.perf_counter()
                if completed % PROGRESS_EVERY_N == 0 or (now - last_log) >= PROGRESS_EVERY_S:
                    elapsed = max(now - start, 1e-6)
                    rate = (completed / elapsed) * 60.0
                    remaining = max(len(pending) - completed, 0)
                    eta_s = (remaining / rate) * 60.0 if rate > 0 else 0.0
                    log.info(
                        "landmarks.progress",
                        done=completed,
                        total=len(pending),
                        rate_per_min=round(rate, 2),
                        eta_seconds=round(eta_s, 1),
                    )
                    last_log = now

    records.extend(pending_records)
    records.sort(key=lambda r: r["clip"])
    write_manifest(records, out_dir.parent / "manifest.jsonl")
    elapsed = time.perf_counter() - start

    unique_labels = len({r["label"] for r in records})
    unique_glosses = len({r["gloss"] for r in records})
    skip_pct = (100.0 * skipped_no_annotation / total) if total else 0.0

    if unique_labels < 2:
        log.warning(
            "landmarks.low_label_diversity",
            unique_labels=unique_labels,
            unique_glosses=unique_glosses,
            clips=len(records),
            hint="check that --annotations maps videos to many distinct glosses",
        )
    if skip_pct > 10.0:
        log.warning(
            "landmarks.high_skip_pct",
            skip_pct=round(skip_pct, 2),
            skipped_no_annotation=skipped_no_annotation,
            total=total,
        )
    log.info(
        "landmarks.done",
        videos_total=total,
        videos_extracted=len(pending_records),
        videos_skipped_existing=skipped_existing,
        videos_skipped_no_annotation=skipped_no_annotation,
        skip_pct=round(skip_pct, 2),
        unique_glosses=unique_glosses,
        unique_labels=unique_labels,
        videos_failed=failed,
        elapsed_seconds=round(elapsed, 2),
        workers=workers,
        dst=str(out_dir),
    )


def write_manifest(records: list[dict], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
