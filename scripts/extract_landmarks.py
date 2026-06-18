from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from signlang.config import load_config, to_dict
from signlang.data.landmarks.holistic import (
    extract_clip_dict,
    write_landmarks_npz,
    write_manifest,
)
from signlang.utils.io import read_json
from signlang.utils.logging import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmarks")
    parser.add_argument("--interim-dir", required=True, help="Directory with preprocessed .mp4")
    parser.add_argument("--out-dir", required=True, help="Output directory for .npz clips")
    parser.add_argument("--annotations", default=None, help="Optional annotation JSON")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Override max_videos from config (positive int, or 0 to disable cap).",
    )
    args = parser.parse_args()

    configure_logging("INFO", json_logs=False)
    log = get_logger("landmarks")

    cfg = to_dict(load_config())
    target_fps = int(cfg["features"]["preprocess"]["target_fps"])
    int(cfg["features"]["mediapipe"]["model_complexity"])

    cfg_max_videos = cfg["features"]["preprocess"].get("max_videos", None)
    if args.max_videos is not None:
        max_videos: int | None = args.max_videos if args.max_videos > 0 else None
    else:
        max_videos = cfg_max_videos

    interim = Path(args.interim_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    annotations = read_json(args.annotations) if args.annotations else None

    paths = sorted(interim.rglob("*.mp4"))
    if max_videos is not None:
        paths = paths[:max_videos]
    log.info(
        "landmarks.start",
        count=len(paths),
        src=str(interim),
        dst=str(out_dir),
        max_videos=max_videos,
    )
    for i, p in enumerate(paths):
        rel = p.relative_to(interim).with_suffix(".npz")
        out_path = out_dir / rel
        if out_path.exists():
            label = 0
            if annotations:
                for k, v in annotations.items():
                    if k in str(p):
                        label = int(v)
                        break
            records.append({"clip": str(rel).replace("\\", "/"), "label": label, "video": str(p.name)})
            continue
        try:
            clip = extract_clip_dict(p, target_fps=target_fps)
            clip["pose"] = clip["pose"].astype(np.float32)
            clip["lh"] = clip["lh"].astype(np.float32)
            clip["rh"] = clip["rh"].astype(np.float32)
            write_landmarks_npz(out_path, clip)
            label = 0
            if annotations:
                for k, v in annotations.items():
                    if k in str(p):
                        label = int(v)
                        break
            records.append({"clip": str(rel).replace("\\", "/"), "label": label, "video": str(p.name)})
        except Exception as exc:
            log.warning("landmarks.skip", path=str(p), error=str(exc))
        if (i + 1) % 100 == 0:
            log.info("landmarks.progress", done=i + 1, total=len(paths))

    write_manifest(records, out_dir.parent / "manifest.jsonl")
    log.info("landmarks.done", clips=len(records), dst=str(out_dir))


if __name__ == "__main__":
    main()
