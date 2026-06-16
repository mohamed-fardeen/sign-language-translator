from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from signlang.utils.io import ensure_dir
from signlang.utils.logging import get_logger

log = get_logger(__name__)


def resample_video(
    src: Path,
    dst: Path,
    target_fps: int = 30,
    frame_size: int = 256,
) -> Path:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    if src_fps <= 0:
        src_fps = target_fps
    step = max(1, round(src_fps / target_fps))

    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame_size
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame_size
    out_size = (frame_size, frame_size)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), fourcc, float(target_fps), out_size)
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            resized = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
            writer.write(resized)
            kept += 1
        idx += 1
    cap.release()
    writer.release()
    log.debug("video.resample", src=str(src), dst=str(dst), kept=kept, src_fps=src_fps, target_fps=target_fps)
    return dst


def isolate_clip(
    video: Path,
    out: Path,
    start_frame: int,
    end_frame: int,
    target_fps: int = 30,
    frame_size: int = 256,
) -> Path:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, float(target_fps), (frame_size, frame_size))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    for _ in range(max(0, end_frame - start_frame)):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA))
    cap.release()
    writer.release()
    return out


def preprocess_directory(
    src_dir: Path,
    dst_dir: Path,
    target_fps: int = 30,
    frame_size: int = 256,
) -> int:
    if not src_dir.exists():
        log.warning("preprocess.skip", reason="missing", src=str(src_dir))
        return 0
    ensure_dir(dst_dir)
    count = 0
    for src in sorted(src_dir.rglob("*.mp4")):
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        if dst.exists():
            continue
        try:
            resample_video(src, dst, target_fps=target_fps, frame_size=frame_size)
            count += 1
        except Exception as exc:
            log.error("preprocess.error", src=str(src), error=str(exc))
    log.info("preprocess.summary", processed=count, src=str(src_dir), dst=str(dst_dir))
    return count


def copy_manifest(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)