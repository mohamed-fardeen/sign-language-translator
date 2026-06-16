from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from signlang.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LandmarkLayout:
    pose_dim: int = 33 * 3
    hand_dim: int = 21 * 3
    face_dim: int = 40 * 3
    face_indices: tuple[int, ...] = (
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    )


LAYOUT = LandmarkLayout()


def _extract_from_frame(holistic, frame_bgr: np.ndarray) -> dict[str, np.ndarray]:

    frame_rgb = frame_bgr[:, :, ::-1]
    res = holistic.process(frame_rgb)
    out: dict[str, np.ndarray] = {}

    pose = np.zeros(LAYOUT.pose_dim, dtype=np.float32)
    if res.pose_landmarks:
        for i, lm in enumerate(res.pose_landmarks.landmark):
            if i * 3 + 2 < LAYOUT.pose_dim:
                pose[i * 3 + 0] = lm.x
                pose[i * 3 + 1] = lm.y
                pose[i * 3 + 2] = lm.z
    out["pose"] = pose

    lh = np.zeros(LAYOUT.hand_dim, dtype=np.float32)
    if res.left_hand_landmarks:
        for i, lm in enumerate(res.left_hand_landmarks.landmark):
            lh[i * 3 + 0] = lm.x
            lh[i * 3 + 1] = lm.y
            lh[i * 3 + 2] = lm.z
    out["lh"] = lh

    rh = np.zeros(LAYOUT.hand_dim, dtype=np.float32)
    if res.right_hand_landmarks:
        for i, lm in enumerate(res.right_hand_landmarks.landmark):
            rh[i * 3 + 0] = lm.x
            rh[i * 3 + 1] = lm.y
            rh[i * 3 + 2] = lm.z
    out["rh"] = rh

    face = np.zeros(LAYOUT.face_dim, dtype=np.float32)
    if res.face_landmarks:
        for out_i, src_i in enumerate(LAYOUT.face_indices):
            if src_i < len(res.face_landmarks.landmark):
                lm = res.face_landmarks.landmark[src_i]
                face[out_i * 3 + 0] = lm.x
                face[out_i * 3 + 1] = lm.y
                face[out_i * 3 + 2] = lm.z
    out["face"] = face
    return out


def extract_video_landmarks(
    video_path: str | Path,
    target_fps: int = 30,
    model_complexity: int = 1,
) -> np.ndarray:
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    if src_fps <= 0:
        src_fps = target_fps
    step = max(1, round(src_fps / target_fps))
    total_dim = LAYOUT.pose_dim + 2 * LAYOUT.hand_dim + LAYOUT.face_dim
    frames: list[np.ndarray] = []

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
    ) as holistic:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                feats = _extract_from_frame(holistic, frame)
                frames.append(
                    np.concatenate(
                        [feats["pose"], feats["lh"], feats["rh"], feats["face"]],
                        axis=0,
                    )
                )
            idx += 1
    cap.release()
    if not frames:
        return np.zeros((0, total_dim), dtype=np.float32)
    return np.stack(frames, axis=0).astype(np.float32)


def extract_clip_dict(
    video_path: str | Path,
    target_fps: int = 30,
) -> dict[str, np.ndarray]:
    arr = extract_video_landmarks(video_path, target_fps=target_fps)
    n = arr.shape[0]
    return {
        "pose": arr[:, : LAYOUT.pose_dim],
        "lh": arr[:, LAYOUT.pose_dim : LAYOUT.pose_dim + LAYOUT.hand_dim],
        "rh": arr[
            :,
            LAYOUT.pose_dim + LAYOUT.hand_dim : LAYOUT.pose_dim + 2 * LAYOUT.hand_dim,
        ],
        "face": arr[:, LAYOUT.pose_dim + 2 * LAYOUT.hand_dim :],
        "mask": np.ones(n, dtype=bool),
    }


def iter_clip_dicts(
    video_paths: Iterator[Path],
    target_fps: int = 30,
) -> Iterator[dict[str, np.ndarray]]:
    for p in video_paths:
        try:
            yield extract_clip_dict(p, target_fps=target_fps)
        except Exception as exc:
            log.warning("landmark.skip", path=str(p), error=str(exc))


def write_landmarks_npz(out_path: Path, clip: dict[str, np.ndarray]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **clip)


def read_landmarks_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def write_manifest(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")