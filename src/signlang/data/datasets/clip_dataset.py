from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from signlang.data.augmentation.landmark_aug import augment_clip


@dataclass
class ClipSample:
    pose: np.ndarray
    lh: np.ndarray
    rh: np.ndarray
    face: np.ndarray
    mask: np.ndarray
    label: int


class SignClipDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        clip_dir: str | Path,
        clip_frames: int = 64,
        augment: bool = False,
        swap_hands_p: float = 0.0,
        zero_hand_p: float = 0.0,
        rotation_deg: float = 15.0,
    ) -> None:
        from signlang.utils.io import read_json  # local to avoid heavy import on cold path

        meta = read_json(manifest_path)
        self.records: list[dict] = meta["records"]
        self.clip_frames = int(meta.get("clip_frames", clip_frames))
        self.clip_dir = Path(clip_dir)
        self.augment = augment
        self.swap_hands_p = swap_hands_p
        self.zero_hand_p = zero_hand_p
        self.rotation_deg = rotation_deg
        self._cache: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, idx: int) -> dict[str, np.ndarray]:
        if idx in self._cache:
            return self._cache[idx]
        rec = self.records[idx]
        path = self.clip_dir / rec["clip"]
        with np.load(path) as data:
            clip = {k: data[k] for k in data.files}
        self._cache[idx] = clip
        return clip

    def _pad_or_truncate(self, arr: np.ndarray, t: int) -> np.ndarray:
        n = arr.shape[0]
        if n == t:
            return arr
        if n > t:
            start = (n - t) // 2
            return arr[start : start + t]
        pad = np.zeros((t - n, *arr.shape[1:]), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        clip = self._load(idx)
        pose = clip["pose"].astype(np.float32)
        lh = clip["lh"].astype(np.float32)
        rh = clip["rh"].astype(np.float32)
        face = clip["face"].astype(np.float32)
        mask = clip.get("mask", np.ones(pose.shape[0], dtype=bool)).astype(bool)

        if self.augment:
            pose, lh, rh, face, mask = augment_clip(
                pose,
                lh,
                rh,
                face,
                mask,
                swap_hands_p=self.swap_hands_p,
                zero_hand_p=self.zero_hand_p,
                rotation_deg=self.rotation_deg,
            )

        pose = self._pad_or_truncate(pose, self.clip_frames)
        lh = self._pad_or_truncate(lh, self.clip_frames)
        rh = self._pad_or_truncate(rh, self.clip_frames)
        face = self._pad_or_truncate(face, self.clip_frames)
        if mask.shape[0] != self.clip_frames:
            m = np.zeros(self.clip_frames, dtype=bool)
            m[: min(mask.shape[0], self.clip_frames)] = mask[: self.clip_frames]
            mask = m

        rec = self.records[idx]
        return {
            "pose": torch.from_numpy(pose),
            "lh": torch.from_numpy(lh),
            "rh": torch.from_numpy(rh),
            "face": torch.from_numpy(face),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
        }


def make_manifest(
    records: list[dict],
    out_path: str | Path,
    clip_frames: int = 64,
) -> None:
    from signlang.utils.io import write_json

    write_json(
        {"clip_frames": clip_frames, "records": records},
        out_path,
    )


def stratified_split(
    records: list[dict],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    import random

    by_label: dict[int, list[dict]] = {}
    for r in records:
        by_label.setdefault(int(r["label"]), []).append(r)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for items in by_label.values():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
    return train, val, test