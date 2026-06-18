from __future__ import annotations

import numpy as np


def _rotate_xy(arr: np.ndarray, angle_rad: float) -> np.ndarray:
    if arr.size == 0:
        return arr
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    out = arr.copy()
    flat = out.reshape(-1, 3)
    x = flat[:, 0].copy()
    y = flat[:, 1].copy()
    flat[:, 0] = c * x - s * y
    flat[:, 1] = s * x + c * y
    return out


def _temporal_stretch(arr: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3 or arr.shape[0] == 0:
        return arr
    t = arr.shape[0]
    new_t = max(1, round(t * factor))
    idx = np.linspace(0, t - 1, new_t).astype(np.int64)
    return arr[idx]


def _frame_dropout(arr: np.ndarray, mask: np.ndarray, p: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    if arr.shape[0] == 0:
        return arr, mask
    keep = np.random.rand(arr.shape[0]) > p
    if not keep.any():
        keep[0] = True
    out = arr[keep]
    out_mask = mask[keep]
    return out, out_mask


def _jitter(arr: np.ndarray, sigma: float = 0.01, z_sigma: float = 0.005) -> np.ndarray:
    if arr.size == 0:
        return arr
    noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    noise[..., 2::3] = np.random.normal(0.0, z_sigma, size=(*arr.shape[:-1], 1)).astype(np.float32)
    out = arr + noise
    out[..., 0] = np.clip(out[..., 0], 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1], 0.0, 1.0)
    return out


def _swap_hands(_pose: np.ndarray, lh: np.ndarray, rh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _pose, rh, lh


def _zero_hand(hand: np.ndarray) -> np.ndarray:
    return np.zeros_like(hand)


def augment_clip(
    pose: np.ndarray,
    lh: np.ndarray,
    rh: np.ndarray,
    mask: np.ndarray,
    rotation_deg: float = 15.0,
    temporal_factors: tuple[float, float] = (0.85, 1.15),
    dropout_p: float = 0.1,
    jitter_sigma: float = 0.01,
    swap_hands_p: float = 0.0,
    zero_hand_p: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    pose = _rotate_xy(pose, np.deg2rad(rng.uniform(-rotation_deg, rotation_deg)))
    lh = _rotate_xy(lh, np.deg2rad(rng.uniform(-rotation_deg, rotation_deg)))
    rh = _rotate_xy(rh, np.deg2rad(rng.uniform(-rotation_deg, rotation_deg)))

    factor = float(rng.uniform(*temporal_factors))
    pose = _temporal_stretch(pose, factor)
    lh = _temporal_stretch(lh, factor)
    rh = _temporal_stretch(rh, factor)
    mask = _temporal_stretch(mask.reshape(-1, 1).astype(np.float32), factor).reshape(-1).astype(bool)

    pose, mask = _frame_dropout(pose, mask, p=dropout_p)
    lh = lh[: pose.shape[0]]
    rh = rh[: pose.shape[0]]
    mask = mask[: pose.shape[0]]

    pose = _jitter(pose, sigma=jitter_sigma)
    lh = _jitter(lh, sigma=jitter_sigma)
    rh = _jitter(rh, sigma=jitter_sigma)

    if swap_hands_p > 0 and rng.random() < swap_hands_p:
        pose, lh, rh = _swap_hands(pose, lh, rh)
    if zero_hand_p > 0:
        if rng.random() < zero_hand_p:
            lh = _zero_hand(lh)
        if rng.random() < zero_hand_p:
            rh = _zero_hand(rh)

    return pose.astype(np.float32), lh.astype(np.float32), rh.astype(np.float32), mask
