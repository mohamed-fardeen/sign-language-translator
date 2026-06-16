from __future__ import annotations

import numpy as np

from signlang.data.augmentation.landmark_aug import augment_clip


def test_augment_clip_returns_correct_shapes() -> None:
    T = 32
    pose = np.random.rand(T, 99).astype(np.float32)
    lh = np.random.rand(T, 63).astype(np.float32)
    rh = np.random.rand(T, 63).astype(np.float32)
    face = np.random.rand(T, 120).astype(np.float32)
    mask = np.ones(T, dtype=bool)
    p, l, r, f, m = augment_clip(pose, lh, rh, face, mask)
    assert p.shape[0] == l.shape[0] == r.shape[0] == f.shape[0] == m.shape[0]
    assert p.shape[1] == 99 and l.shape[1] == 63 and r.shape[1] == 63 and f.shape[1] == 120
    assert m.dtype == bool


def test_augment_clip_jitter_keeps_xy_in_range() -> None:
    T = 16
    pose = np.full((T, 99), 0.5, dtype=np.float32)
    pose[..., 0] = 0.99
    pose[..., 1] = 0.01
    p, _, _, _, _ = augment_clip(pose, np.zeros((T, 63), np.float32),
                                 np.zeros((T, 63), np.float32),
                                 np.zeros((T, 120), np.float32),
                                 np.ones(T, dtype=bool))
    assert (p[..., 0] <= 1.0).all() and (p[..., 0] >= 0.0).all()
    assert (p[..., 1] <= 1.0).all() and (p[..., 1] >= 0.0).all()
