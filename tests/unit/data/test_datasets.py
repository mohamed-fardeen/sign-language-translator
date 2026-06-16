from __future__ import annotations

import numpy as np

from signlang.data.datasets.clip_dataset import SignClipDataset, make_manifest, stratified_split


def test_make_manifest_round_trip(tmp_path) -> None:
    records = [{"clip": "a.npz", "label": 1, "video": "a.mp4"}]
    p = tmp_path / "m.json"
    make_manifest(records, p, clip_frames=64)
    ds = SignClipDataset(p, tmp_path, clip_frames=64)
    assert len(ds) == 1
    assert ds.clip_frames == 64


def test_stratified_split_preserves_labels() -> None:
    records = [{"label": i % 3, "clip": f"c_{i}.npz"} for i in range(30)]
    train, val, test = stratified_split(records, val_ratio=0.2, test_ratio=0.2, seed=42)
    assert len(train) + len(val) + len(test) == 30
    assert set(r["label"] for r in train) == {0, 1, 2}


def test_clip_pad_truncate() -> None:
    rng = np.random.default_rng(0)
    arr = rng.random((100, 10), dtype=np.float32)
    ds_records = [{"clip": "x.npz", "label": 0, "video": "x.mp4"}]
    from pathlib import Path

    p = Path("/tmp/m.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    make_manifest(ds_records, p, clip_frames=64)
    ds = SignClipDataset(p, Path("/tmp"), clip_frames=64)
    out = ds._pad_or_truncate(arr, ds.clip_frames)
    assert out.shape == (64, 10)
