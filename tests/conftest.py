from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def small_clip() -> dict[str, np.ndarray]:
    T = 64
    rng = np.random.default_rng(0)
    return {
        "pose": rng.random((T, 99), dtype=np.float32),
        "lh": rng.random((T, 63), dtype=np.float32),
        "rh": rng.random((T, 63), dtype=np.float32),
        "face": rng.random((T, 120), dtype=np.float32),
        "mask": np.ones(T, dtype=bool),
    }


@pytest.fixture
def tiny_manifest(tmp_path: Path) -> Path:
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    n = 8
    for i in range(n):
        T = 64
        np.savez_compressed(
            clip_dir / f"clip_{i:02d}.npz",
            pose=np.zeros((T, 99), dtype=np.float32),
            lh=np.zeros((T, 63), dtype=np.float32),
            rh=np.zeros((T, 63), dtype=np.float32),
            face=np.zeros((T, 120), dtype=np.float32),
            mask=np.ones(T, dtype=bool),
        )
    manifest = tmp_path / "train.json"
    manifest.write_text(
        json.dumps(
            {
                "clip_frames": 64,
                "records": [
                    {"clip": f"clip_{i:02d}.npz", "label": i % 4, "video": f"v_{i:02d}.mp4"}
                    for i in range(n)
                ],
            }
        )
    )
    return manifest


@pytest.fixture
def fake_vocab() -> dict:
    return {
        "vocab_size": 4,
        "blank_id": 0,
        "id_to_gloss": {"1": "hello", "2": "thanks", "3": "yes", "4": "no"},
        "gloss_to_id": {"hello": 1, "thanks": 2, "yes": 3, "no": 4},
    }
