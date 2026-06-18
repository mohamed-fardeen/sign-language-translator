from __future__ import annotations

import numpy as np

from signlang.data.datamodules import SignDataModule


def test_datamodule_constructs(tmp_path) -> None:
    features_dir = tmp_path / "features"
    (features_dir / "clips").mkdir(parents=True)

    import json

    T = 64
    for i in range(4):
        np.savez_compressed(
            features_dir / "clips" / f"clip_{i:02d}.npz",
            pose=np.zeros((T, 99), dtype=np.float32),
            lh=np.zeros((T, 63), dtype=np.float32),
            rh=np.zeros((T, 63), dtype=np.float32),
            mask=np.ones(T, dtype=bool),
        )

    recs = [{"clip": f"clip_{i:02d}.npz", "label": i % 2, "video": f"v_{i:02d}.mp4"} for i in range(4)]
    (features_dir / "train.json").write_text(
        json.dumps({"clip_frames": T, "records": recs})
    )
    (features_dir / "val.json").write_text(
        json.dumps({"clip_frames": T, "records": recs[:2]})
    )
    (features_dir / "test.json").write_text(
        json.dumps({"clip_frames": T, "records": recs[2:]})
    )

    dm = SignDataModule(features_dir=features_dir, batch_size=2, num_workers=0)
    dm.setup("fit")
    assert dm.train_ds is not None and dm.val_ds is not None
    assert len(dm.train_ds) == 4
    assert len(dm.val_ds) == 2
