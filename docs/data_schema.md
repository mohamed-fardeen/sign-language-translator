# Data schema

## Datasets
- **WLASL** -- https://www.kaggle.com/datasets/risangbudi07/wlasl-processed
- **MS-ASL** -- https://www.kaggle.com/datasets/soongpal/ms-asl
- **ASL Citizen** -- https://www.kaggle.com/datasets/grassknoted/asl-citizen

We filter each dataset to the locked 500-gloss vocabulary and use the
intersection of frequent glosses across all three sources.

## Per-frame features (MediaPipe Holistic, MVP)
| Stream    | Layout                | Dims  |
|-----------|-----------------------|-------|
| Pose      | 33 keypoints (x,y,z)  | 99    |
| Left hand | 21 keypoints          | 63    |
| Right hand| 21 keypoints          | 63    |
| **Total** |                       | **225** |

**MVP scope**: face landmarks are not extracted. The face branch of
MediaPipe Holistic is disabled in `configs/features/mediapipe_holistic.yaml`
(`extract_face: false`). This is a deliberate scope reduction for fast
MVP iteration; see `ARCHITECTURE.md` for context. Re-enabling face
extraction would require changes to the model, the dataset, and the
browser app.

## Clip layout
- Variable raw length, **padded/truncated to T=64 frames** at 30 fps.
- Stored per clip as a single `.npz` containing `pose`, `lh`, `rh`,
  `mask` arrays (no face).

## Manifests
Each split (`train.json`, `val.json`, `test.json`) is a JSON file:

```json
{
  "clip_frames": 64,
  "records": [
    { "clip": "clip_0001.npz", "label": 17, "video": "v_0001.mp4" }
  ]
}
```

`label` is the 1-based gloss id (0 is reserved for the CTC blank).

## Vocab
`data/vocab/vocab.json`:

```json
{
  "vocab_size": 500,
  "blank_id": 0,
  "id_to_gloss": { "1": "hello", "2": "thanks", ... },
  "gloss_to_id": { "hello": 1, "thanks": 2, ... }
}
```

## DVC pipeline
See `dvc.yaml`. Stages: `preprocess -> landmarks -> features ->
train -> evaluate`. Re-run with `dvc repro`.

## MVP video cap

`configs/features/mediapipe_holistic.yaml` ships with
`preprocess.max_videos: 3000`. The extractor processes at most the
first N videos (sorted by path) and skips the rest. To run on the
full dataset, set `max_videos: null` (or remove the key) in that file
and re-run extraction. Existing `.npz` files are skipped on restart
(resumable).
