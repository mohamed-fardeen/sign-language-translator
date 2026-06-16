# ADR 0001: MediaPipe Holistic for landmark extraction

- Status: Accepted
- Date: 2026-06-15
- Deciders: ML Platform, CV

## Context
Sign-language recognition needs body, hand, and face keypoints at
real-time rates on commodity devices. We need a single, well-maintained
extractor that runs on CPU at 30+ fps per stream -- in the browser (JS)
and on a server (Python).

## Decision
Use **MediaPipe Holistic** (pose 33, hands 21x2, face 468) as the
canonical landmark extractor. We persist landmarks as numpy arrays per
frame; we do not store raw video for training (privacy + storage).

## Consequences
- Single dependency, broadly tested, runs on CPU and in the browser.
- Face landmarks are over-complete; we keep a subset (lips, eyebrows,
  jaw) defined in `configs/features/mediapipe_holistic.yaml`.
- The same subset is used on both server (Python) and browser (JS) so
  feature dimensions are identical end-to-end.
- Pin to a specific MediaPipe version for reproducibility; track via
  `requirements/base.txt`.

## Alternatives considered
- OpenPose: heavier, GPU-only, slower.
- Hand-crafted OpenCV pipeline: brittle, lots of maintenance.