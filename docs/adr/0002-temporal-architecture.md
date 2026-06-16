# ADR 0002: Temporal model -- Transformer encoder only (v1, locked)

- Status: Accepted (v1, frozen)
- Date: 2026-06-15
- Deciders: ML Platform

## Context
After per-frame landmark encoding, we need a temporal backbone.

## Decision
Default to **Transformer encoder (6L, d=512, 8 heads, pre-LN, GELU)**
and only the Transformer encoder. BiLSTM and ST-GCN are removed from
v1 to keep one well-tuned backbone.

## Consequences
- Transformer gives best accuracy/latency on our target hardware.
- Easy to quantize to INT8 for the browser (ONNX Runtime Web).
- Requires longer training schedules than LSTM; mitigated by warmup and
  cosine decay.
- Larger model than BiLSTM, but still small enough for browser edge
  (~10M params target after INT8 quantization).

## Alternatives considered (deferred)
- BiLSTM: faster to train, slightly worse accuracy.
- ST-GCN: more parameters, harder to export to ONNX, marginal gains.
- 3D CNN over landmark heatmaps: heavy, harder to deploy.