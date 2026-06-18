# Model card

## Overview
ASL isolated-sign recognizer. Given a 1-3 s video clip of a single
sign, outputs the top-1 gloss token (plus top-k and confidence) from
a fixed 500-gloss vocabulary.

## Model
- **Backbone**: 6-layer Transformer encoder, d=512, 8 heads, GELU, pre-LN.
- **Input**: per-stream MLPs (128 dim) over pose, left hand, right hand
  (225 dim total), then fusion (concat) to 512 dim. No face stream in v1.
- **Head**: CTC over 500 glosses + blank = 501 logits per frame.
- **Loss**: CTC + L2 weight decay (1e-5).
- **Optimizer**: AdamW (lr=3e-4, betas=(0.9, 0.98)).
- **Schedule**: cosine with 5% warmup.
- **Augmentations** (training only): rotation (+/-15 deg), temporal
  stretch (0.85-1.15x), random frame dropout (10%), Gaussian coordinate
  jitter, hand-swap and zero-hand simulation.

## Decode
- **Browser (default)**: greedy CTC argmax, ONNX Runtime Web (WASM/WebGL).
- **Server**: beam search (beam=8), vocabulary-constrained. No language
  model in v1.

## Export
- **Server**: TorchScript (FP16), `scripts/export_torchscript.py`.
- **Browser**: ONNX (FP16) -> INT8 dynamic quantization, `scripts/export_onnx.py`.

## Intended use
- Single signer, controlled lighting, isolated sign.
- English-language context, ASL grammar.

## Out-of-scope
- Continuous / sentence-level signs.
- Fingerspelling.
- Sign-to-English translation.
- Other sign languages.
- Real-time streaming.

## Performance budget (p95)
- Browser: <= 120 ms / clip.
- Server:  <= 300 ms / clip.

## Limitations
- Vocabulary is locked to 500 glosses; OOV signs are not recognised.
- Trained signers only (no child, no severe mobility differences).
- Trained lighting; extreme low-light or backlight may degrade accuracy.
