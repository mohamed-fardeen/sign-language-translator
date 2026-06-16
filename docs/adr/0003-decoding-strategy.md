# ADR 0003: Decoding -- CTC, no LM, isolated clips (v1, locked)

- Status: Accepted (v1, frozen)
- Date: 2026-06-15
- Deciders: ML Platform, Serving

## Context
v1 ships isolated signs only (single clip in, single gloss out). This
simplifies decoding considerably: there is no need for a segmenter, no
sliding window, and no language model in v1.

## Decision
- **Head**: CTC over the 500-gloss vocabulary (+ blank = 501 logits).
- **Input**: a single fixed-length clip (T=64 frames, pad/truncate).
- **Decoder**:
  - **Browser edge**: greedy CTC argmax. Fast, no extra deps.
  - **Server**: beam search with beam=8, vocabulary-constrained to the
    500-gloss set. No language model in v1.
- **Output**: a single gloss token (top-1) plus top-k and confidence.
- **Removed from earlier draft**: KenLM LM rescoring, sliding window,
  smoothing ring buffer, segmenter head. These were for continuous
  signs; v1 has no continuous-sign support.

## Consequences
- CTC is well-understood, easy to evaluate (top-1 accuracy, top-5,
  CER).
- Greedy decode keeps the browser bundle small and dependency-free.
- No sign/segmenter head, so no auxiliary loss; v1 trainer is
  single-objective (CTC + weight decay).

## Alternatives considered (deferred)
- Seq2seq with attention: better for continuous sign and gloss-to-text.
- KenLM or neural LM rescoring: deferred until we add continuous signs.
- Token-level classification: requires pre-segmentation, bad UX.