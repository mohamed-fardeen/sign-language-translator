# ADR 0006: Browser edge inference via ONNX Runtime Web (v1, locked)

- Status: Accepted (v1, frozen)
- Date: 2026-06-15
- Deciders: ML Platform, Web

## Context
v1's primary user experience is a web app. The browser should run
inference locally for privacy and latency, with the Render-hosted
FastAPI as a fallback.

## Decision
- Serve the model as a **quantized ONNX (INT8)** file via the public
  Hugging Face Space (or a GitHub Release) at a pinned versioned path.
- The browser app loads it with **ONNX Runtime Web** (WASM backend by
  default; WebGL when available).
- MediaPipe Holistic runs in the browser via `@mediapipe/holistic` (JS).
- The web app is plain HTML/JS, hosted on the HF Space (no React build
  step) to keep the surface area small.
- The vocab JSON and any preprocessing constants are co-located with
  the ONNX file in the same versioned directory.

## Consequences
- Zero video ever leaves the user's machine for the default path.
- The Render endpoint is only hit on user action ("Use server model")
  and is the source of telemetry / quality comparisons.
- Versioning: the client pins a model version in localStorage;
  upgrades are non-blocking background fetches.

## Alternatives considered
- React + Webpack build: nicer DX, more moving parts. Deferred.
- WebGPU backend: not yet broadly available; out of scope for v1.