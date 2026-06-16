# Real-Time Sign Language Translator -- Architecture

> Status: **FROZEN v1** . This document is the final, locked design.
> Any change requires a new ADR and a deliberate unfreeze.
>
> Project: production-quality ML Engineer portfolio project.
> Priorities: **buildability, resume value, zero/low cost, production
> engineering practices, public demo.**

## Scope at a glance
- **ASL only**, **isolated signs**, **500-gloss vocabulary**, locked.
- **Output**: gloss token + top-k + confidence. No English translation.
- **Two inference paths**: browser (default UX, ONNX Runtime Web) and
  server (fallback + analytics, FastAPI on Render).
- **Frontend**: Hugging Face Spaces (public demo).
- **Backend**: FastAPI in Docker, hosted on Render.
- **Training**: local machine, Kaggle, or Google Colab. PyTorch
  Lightning + MLflow + DVC.
- **Auth**: JWT (HS256).
- **No cloud-specific infrastructure** (no AWS, no ECR, no SageMaker,
  no CloudFront, no Terraform, no Kubernetes, no message queues, no
  microservices, no distributed training).

---

## 1. Goals and Non-Goals

### 1.1 Goals
- **ASL isolated-sign recognition**: input = a short sign clip (32-96
  frames at 30 fps); output = the top-1 gloss token plus top-k and
  confidence.
- **Real-time**: p95 latency <= 120 ms (browser) and <= 300 ms (server)
  per clip.
- **Two inference paths**:
  - *Browser edge* (default UX) -- MediaPipe Holistic in JS, ONNX
    Runtime Web (WASM/WebGL) on device; nothing leaves the browser for
    inference.
  - *Server* (fallback / analytics) -- FastAPI in Docker on Render;
    used when the browser model is unavailable or for telemetry.
- **Public demo** on Hugging Face Spaces pointing at the Render-hosted
  API.
- **MLOps-first**: every experiment, dataset, and model is tracked and
  reproducible.
- **Production engineering practices**: structured logging, Prometheus
  metrics, OpenTelemetry tracing, drift detection, automated tests
  (unit + integration + e2e), CI on every PR.

### 1.2 Non-Goals
- Continuous / sentence-level sign recognition.
- Gloss-to-English translation.
- Sign languages other than ASL.
- Fingerspelling recognition.
- Sign generation (text to sign).
- Speaker-dependent personalization.
- Mobile-native edge (browser only).
- AWS or any other specific cloud provider.
- Kubernetes, microservices, message queues, distributed training.
- Enterprise / multi-tenant infrastructure.

---

## 2. System Overview

```
   +-------------------+   REST (HTTPS)         +-------------------------+
   |  HF Spaces        |   JWT auth             |  Render (Docker)        |
   |  (static demo)    | ---------------------> |  FastAPI service        |
   |                   |                        |  +-------------------+  |
   |  - webcam UI      |   (optional fallback)  |  | Sign Model        |  |
   |  - MediaPipe JS   |                        |  | (TorchScript FP16)|  |
   |  - ONNX Runtime   |                        |  +-------------------+  |
   |    Web (INT8)     |                        +------------+------------+
   |  - inference      |                                     |
   +-------------------+                                     v
            ^                                       +---------+---------+
            |                                       |  Prometheus +     |
   local    | on-device                             |  OpenTelemetry    |
   inference| (default)                             |  exporter         |
            v                                       +-------------------+
   +-------------------+
   |  Sign output      |   (gloss token, top-k, confidence)
   +-------------------+


   Training plane (local / Kaggle / Colab):
   +-----------+   +----------------+   +---------------+
   |  Data     |   |  Trainer       |   |  MLflow       |
   |  Pipeline |-> |  (PyTorch      |-> |  (local       |
   |  (DVC +   |   |   Lightning +  |   |   tracking    |
   |   local)  |   |   Hydra)       |   |   server)     |
   +-----------+   +----------------+   +---------------+
        |                                     |
        v                                     v
   +-----------+                       +---------------+
   |  Local FS |                       |  Local FS or  |
   |  /data    |                       |  /artifacts   |
   +-----------+                       +---------------+
                                            |
                                            v
                                   +-------------------+
                                   |  ONNX (INT8) +    |
                                   |  TorchScript FP16 |
                                   +-------------------+
                                            |
                       +--------------------+--------------------+
                       v                                         v
            +-------------------+                       +-------------------+
            |  HF Spaces        |                       |  Render (FastAPI) |
            |  (browser model)  |                       |  (server model)   |
            +-------------------+                       +-------------------+
```

**Key flows**
- **Inference (browser, default)**: webcam -> MediaPipe Holistic in JS
  -> ONNX Runtime Web (INT8) on device -> gloss token + confidence.
  Nothing leaves the browser.
- **Inference (server, optional)**: same landmarks -> FastAPI on Render
  (TorchScript FP16) -> gloss token. Used as fallback and for
  telemetry-gated quality comparisons.
- **Training (offline)**: raw videos -> preprocess -> landmark
  extraction (cached) -> DVC versioned -> PyTorch Lightning trainer
  with Hydra config and MLflow logging -> best checkpoint -> export
  ONNX (INT8, browser) + TorchScript (FP16, server) -> publish
  artifacts to the places the runtime can reach them (HF Spaces for the
  browser bundle, Render env or release for the server model).

---

## 3. Inference Architecture

### 3.1 Two paths

| Aspect              | Browser edge (default UX)    | Server (fallback / analytics)|
|---------------------|------------------------------|------------------------------|
| Latency budget      | 120 ms p95 per clip          | 300 ms p95 per clip          |
| Model format        | ONNX INT8                    | TorchScript FP16             |
| Model size          | <= 10M params                | up to ~30M params            |
| Privacy             | nothing leaves device        | landmarks leave device       |
| MediaPipe           | `@mediapipe/holistic` (JS)   | same (browser still extracts)|
| Runtime             | ONNX Runtime Web (WASM/WebGL)| PyTorch + TorchScript        |
| Hosted at           | Hugging Face Spaces          | Render (Docker)              |

### 3.2 Isolated-sign protocol
- A "clip" is a single sign: 32-96 frames (~1-3 s at 30 fps).
- Client records the clip, downsamples/pads to a fixed `T=64` frames,
  and runs the ONNX model locally.
- Output: `{gloss_id, gloss_label, confidence, top_k[]}`.
- No streaming, no segmenter head, no partial/final state machine.

### 3.3 Server REST API (FastAPI on Render)
- `POST /v1/auth/token`  -- exchange username/password (or anonymous
  device key) for a short-lived JWT (HS256, 15 min).
- `POST /v1/predict`     -- accept a landmark clip JSON, return top-k
  glosses.
- `GET  /v1/models`      -- list active model versions.
- `GET  /v1/health`      -- liveness / readiness.
- `GET  /metrics`        -- Prometheus metrics.
- WebSocket is **not** used in v1 (no streaming, isolated only).

### 3.4 Decoding
- CTC head with `blank` + 500 glosses = 501 logits per frame.
- **Browser edge**: greedy decode (fastest, no extra deps).
- **Server**: beam search (beam=8), vocabulary-constrained to the
  500-gloss set. No language model in v1.
- Clip-level prediction: collapse CTC over time, take argmax.

---

## 4. Data Pipeline

### 4.1 Datasets (v1)
- **WLASL** (filtered to the 500-gloss subset we ship).
- **MS-ASL** (filtered to the 500-gloss subset we ship).
- **ASL Citizen** (filtered to the 500-gloss subset we ship).
- The 500-gloss vocabulary is **locked in v1** and stored in
  `datasets/vocab/vocab.json`. It is the intersection of frequent glosses
  across the three sources, curated for diversity and signer count.

### 4.2 Stages

```
raw videos (.mp4, varied fps/res)
        |  download_datasets.py
        v
raw/        (local: data/raw/<dataset>/<video_id>.mp4)
        |  preprocess_videos.py
        v
interim/    (resampled 30 fps, 256x256, audio stripped, isolated clips)
        |  extract_landmarks.py  (MediaPipe Holistic, batch)
        v
processed/  (per-frame .npz: pose, lh, rh, face, mask)
        |  build_clips.py
        v
features/   (clip-level .npz: fixed 64 frames, manifest.jsonl,
             vocab.json, train/val/test splits)
        |  DVC-tracked
        v
datasets/   (versioned, content-addressed, local + optional remote)
```

### 4.3 Storage and versioning
- **Local filesystem** is the primary store. Each developer / training
  run has the same layout.
- **DVC** tracks feature manifests, vocab, and split files; remote is
  optional and only configured if the developer wants it.
- **MLflow** records the DVC hash as a run parameter.
- **No raw video leaves the machine** unless the developer explicitly
  pushes it. No public buckets.

### 4.4 Augmentation (training only)
Landmark-space augmentations (no pixel-level warps):
- Random rotation around the upper body axis (+/- 15 deg).
- Temporal stretch (0.85x to 1.15x).
- Random frame dropout (simulate fast signing).
- Joint-coordinate jitter (Gaussian on (x,y), z scaled by depth).
- Hand swap and "missing hand" simulation (real-world failure mode).

---

## 5. Model Architecture

### 5.1 Input streams (per frame, MediaPipe Holistic)
| Stream         | Source         | Dims (x,y,z) |
|----------------|----------------|--------------|
| Pose           | 33 keypoints   | 99           |
| Left hand      | 21 keypoints   | 63           |
| Right hand     | 21 keypoints   | 63           |
| Face (subset)  | ~40 keypoints  | 120          |
| **Concat**     |                | **345**      |

Face subset = lips + eyebrows + jaw (the visually meaningful channels
for non-manual markers). A canonical face-subset definition lives in
`configs/features/mediapipe_holistic.yaml`.

### 5.2 Network

```
       Pose (B,T,99)  LH (B,T,63)  RH (B,T,63)  Face (B,T,120)
            |              |            |             |
            v              v            v             v
       +-------+      +-------+    +-------+     +-------+
       | MLP   |      | MLP   |    | MLP   |     | MLP   |   per-stream encoders
       | LN,GELU|     | LN,GELU|   | LN,GELU|    | LN,GELU|   (linear -> 128)
       +---+---+      +---+---+    +---+---+     +---+---+
           \              \           /             /
            +--------------+---------+-------------+
                                  v
                         +-------------------+
                         |  Fusion (concat)  |
                         |    -> 512 dim     |
                         +---------+---------+
                                   v
                  +----------------------------------+
                  |  Temporal Backbone               |
                  |  Transformer encoder (6L, d=512, |
                  |  8 heads, GELU, pre-LN)         |
                  +-----------------+----------------+
                                    v
                          +-------------------+
                          |  CTC Head (501)   |
                          +---------+---------+
                                    v
                         gloss logits (B,T,501)
```

### 5.3 Heads
- **Primary**: CTC over the 500-gloss vocabulary (+ blank = 501 logits).
- No segmenter head in v1 (isolated signs only, no idle/active gating).
- No seq2seq / no English translation in v1.

### 5.4 Loss
```
L = L_ctc + lambda_reg * ||theta||^2
```
Defaults: `lambda_reg=1e-5`. Single objective keeps the v1 trainer simple.

### 5.5 Decoding
- **Browser edge**: greedy CTC (fastest, no extra deps).
- **Server**: beam search with beam=8, vocabulary-constrained to the
  500-gloss set. No language model in v1.
- Clip-level prediction: collapse CTC over time, take argmax.

### 5.6 Export
- **TorchScript (FP16)** for the **server** (Render FastAPI container).
- **ONNX (FP16, dynamic `T` axis)** -> then
  `onnxruntime.quantization` (dynamic INT8) -> ship to the browser.
- The browser loads the ONNX model directly via ONNX Runtime Web
  (WASM/WebGL backend).
- Both exports are produced by `scripts/export_onnx.py` and
  `scripts/export_torchscript.py` and logged as MLflow artifacts.

---

## 6. Training and MLOps

### 6.1 Training environment
- **Local machine** (developer laptop / desktop with a CUDA GPU if
  available; CPU is fine for small smoke runs).
- **Kaggle Notebooks** -- free GPU; well-suited for the v1 training
  scale.
- **Google Colab** -- free / Colab Pro GPU; same code path.
- One trainer entry point: `python scripts/train.py` works in all
  three. The only environment-specific concern is the artifact output
  directory (mounted drive on Kaggle/Colab, local FS on a laptop).

### 6.2 Trainer
- **PyTorch** + **PyTorch Lightning** for clean training loops.
- **Hydra** (`hydra-core` + `omegaconf`) composes configs from
  `configs/`. Single command: `python scripts/train.py
  model=sign_transformer data=wlasl`.
- Single-node by default; multi-GPU DDP via
  `Trainer(strategy="ddp")` only when a multi-GPU machine is
  available.
- Mixed precision (bf16 on Ampere+, fp16 fallback).
- Gradient clipping (`max_norm=1.0`).
- AdamW + cosine schedule with 5% warmup.
- Epoch-based eval; early stopping on `val_accuracy` (top-1 gloss).

### 6.3 Experiment tracking (MLflow)
- **MLflow tracking server** runs locally (`mlflow server
  --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns.db
  --default-artifact-root ./artifacts`) for v1. No managed / cloud
  MLflow in v1.
- Every run logs:
  - Config snapshot (Hydra YAML).
  - Git SHA, DVC dataset hash, Python/dependency versions.
  - Per-epoch metrics (`train_loss`, `val_accuracy`, `val_top5_accuracy`,
    `val_cer`, `lr`, `grad_norm`).
  - System metrics (GPU util, mem, throughput).
  - Model checkpoints (top-3 by `val_accuracy`).
  - ONNX (INT8) and TorchScript (FP16) exports.
- **Model Registry** with stages: `None -> Staging -> Production ->
  Archived`. Promotion requires: `val_accuracy >= production_acc *
  1.01` AND a browser-ONNX smoke test passes in CI.

### 6.4 Data versioning
- **DVC** for feature manifests, vocab, and split files.
- Pipeline (`dvc.yaml`) declares: `preprocess -> landmarks -> features
  -> train -> evaluate`. Re-running respects cache.
- The DVC remote is optional and configured by the developer
  (defaults to local).

### 6.5 Reproducibility
- Pinned `requirements/*.txt` files.
- Seeded RNG (Python, NumPy, PyTorch, CUDA).
- Single command to reproduce: `make reproduce EXP=<run_id>`.
- The exact Python and dependency versions are recorded in the MLflow
  run.

### 6.6 CI/CD (GitHub Actions)
- **CI on every PR**:
  - Lint (`ruff check`, `ruff format --check`).
  - Type check (`mypy src`).
  - Unit + integration tests (`pytest tests/unit tests/integration`).
  - Build Docker images for the FastAPI service.
- **Publish workflow** (manual dispatch, on tag, or on merge to
  `main`):
  - Run the trainer against the latest DVC-tracked features.
  - Promote the run in MLflow.
  - Export ONNX + TorchScript and upload artifacts to a release
    artifact (GitHub Release) and to a place the demo can read them
    (Hugging Face Space repo for the browser bundle; Render env or
    release for the server model).
  - Trigger Render deploy hook (Render is configured by Git integration
    or by a deploy hook URL stored as a GitHub Actions secret).

### 6.7 Resume-aligned claims (truthful, v1)
- "Built a Transformer-based sign language recognizer (PyTorch
  Lightning)."
- "Used MediaPipe Holistic for landmark extraction."
- "Tracked experiments with MLflow; versioned datasets with DVC."
- "Containerized the inference service with Docker; deployed on
  Render."
- "Exported the model to ONNX (INT8) and shipped it to the browser via
  ONNX Runtime Web."
- "CI/CD with GitHub Actions (lint, type, test, build, publish)."
- **Do not** claim AWS, SageMaker, ECR, CloudFront, or Terraform --
  none of those are in v1.

---

## 7. Serving Architecture

### 7.1 Why FastAPI
- Native async, first-class Pydantic schemas, OpenAPI docs out of the
  box.
- Plays well with `uvicorn`/`gunicorn` and ASGI middleware.
- One process handles auth, batching, and the model forward pass.

### 7.2 Process topology (Render)
- Single FastAPI app, containerized via `docker/serve.Dockerfile`.
- Render runs the container from a GitHub-connected repo or from an
  image registry of the developer's choice.
- The model is loaded once at startup and reused for every request.
- Health: `/v1/health` (liveness + readiness, model warmup state).

### 7.3 Endpoints (v1)
| Method | Path                | Auth   | Purpose                                |
|--------|---------------------|--------|----------------------------------------|
| POST   | `/v1/auth/token`    | none   | issue short-lived JWT                  |
| POST   | `/v1/predict`       | JWT    | one-shot gloss prediction              |
| GET    | `/v1/models`        | JWT    | list active model versions             |
| GET    | `/v1/health`        | none   | liveness/readiness                     |
| GET    | `/metrics`          | none   | Prometheus metrics                     |

There is **no WebSocket endpoint** in v1 (no streaming, isolated only).

### 7.4 Schema
- Request: `{clip: {pose: float[T,99], lh: float[T,63], rh: float[T,63],
  face: float[T,120], mask: bool[T]}}`, `T` is padded/truncated to 64.
- Response: `{gloss_id, gloss_label, confidence, top_k: [{id, label,
  prob}], latency_ms, model_version}`.

### 7.5 Security
- **JWT (HS256)** for v1. The browser app is anonymous and obtains a
  short-lived (15 min) token via `POST /v1/auth/token` using a device
  key issued at first load. Tokens are validated by FastAPI middleware
  on every protected route. (See ADR 0007.)
- Rate limit per token (token bucket; 60 req/min default).
- The JWT signing secret is read from an environment variable on
  Render; for local dev it lives in `.env`. No secrets in the repo.
- Model outputs are never logged with raw user data; only feature
  statistics (mean, std, NaN-rate) for drift detection.

### 7.6 Caching
- Per-process landmark normalizer cache.
- No response cache in v1 (kept simple; added only if observed).

---

## 8. Deployment

### 8.1 Where things run
| Component                     | Where it runs                                |
|-------------------------------|----------------------------------------------|
| Data download / preprocess    | Local, Kaggle, or Colab                      |
| Training                      | Local, Kaggle, or Colab                      |
| MLflow tracking server        | Local (developer machine)                    |
| DVC remote                    | Local (or developer-configured remote)       |
| Model export (ONNX/TS)        | Same env as training                         |
| Frontend (HF Spaces demo)     | Hugging Face Spaces (Gradio or static HTML)  |
| Backend (FastAPI in Docker)   | Render                                        |
| Browser inference             | End-user browser (ONNX Runtime Web)          |

### 8.2 Hugging Face Spaces (frontend / demo)
- Hosts the static web app (HTML/JS) for the public demo.
- The web app loads:
  - the browser ONNX model from a pinned release URL (HF Space
    `resolve/main` or GitHub Release),
  - the vocab JSON from the same source.
- Either a Gradio Space or a static HTML Space works. The repo includes
  `hf_space/web/` with the assets.

### 8.3 Render (backend)
- Render service is defined in `render.yaml` at the repo root.
- It builds from `docker/serve.Dockerfile`, runs the FastAPI app on
  the port Render exposes.
- Environment variables (see `.env.example`):
  `JWT_SECRET`, `MODEL_VERSION`, `BEAM_SIZE`, etc.
- Free or starter instance is sufficient for v1 traffic.

### 8.4 Containers
- `docker/serve.Dockerfile`  -- slim PyTorch (CPU is fine for v1),
  FastAPI entrypoint. The image Render builds.
- `docker/train.Dockerfile`  -- PyTorch + optional CUDA, training
  entrypoint; used by CI for reproducible builds and locally via
  `docker run --gpus all`.
- `docker/client.Dockerfile` -- minimal headless OpenCV + MediaPipe
  client for E2E tests.
- `docker-compose.yml`       -- local stack: MLflow + app.
- `docker-compose.dev.yml`   -- adds Jupyter and TensorBoard.

### 8.5 Environments
- `dev`     -- local docker-compose; smoke test only.
- `staging` -- Render preview service (auto-created on PRs by Render's
  Git integration) or a manually-deployed Render service.
- `prod`    -- Render production service. Auto-deploy on `main`.

### 8.6 CI/CD flow
```
PR   -> CI (lint, type, unit, integration, build images)
main -> publish (train -> eval gate -> register -> export
       -> publish browser bundle to HF Space
       -> trigger Render deploy)
```

### 8.7 Rollback
- Render: revert to previous deploy from the Render dashboard.
- Hugging Face Space: revert to a previous git revision of the Space
  repo.
- MLflow: stage regression `Production -> Archived` on the bad version.

---

## 9. Observability

### 9.1 Logging
- Structured JSON logs (structlog / loguru).
- Correlation ID per request, propagated to downstream calls.
- Levels: INFO (default), DEBUG (verbose per-request), ERROR (with
  traceback).
- In production, Render's log drain captures stdout/stderr; in local
  dev, logs go to the terminal.

### 9.2 Metrics (Prometheus)
- Server latency: `predict_latency_seconds{quantile}`.
- Throughput: `predictions_total`, `auth_tokens_issued_total`.
- Quality proxies: `prediction_confidence_histogram`,
  `no_hand_detected_total`, `low_confidence_predictions_total`.
- Model: `model_load_seconds`, `model_version_info{version}`.
- System: CPU, memory, batch size histogram.
- Browser side (sampled): end-to-end clip latency, model load time, ORT
  backend in use (WASM vs WebGL).

### 9.3 Tracing
- OpenTelemetry instrumentation on FastAPI and the model forward.
- One span per `/v1/predict` request, with attributes for batch size
  and clip length.
- The OTLP exporter points to a local collector (`otel-collector`) in
  dev; in production it can point at any OTLP-compatible backend the
  developer wants (or simply log spans in v1).

### 9.4 Drift detection
- Track input feature distributions (per-joint mean/std, NaN-rate,
  zero-hand-rate).
- Track output distribution (top-k gloss frequency, mean confidence).
- Trigger retrain when PSI > 0.2 over a 24h window.
- Implemented as a scheduled script (`monitoring/drift_detection.py`)
  that reads from logs and writes a report.

### 9.5 Alerting
- Out of scope for v1 (no managed alerting target). The
  `monitoring/alerts.yaml` file is provided as a starting point for
  whoever wires up a backend later.

---

## 10. Performance Budgets

Per-clip p95 budgets:

| Stage                          | Server    | Browser edge |
|--------------------------------|-----------|--------------|
| Camera capture + MediaPipe     | n/a (landmarks arrive) | 25 ms |
| Network round trip             | 20 ms     | 0 ms         |
| Server batching window         | 30 ms     | n/a          |
| Model forward (1 clip)         | 60 ms     | 50 ms        |
| Decode (greedy / beam=8)       | 5 ms      | 5 ms         |
| **Total**                      | **~115 ms** | **~80 ms** |
| Headroom to target             | to 300 ms | to 120 ms    |

---

## 11. Folder Structure (with rationale)

```
sign-language-translator/
|-- ARCHITECTURE.md                    # this document (FROZEN v1)
|-- README.md                          # quickstart, pointer to docs
|-- LICENSE
|-- Makefile                           # canonical command surface
|-- pyproject.toml                     # build, ruff, mypy config
|-- render.yaml                        # Render service definition
|-- requirements/
|   |-- base.txt                       # shared deps
|   |-- train.txt                      # + pytorch, lightning, mlflow, hydra, dvc
|   |-- serve.txt                      # + fastapi, uvicorn, onnxruntime
|   `-- dev.txt                        # + ruff, mypy, pytest
|-- .env.example
|-- .gitignore
|-- .dockerignore
|-- .pre-commit-config.yaml
|-- docker-compose.yml                 # local MLflow + app
|-- docker-compose.dev.yml             # + jupyter, tensorboard, otel-collector
|-- docker/
|   |-- train.Dockerfile
|   |-- serve.Dockerfile               # the image Render builds
|   `-- client.Dockerfile              # headless client for E2E tests
|
|-- configs/                           # all hyperparameters (Hydra)
|   |-- base.yaml                      # hydra defaults
|   |-- data/wlasl.yaml                # filtered to 500-gloss subset
|   |-- data/msasl.yaml                # filtered to 500-gloss subset
|   |-- data/asl_citizen.yaml          # filtered to 500-gloss subset
|   |-- features/mediapipe_holistic.yaml
|   |-- model/sign_transformer.yaml    # 6L Transformer, d=512
|   |-- train/default.yaml
|   |-- train/ablations/*.yaml
|   |-- serve/local.yaml
|   `-- serve/production.yaml
|
|-- data/                              # gitignored
|   |-- README.md
|   `-- .gitkeep
|-- datasets/                          # versioned, DVC-tracked
|   |-- README.md
|   |-- external/                      # raw imports (.gitkeep)
|   |-- annotations/                   # COCO-style + vocab
|   `-- vocab/                         # built by build_vocab.py
|
|-- src/signlang/                      # the package (importable)
|   |-- config.py                      # hydra/omegaconf entry point
|   |-- data/
|   |   |-- ingestion/                 # downloaders, validators
|   |   |-- preprocessing/             # fps/resize/audio strip
|   |   |-- landmarks/                 # mediapipe wrappers
|   |   |-- augmentation/              # landmark-space augs
|   |   |-- datasets/                  # clip datasets
|   |   `-- datamodules.py             # lightning DataModule
|   |-- models/
|   |   |-- encoders/                  # per-stream MLPs
|   |   |-- backbones/                 # transformer encoder
|   |   |-- heads/                     # ctc head
|   |   |-- losses.py                  # CTC loss (+ weight decay)
|   |   `-- sign_model.py              # LightningModule wrapper
|   |-- training/
|   |   |-- trainer.py                 # build_trainer()
|   |   |-- callbacks/                 # mlflow, ckpt, early stop
|   |   |-- schedulers.py
|   |   `-- optimizers.py
|   |-- inference/
|   |   |-- predictor.py               # batched async predictor
|   |   `-- postprocess.py             # CTC -> top-k glosses
|   |-- serving/
|   |   |-- app.py                     # FastAPI factory
|   |   |-- api/v1/                    # auth, predict, models, health
|   |   |-- schemas/                   # pydantic models
|   |   `-- middleware.py              # JWT, rate limit, tracing
|   |-- web/                           # browser app
|   |   |-- index.html                 # entry point served by FastAPI / HF
|   |   |-- js/
|   |   |   |-- app.js                 # UI glue
|   |   |   |-- capture.js             # getUserMedia + MediaPipe
|   |   |   `-- inference.js           # ONNX Runtime Web wrapper
|   |   `-- css/styles.css
|   |-- tracking/
|   |   `-- mlflow_utils.py
|   |-- evaluation/
|   |   |-- metrics.py                 # top-1, top-5, CER
|   |   `-- reports.py
|   |-- pipelines/
|   |   |-- training_pipeline.py
|   |   `-- inference_pipeline.py
|   `-- utils/                         # io, logging, timing, seeding
|
|-- scripts/                           # CLI entry points (thin)
|   |-- download_datasets.py
|   |-- preprocess_videos.py
|   |-- extract_landmarks.py
|   |-- build_vocab.py
|   |-- train.py
|   |-- evaluate.py
|   |-- export_onnx.py
|   |-- export_torchscript.py
|   `-- run_mlflow_server.sh
|
|-- notebooks/                         # exploration only; never ships
|   `-- .gitkeep
|
|-- tests/
|   |-- conftest.py
|   |-- unit/
|   |   |-- data/                      # dataset shape, augmentations
|   |   |-- models/                    # forward shapes, loss values
|   |   |-- inference/                 # decode
|   |   `-- api/                       # FastAPI TestClient
|   |-- integration/
|   |   |-- training/                  # 1-epoch dry run on CPU
|   |   `-- serving/                   # batched predictor + auth
|   |-- e2e/
|   |   `-- test_browser_to_endpoint.py # synthetic frames -> REST -> gloss
|   `-- fixtures/                      # tiny synthetic clips
|
|-- hf_space/                          # files pushed to the HF Space repo
|   |-- README.md                      # Space metadata
|   |-- app.py                         # Gradio entry (optional)
|   `-- web/                           # mirror of src/signlang/web/
|
|-- monitoring/
|   |-- drift_detection.py             # scheduled job
|   `-- alerts.yaml
|
|-- docs/
|   |-- README.md
|   |-- api_reference.md
|   |-- data_schema.md
|   |-- model_card.md
|   |-- deployment.md
|   |-- operations.md
|   `-- adr/                           # architecture decision records
|       |-- 0001-mediapipe-holistic.md
|       |-- 0002-temporal-architecture.md
|       |-- 0003-decoding-strategy.md
|       |-- 0004-deployment-target.md
|       |-- 0005-data-versioning.md
|       |-- 0006-browser-edge-onnx.md
|       `-- 0007-jwt-auth.md
|
|-- mlruns/                            # gitignored (MLflow local store)
|-- artifacts/                         # gitignored
|   |-- checkpoints/
|   |-- exported/                      # ONNX + TorchScript
|   `-- logs/
|-- .github/workflows/{ci.yml, publish.yml}
`-- dvc.yaml, params.yaml              # DVC pipeline + params
```

### 11.1 Why this structure
- **One importable package (`src/signlang`)** keeps tests, notebooks,
  and scripts referencing the same code path. Avoids `sys.path` hacks.
- **Configs separate from code** so a single trainer serves many model
  recipes; experiment deltas are config diffs, not code diffs.
- **Train/serve/web are siblings** under the same package; this
  forces them to share types and schemas, preventing drift.
- **`render.yaml`** at the repo root is the single source of truth for
  the Render service definition.
- **`hf_space/`** holds the files that get pushed to the public
  Hugging Face Space repo (kept separate from the main package so
  HF's release flow is obvious).
- **`docs/adr/`** captures *why* a decision was made; future agents
  and humans do not re-litigate.

---

## 12. MLOps Stack (locked)

| Concern               | Tool                           |
|-----------------------|--------------------------------|
| Training framework    | PyTorch                        |
| Trainer scaffolding   | PyTorch Lightning              |
| Config composition    | Hydra (omegaconf)              |
| Experiment tracking   | MLflow                         |
| Data versioning       | DVC                            |
| CI/CD                 | GitHub Actions                 |
| Containerization      | Docker                         |
| Backend hosting       | Render                         |
| Frontend hosting      | Hugging Face Spaces            |
| Browser inference     | ONNX Runtime Web               |
| Model export (server) | TorchScript (FP16)             |
| Model export (client) | ONNX (INT8)                    |
| Auth                  | JWT (HS256)                    |
| Observability         | structlog, Prometheus, OpenTelemetry |

**Not in v1** (and intentionally not added): AWS, SageMaker, ECR,
CloudFront, Terraform, ALB, Kubernetes, microservices, message queues,
distributed training, KenLM, ST-GCN, BiLSTM alternatives, seq2seq,
multi-language, fingerspelling, continuous-sign recognition, gloss-to-
English translation.

---

## 13. Performance Targets

- **Browser p95 <= 120 ms** per clip.
- **Server p95 <= 300 ms** per clip.
- See section 10 for the per-stage budget breakdown.

---

## 14. Resume Alignment

The final implementation should honestly support these resume claims:

- Transformer-based sign language recognition.
- MediaPipe landmark extraction.
- MLflow experiment tracking.
- DVC data versioning.
- Docker containerization.
- FastAPI inference service.
- ONNX model optimization (INT8 quantization).
- Browser-based ONNX Runtime deployment.
- GitHub Actions CI/CD.

It should **not** claim AWS, SageMaker, ECR, CloudFront, Terraform,
Kubernetes, or any other infrastructure not listed in section 12.

---

## 15. Frozen

This document is the final, locked architecture for v1. Any change
requires a new ADR and an explicit unfreeze. New cloud providers, new
infrastructure, research features outside scope, and future
optimizations are not part of v1.