# Real-Time Sign Language Translator

Production-quality real-time **ASL isolated-sign recognizer**.
See [`ARCHITECTURE.md`](./ARCHITECTURE.md) -- **frozen v1**.

## v1 Scope
- **ASL only**, **isolated signs**, **500-gloss vocabulary**.
- **Output**: gloss token + top-k + confidence. No English translation.
- **Inference**: browser (default) via MediaPipe JS + ONNX Runtime Web;
  server (fallback / analytics) via FastAPI in Docker on Render.
- **Frontend demo**: Hugging Face Spaces.
- **Training**: local / Kaggle / Colab, PyTorch Lightning + Hydra +
  MLflow + DVC.
- **Auth**: JWT (HS256).

## Stack
Python, OpenCV, MediaPipe, PyTorch, PyTorch Lightning, Hydra, MLflow,
DVC, FastAPI, Docker, ONNX, ONNX Runtime Web, GitHub Actions, Render,
Hugging Face Spaces.

## Quickstart
```bash
cp .env.example .env
make dev-install
make stack            # local MLflow + app
make preprocess
make landmarks
make train
make serve            # FastAPI on :8000
# browser demo:
open http://localhost:8000/web/
```

## Layout
- `src/signlang/`    -- importable package (data, models, training,
                         inference, serving, web)
- `configs/`         -- Hydra configs (all hyperparameters)
- `scripts/`         -- CLI entry points
- `hf_space/`        -- files pushed to the public HF Space repo
- `docs/adr/`        -- architecture decision records (0001-0009)
- `tests/`           -- unit, integration, e2e
- `render.yaml`      -- Render service definition
- `dvc.yaml`         -- DVC pipeline

## Status
Architecture: **FROZEN v1**. No AWS, no SageMaker, no Terraform, no
Kubernetes. Resume-aligned to:
- Transformer-based sign language recognition
- MediaPipe landmark extraction
- MLflow experiment tracking
- DVC data versioning
- Docker containerization
- FastAPI inference service
- ONNX model optimization
- Browser-based ONNX Runtime deployment
- GitHub Actions CI/CD