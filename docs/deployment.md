# Deployment

## Components
| Component     | Where                                              |
|---------------|----------------------------------------------------|
| Training      | Local / Kaggle / Colab                            |
| MLflow server | Local developer machine                            |
| DVC remote    | Local (or developer-configured)                   |
| Browser ONNX  | Hugging Face Space (public)                        |
| Server (API)  | Render (FastAPI in Docker)                         |
| Auth secret   | Render env var `JWT_SECRET`                        |

## Render
`render.yaml` at the repo root is the source of truth.

1. Push the repo to GitHub.
2. In Render, "New -> Web Service" -> connect the GitHub repo.
3. Render auto-detects `render.yaml` and provisions the service.
4. Set `JWT_SECRET` in the Render service env (not committed).
5. Render builds `docker/serve.Dockerfile` and runs the FastAPI app.

Health check: `GET /v1/health` -> `{ "status": "ok", "model_loaded": true }`.

The model artifact (`artifacts/exported/server/model.pt`) is produced by
`scripts/export_torchscript.py` from the best checkpoint in
`artifacts/checkpoints/`. Upload it to the Render service via the
GitHub Actions publish workflow, or include it in the Docker image
during the build (simplest: copy into `artifacts/exported/server/`).

## Hugging Face Space
The Space is in `hf_space/`. The repo at the top level mirrors that
folder on the HF Space side. The Space runs `app.py` (Gradio) which
links out to the live browser demo at the Render URL.

## Publish workflow
`.github/workflows/publish.yml` runs on tag push or manual dispatch.
It (optionally) trains, exports ONNX + TorchScript, uploads artifacts,
and triggers the Render deploy hook.
