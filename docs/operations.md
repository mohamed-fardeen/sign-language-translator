# Operations

## Local dev
```bash
cp .env.example .env
make dev-install
make stack         # MLflow + app
make serve         # FastAPI on :8000
```

## Training
```bash
python scripts/train.py data=wlasl
# or
make train
```

## Evaluation
```bash
python scripts/evaluate.py --ckpt artifacts/checkpoints/sign-epoch*.ckpt
```

## Export
```bash
python scripts/export_onnx.py --ckpt artifacts/checkpoints/best.ckpt --out-dir artifacts/exported/browser
python scripts/export_torchscript.py --ckpt artifacts/checkpoints/best.ckpt --out-dir artifacts/exported/server
```

## Observability
- Structured JSON logs (structlog).
- Prometheus metrics at `GET /metrics`.
- Optional OpenTelemetry via `OTEL_EXPORTER_OTLP_ENDPOINT`.

## Drift detection
`monitoring/drift_detection.py` reads logs and computes:
- Per-joint mean/std shift (PSI).
- Output top-k frequency shift.
- Mean confidence drift.

Run on a schedule. PSI > 0.2 over 24h triggers a retrain ticket.

## Troubleshooting
- **Model not loaded**: confirm `MODEL_PATH` points to a valid `.pt`
  file produced by `scripts/export_torchscript.py`.
- **401 from /v1/predict**: token expired or missing. Reissue via
  `/v1/auth/token`.
- **Browser model not loading**: confirm `/artifacts/exported/browser/model_int8.onnx`
  is reachable; CORS must allow the request origin.
- **Long server latency**: confirm `precision` is `bf16-mixed` on the
  training host; `cpu` is expected to be slow.
