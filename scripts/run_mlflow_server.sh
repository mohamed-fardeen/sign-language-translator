#!/usr/bin/env bash
set -euo pipefail
PORT="${MLFLOW_PORT:-5000}"
HOST="${MLFLOW_HOST:-0.0.0.0}"
ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-./artifacts}"
BACKEND_URI="${MLFLOW_BACKEND_URI:-sqlite:///$(pwd)/mlruns.db}"

exec mlflow server \
    --host "$HOST" \
    --port "$PORT" \
    --backend-store-uri "$BACKEND_URI" \
    --default-artifact-root "$ARTIFACT_ROOT"
