FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsm6 \
        libxext6 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ ./requirements/
RUN pip install -r requirements/base.txt -r requirements/serve.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY pyproject.toml ./
COPY .env.example ./.env.example

ENV PYTHONPATH=/app/src \
    APP_ENV=production \
    SERVE_ENV=production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/v1/health || exit 1

CMD ["uvicorn", "signlang.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
