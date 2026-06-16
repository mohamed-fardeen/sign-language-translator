FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ ./requirements/
RUN pip install -r requirements/base.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "pytest", "tests/e2e", "-v"]
