.PHONY: help install dev-install lint type test unit integration e2e \
        preprocess landmarks train evaluate export serve mlflow stack \
        docker-train docker-serve docker-client hf-sync render-logs clean

PY ?= python
PIP ?= pip

help:
	@echo "Targets:"
	@echo "  install / dev-install   - install dependencies"
	@echo "  lint / type / test      - quality gates"
	@echo "  preprocess / landmarks  - data pipeline"
	@echo "  train / evaluate / export"
	@echo "  serve / mlflow / stack  - local infra"
	@echo "  docker-{train,serve,client} - build images"
	@echo "  hf-sync                 - mirror web/ to the HF Space repo"
	@echo "  render-logs             - tail Render service logs"

install:
	$(PIP) install -r requirements/base.txt

dev-install:
	$(PIP) install -r requirements/dev.txt -r requirements/train.txt

lint:
	ruff check src tests
	ruff format --check src tests

type:
	mypy src

unit:
	pytest tests/unit -q

integration:
	pytest tests/integration -q

e2e:
	pytest tests/e2e -q

test: unit integration

preprocess:
	$(PY) scripts/preprocess_videos.py

landmarks:
	$(PY) scripts/extract_landmarks.py

train:
	$(PY) scripts/train.py

evaluate:
	$(PY) scripts/evaluate.py

export:
	$(PY) scripts/export_onnx.py
	$(PY) scripts/export_torchscript.py

serve:
	uvicorn signlang.serving.app:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	bash scripts/run_mlflow_server.sh

stack:
	docker compose -f docker-compose.yml up -d

docker-train:
	docker build -f docker/train.Dockerfile -t signlang-train:latest .

docker-serve:
	docker build -f docker/serve.Dockerfile -t signlang-serve:latest .

docker-client:
	docker build -f docker/client.Dockerfile -t signlang-client:latest .

hf-sync:
	@echo "Mirror hf_space/ to the Hugging Face Space repo (configure the"
	@echo "remote first, e.g. `git remote add hf <space-git-url>`)."
	git push hf `git subtree split --prefix hf_space HEAD`:main --force

render-logs:
	@echo "Tail Render logs: open the Render dashboard for the service."

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +