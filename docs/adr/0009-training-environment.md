# ADR 0009: Training environment -- local, Kaggle, or Colab (v1, locked)

- Status: Accepted (v1, frozen)
- Date: 2026-06-15
- Deciders: ML Platform

## Context
Training a 500-gloss isolated-sign CTC model fits comfortably in the
free GPU tier of Kaggle and Colab. Adding managed training would push
the project into enterprise-scale territory and conflict with the
"zero/low cost" priority.

## Decision
- **Local machine** is the default development environment.
- **Kaggle Notebooks** and **Google Colab** are first-class training
  environments for the v1 model size. The same `scripts/train.py`
  entry point works in all three.
- Environment-specific concerns (artifact output path, DVC remote,
  MLflow tracking URI) are surfaced via env vars and a single
  `configs/env/{local,kaggle,colab}.yaml` (no code branches).
- No managed training service (SageMaker, Vertex, etc.) in v1.

## Consequences
- Zero training cost on Kaggle/Colab free tiers.
- One trainer, three runtimes -- no specialized code paths to maintain.
- Reproducibility is preserved by pinning dependencies and seeding
  RNGs; the exact Python version is recorded in the MLflow run.

## Alternatives considered
- SageMaker Training Jobs: removed with the AWS footprint.
- Self-hosted GPU box: ongoing cost, no clear win.
- Lambda / spot instances: too much orchestration for a portfolio
  project.