# ADR 0008: Hydra for config composition (v1)

- Status: Accepted
- Date: 2026-06-15
- Deciders: ML Platform

## Context
The trainer needs to compose many settings (model, data, optimizer,
augmentation) and run many ablations. Hard-coding arguments or
hand-rolled YAML loaders get unwieldy fast.

## Decision
Use **Hydra** (`hydra-core` + `omegaconf`) for config composition.
- One `configs/base.yaml` declares defaults.
- Per-concern YAMLs under `configs/{data,model,features,train,serve}/`.
- CLI overrides: `python scripts/train.py model=sign_transformer
  data=wlasl train.lr=3e-4`.
- The full resolved config is logged to MLflow on every run.

## Consequences
- Experiment deltas are config diffs, not code diffs.
- The trainer script is one line long and trivially testable.
- Hydra's defaults list makes new ablations safe (one file added, no
  other changes).

## Alternatives considered
- Plain argparse + YAML: works for one config, gets ugly fast.
- Sacred / gin: smaller community, less momentum.