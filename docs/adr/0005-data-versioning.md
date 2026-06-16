# ADR 0005: DVC for dataset versioning, local-first (v1)

- Status: Accepted
- Date: 2026-06-15
- Deciders: ML Platform

## Context
We need reproducible training where the exact feature set, vocabulary,
and splits are tied to each run. The portfolio project should not
require an S3 bucket by default.

## Decision
- **DVC** tracks feature manifests, vocabularies, and split files.
- The **default DVC remote is local** (`./.dvc/cache` or a local
  directory); the developer can opt into a remote (S3, GCS, SSH) at
  any time without code changes.
- **MLflow** records the DVC hash as a run parameter.
- The DVC pipeline (`dvc.yaml`) declares: `preprocess -> landmarks ->
  features -> train -> evaluate`. Cache is honored across runs.

## Consequences
- Dataset and code share a single reproducibility story.
- A run can be re-executed via `dvc repro` against the recorded hash.
- Two sources of truth (DVC + MLflow params) are kept in sync via a
  pre-train hook.

## Alternatives considered
- LakeFS / Pachyderm: heavier; out of scope for v1.
- Manual remote versioning: error-prone, no pipeline semantics.