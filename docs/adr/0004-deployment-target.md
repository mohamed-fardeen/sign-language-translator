# ADR 0004: Deployment target -- Render (backend) + Hugging Face Spaces (frontend) (v1, locked)

- Status: Accepted (v1, frozen)
- Date: 2026-06-15
- Deciders: ML Platform

## Context
Earlier drafts targeted AWS SageMaker, ECR, CloudFront, and Terraform.
For an ML Engineer portfolio project with priorities on buildability
and zero/low cost, that footprint is overkill.

## Decision
- **Backend**: FastAPI in Docker, hosted on **Render**. Single
  service, `render.yaml` at the repo root, no production variants, no
  canary in v1. Rollback = previous Render deploy.
- **Frontend / public demo**: **Hugging Face Spaces** (Gradio or
  static HTML Space). The Space hosts the browser ONNX bundle and
  vocab JSON.
- **Training**: local machine, **Kaggle Notebooks**, or **Google
  Colab**. No managed training service.
- **Auth**: JWT (HS256), 15 min TTL, signing secret from a Render env
  var (see ADR 0007).
- **Removed from earlier drafts**: SageMaker endpoints, ECR, CloudFront,
  Terraform, ALB, AWS Secrets Manager, production variants, canary
  deploys.

## Consequences
- Zero cloud-vendor lock-in and zero infrastructure to maintain.
- One config file (`render.yaml`) describes the backend.
- One Git repo (or a mirror) describes the HF Space.
- Browser inference is unaffected -- still ONNX Runtime Web.
- Performance and reliability characteristics are bounded by Render's
  free / starter tier; acceptable for v1.

## Alternatives considered
- Fly.io / Railway: similar profile; Render is chosen for the simplest
  Git-deploy story.
- Self-host on a small VM: more ops burden, no clear win for a
  portfolio project.
- AWS (SageMaker, ECS, Lambda): removed from v1 scope.