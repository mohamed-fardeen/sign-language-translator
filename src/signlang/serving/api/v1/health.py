from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from signlang.serving.schemas.request import HealthResponse, ModelInfo
from signlang.serving.security import require_auth

router = APIRouter(prefix="/v1", tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    registry = getattr(request.app.state, "registry", None)
    if registry is None or not registry.list():
        return HealthResponse(status="starting", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=registry.active_key,
        device=str(next(registry.active.model.parameters()).device),
    )


@router.get("/models", response_model=list[ModelInfo])
def list_models(request: Request, _claims=Depends(require_auth)) -> list[ModelInfo]:
    registry = request.app.state.registry
    out: list[ModelInfo] = []
    for key in registry.list():
        name, _, version = key.partition("@")
        m = registry._models[key]
        out.append(
            ModelInfo(
                name=name,
                version=version,
                backend="torchscript",
                device=str(next(m.model.parameters()).device),
            )
        )
    return out
