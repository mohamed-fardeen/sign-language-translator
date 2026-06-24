from __future__ import annotations

from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    device_key: str = Field(min_length=8, max_length=128)
    scope: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class ClipPayload(BaseModel):
    pose: list[list[float]] = Field(..., description="(T, 99) per-frame pose landmarks")
    lh: list[list[float]] = Field(..., description="(T, 63) left-hand landmarks")
    rh: list[list[float]] = Field(..., description="(T, 63) right-hand landmarks")
    mask: list[bool] | None = Field(default=None, description="(T,) valid-frame mask")


class PredictRequest(BaseModel):
    clip: ClipPayload
    top_k: int = Field(default=5, ge=1, le=50)
    # CTC: ``beam_size`` removed. v1 uses single-label classification; argmax
    # over the (B, num_classes) logits is the only decoding strategy.


class PredictionItem(BaseModel):
    id: int
    label: str
    prob: float


class PredictResponse(BaseModel):
    gloss_id: int
    gloss_label: str
    confidence: float
    top_k: list[PredictionItem]
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    device: str | None = None


class ModelInfo(BaseModel):
    name: str
    version: str
    backend: str
    device: str
