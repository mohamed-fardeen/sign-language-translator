from __future__ import annotations

from fastapi import APIRouter, Request

from signlang.serving.schemas.request import TokenRequest, TokenResponse
from signlang.serving.security import get_jwt_config, issue_token

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
def issue(req: TokenRequest, request: Request) -> TokenResponse:
    cfg = get_jwt_config(request)
    if len(req.device_key) < 8:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="device_key too short")
    token = issue_token(cfg, subject=req.device_key[:32], scope=req.scope)
    return TokenResponse(access_token=token, expires_in=cfg.ttl_seconds)
