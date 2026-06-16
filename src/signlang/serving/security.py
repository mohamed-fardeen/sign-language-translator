from __future__ import annotations

import time
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from signlang.utils.logging import get_logger

log = get_logger(__name__)


class JWTConfig(BaseModel):
    secret: str
    algorithm: str = "HS256"
    ttl_seconds: int = 900
    default_scope: str = "predict"


_bearer = HTTPBearer(auto_error=False)


def issue_token(
    cfg: JWTConfig,
    subject: str,
    scope: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + cfg.ttl_seconds,
        "scope": scope or cfg.default_scope,
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, cfg.secret, algorithm=cfg.algorithm)


def verify_token(cfg: JWTConfig, token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, cfg.secret, algorithms=[cfg.algorithm])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {exc}")


def get_jwt_config(request: Request) -> JWTConfig:
    cfg = getattr(request.app.state, "jwt_config", None)
    if cfg is None:
        raise HTTPException(status_code=500, detail="JWT config not initialised")
    return cfg


def require_auth(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    cfg = get_jwt_config(request)
    payload = verify_token(cfg, creds.credentials)
    return payload
