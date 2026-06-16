from __future__ import annotations

from fastapi import APIRouter

from signlang.serving.api.v1 import auth, health, predict

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(health.router)
api_router.include_router(predict.router)
