from __future__ import annotations

import time
from collections.abc import Callable

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            path=request.url.path, method=request.method
        )
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            structlog.get_logger().exception("http.error")
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        structlog.get_logger().info(
            "http.request",
            status=response.status_code,
            latency_ms=round(elapsed_ms, 3),
        )
        return response
