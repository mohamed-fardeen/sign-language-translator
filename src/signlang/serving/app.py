from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from signlang.config import load_config, to_dict
from signlang.serving.api.v1 import api_router
from signlang.serving.middleware import StructuredLoggingMiddleware
from signlang.serving.model_registry import ModelRegistry
from signlang.serving.observability import instrument_fastapi, setup_tracing
from signlang.serving.security import JWTConfig
from signlang.utils.io import read_json
from signlang.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

SERVE_ENV = os.environ.get("SERVE_ENV", "local")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = to_dict(load_config([f"serve={SERVE_ENV}"]))
    configure_logging(cfg["serve"]["observability"]["log_level"], json_logs=True)

    app.state.cfg = cfg
    app.state.predict_cfg = {
        "clip_frames": int(cfg["model"].get("clip_frames", 64)),
        "beam_size": int(cfg["model"].get("beam_size", 1)),
    }
    app.state.registry = ModelRegistry()

    secret = os.environ.get(cfg["serve"]["auth"]["jwt_secret_env"], "")
    if not secret:
        secret = "dev-only-secret-change-me"
        log.warning("jwt.using_default_secret", hint="set JWT_SECRET in env")
    app.state.jwt_config = JWTConfig(
        secret=secret,
        algorithm=cfg["serve"]["auth"]["jwt_algorithm"],
        ttl_seconds=int(cfg["serve"]["auth"]["jwt_ttl_seconds"]),
        default_scope=cfg["serve"]["auth"]["default_scope"],
    )

    vocab_path = Path(cfg["paths"]["vocab_path"])
    app.state.vocab = read_json(vocab_path) if vocab_path.exists() else {
        "id_to_gloss": {}, "gloss_to_id": {}
    }

    model_path = os.environ.get("MODEL_PATH", cfg["serve"]["model"]["path"])
    if Path(model_path).exists():
        app.state.registry.load(
            name="signlang",
            version=os.environ.get("MODEL_VERSION", "latest"),
            path=model_path,
            device=cfg["serve"]["model"]["device"],
        )
        log.info("model.ready", path=model_path)
    else:
        log.warning("model.missing", path=model_path)

    log.info("serve.startup.done", env=SERVE_ENV)
    yield
    log.info("serve.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="signlang API",
        version="0.1.0",
        description="ASL isolated-sign recognizer (Transformer + CTC).",
        lifespan=lifespan,
    )

    cfg = to_dict(load_config([f"serve={SERVE_ENV}"]))
    if cfg["serve"]["observability"].get("enable_tracing", True):
        setup_tracing(
            service_name="signlang-api",
            otlp_endpoint=cfg["serve"]["observability"].get("otlp_endpoint") or None,
        )
        instrument_fastapi(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg["serve"]["api"]["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(StructuredLoggingMiddleware)

    app.include_router(api_router)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse(url="/web/")

    web_dir = Path(__file__).resolve().parents[2] / "web"
    if web_dir.exists():
        app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app


app = create_app()
