"""End-to-end test: synthetic clip -> token -> predict -> result.

This test is skipped automatically if the heavy deps (torch, mediapipe)
are not present. It runs the full FastAPI stack with a fake model.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.e2e


def test_browser_to_endpoint_synthetic_clip() -> None:
    os.environ.setdefault("SERVE_ENV", "local")
    os.environ.setdefault("JWT_SECRET", "e2e-test-secret" + "x" * 20)
    import numpy as np
    import torch
    from fastapi.testclient import TestClient

    from signlang.serving.app import create_app
    from signlang.serving.model_registry import ModelRegistry
    from signlang.serving.security import JWTConfig

    app = create_app()
    with TestClient(app) as c:
        class _FakeModel:
            def parameters(self):
                yield torch.zeros(1)

            def __call__(self, pose, lh, rh, face):
                B, T, _ = pose.shape
                logits = torch.zeros(B, T, 501)
                logits[..., 7] = 5.0
                return logits

        c.app.state.registry = ModelRegistry()
        c.app.state.registry._models["signlang@e2e"] = type("M", (), {"model": _FakeModel()})()
        c.app.state.registry._active = "signlang@e2e"
        c.app.state.jwt_config = JWTConfig(
            secret=os.environ["JWT_SECRET"], algorithm="HS256", ttl_seconds=900, default_scope="predict"
        )
        c.app.state.vocab = {"id_to_gloss": {"7": "thanks"}, "gloss_to_id": {"thanks": 7}}
        c.app.state.predict_cfg = {"clip_frames": 64, "beam_size": 1}

        tok = c.post("/v1/auth/token", json={"device_key": "e2e-device-1234"}).json()["access_token"]
        T = 64
        clip = {
            "pose": np.zeros((T, 99), dtype=np.float32).tolist(),
            "lh": np.zeros((T, 63), dtype=np.float32).tolist(),
            "rh": np.zeros((T, 63), dtype=np.float32).tolist(),
            "face": np.zeros((T, 120), dtype=np.float32).tolist(),
            "mask": [True] * T,
        }
        res = c.post(
            "/v1/predict",
            json={"clip": clip, "top_k": 3},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["gloss_label"] == "thanks"
        assert body["model_version"] == "signlang@e2e"
