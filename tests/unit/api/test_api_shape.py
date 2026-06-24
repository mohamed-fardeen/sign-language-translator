from __future__ import annotations

from fastapi.testclient import TestClient

from signlang.serving.app import create_app
from signlang.serving.model_registry import ModelRegistry
from signlang.serving.security import JWTConfig


def test_predict_shape_mismatch_returns_422(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SERVE_ENV", "local")
    monkeypatch.setenv("JWT_SECRET", "test-secret" + "x" * 20)
    app = create_app()
    with TestClient(app) as c:
        import torch

        class _FakeModel:
            def parameters(self):
                yield torch.zeros(1)

            def __call__(self, pose, lh, rh):
                B, T, _ = pose.shape
                # v1: classification logits of shape (B, num_classes)
                return torch.zeros(B, 501)

        c.app.state.registry = ModelRegistry()
        c.app.state.registry._models["signlang@test"] = type("M", (), {"model": _FakeModel()})()
        c.app.state.registry._active = "signlang@test"
        c.app.state.jwt_config = JWTConfig(secret="test-secret" + "x" * 20, algorithm="HS256", ttl_seconds=900, default_scope="predict")
        c.app.state.vocab = {"id_to_gloss": {}, "gloss_to_id": {}}
        c.app.state.predict_cfg = {"clip_frames": 64, "beam_size": 1}

        tok = c.post("/v1/auth/token", json={"device_key": "abcdefgh"}).json()["access_token"]
        bad = {
            "clip": {
                "pose": [[[0.0] * 50]],
                "lh": [[[0.0] * 63]],
                "rh": [[[0.0] * 63]],
                "mask": [True],
            }
        }
        res = c.post("/v1/predict", json=bad, headers={"Authorization": f"Bearer {tok}"})
        assert res.status_code == 422
