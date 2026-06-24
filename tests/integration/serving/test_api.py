from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from signlang.serving.app import create_app
from signlang.serving.model_registry import ModelRegistry
from signlang.serving.security import JWTConfig


class _FakeModel:
    """Fake model that mimics v1 single-label classification.

    Returns logits of shape (B, num_classes). Class index 0 is pushed
    to a high value so ``argmax + 1 = 1`` maps to the first manifest
    label, which in the test vocab is "hello".
    """

    def __init__(self) -> None:
        import torch

        self._t = torch

    def parameters(self):
        yield self._t.zeros(1)

    def eval(self):
        return self

    def __call__(self, pose, lh, rh):
        B, T, _ = pose.shape
        num_classes = 184  # matches the vocab in tests/conftest.py
        logits = self._t.zeros(B, num_classes)
        logits[..., 0] = 5.0  # class 0 -> manifest label 1
        return logits


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("SERVE_ENV", "local")
    monkeypatch.setenv("JWT_SECRET", "test-secret" + "x" * 20)
    app = create_app()
    with TestClient(app) as c:
        registry: ModelRegistry = c.app.state.registry
        fake = _FakeModel()
        registry._models["signlang@test"] = type("M", (), {"model": fake})()
        registry._active = "signlang@test"
        c.app.state.jwt_config = JWTConfig(
            secret="test-secret" + "x" * 20,
            algorithm="HS256",
            ttl_seconds=900,
            default_scope="predict",
        )
        c.app.state.vocab = {
            "id_to_gloss": {"1": "hello"},
            "gloss_to_id": {"hello": 1},
        }
        c.app.state.predict_cfg = {"clip_frames": 64, "beam_size": 1}
        yield c


def _clip_payload() -> dict:
    T = 64
    return {
        "clip": {
            "pose": np.zeros((T, 99), dtype=np.float32).tolist(),
            "lh": np.zeros((T, 63), dtype=np.float32).tolist(),
            "rh": np.zeros((T, 63), dtype=np.float32).tolist(),
            "mask": [True] * T,
        },
        "top_k": 3,
    }


def test_health_endpoint(client: TestClient) -> None:
    res = client.get("/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_token_and_predict_round_trip(client: TestClient) -> None:
    res = client.post("/v1/auth/token", json={"device_key": "abcdefgh"})
    assert res.status_code == 200
    token = res.json()["access_token"]

    res = client.post(
        "/v1/predict",
        json=_clip_payload(),
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["gloss_id"] == 1
    assert body["gloss_label"] == "hello"
    assert body["model_version"] == "signlang@test"


def test_predict_requires_auth(client: TestClient) -> None:
    res = client.post("/v1/predict", json=_clip_payload())
    assert res.status_code == 401