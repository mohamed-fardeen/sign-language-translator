from __future__ import annotations

import time

import pytest

from signlang.serving.security import JWTConfig, issue_token, verify_token


def test_issue_and_verify_token() -> None:
    cfg = JWTConfig(secret="x" * 32, algorithm="HS256", ttl_seconds=60, default_scope="predict")
    tok = issue_token(cfg, subject="dev", scope="predict")
    payload = verify_token(cfg, tok)
    assert payload["sub"] == "dev"
    assert payload["scope"] == "predict"


def test_expired_token() -> None:
    cfg = JWTConfig(secret="x" * 32, algorithm="HS256", ttl_seconds=1, default_scope="predict")
    tok = issue_token(cfg, subject="dev")
    time.sleep(1.2)
    with pytest.raises(Exception):
        verify_token(cfg, tok)


def test_invalid_token() -> None:
    cfg = JWTConfig(secret="x" * 32, algorithm="HS256", ttl_seconds=60, default_scope="predict")
    with pytest.raises(Exception):
        verify_token(cfg, "not-a-jwt")
