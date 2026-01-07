import importlib

import pytest

from app.api import deps


EXPECTED_PATHS = {
    "/api/geo/score",
    "/api/rewrite",
    "/api/brand-brief/validate",
    "/api/cot/stage1/run",
    "/api/billing/checkout",
    "/api/billing/webhook",
    "/api/quota/me",
}


def test_import_app():
    module = importlib.import_module("api_brand")
    assert hasattr(module, "app")


def test_routes_exist(client):
    from api_brand import app

    paths = {route.path for route in app.routes}
    for expected in EXPECTED_PATHS:
        assert expected in paths


def test_quota_me_no_auth_behavior(client):
    response = client.get("/api/quota/me")
    assert response.status_code == 200
    payload = response.json()
    assert "ok" in payload
    if not payload.get("ok"):
        assert payload.get("error")


def test_geo_score_triggers_quota_consume(client, monkeypatch):
    from api_brand import app

    consume_calls: list[dict] = []

    def fake_consume_tokens(**kwargs):
        consume_calls.append(kwargs)
        return {
            "ok": True,
            "tokens_limit": kwargs.get("tokens_total", 0),
            "tokens_used": 0,
            "tokens_remaining": kwargs.get("tokens_total", 0),
            "yyyymm": "209912",
        }

    def fake_require_user():
        return {"user_id": "test-user", "tier": "alpha_base"}

    def fake_geo_score_pipeline(**kwargs):
        return {
            "ok": True,
            "geo_score": 95.0,
            "grade": "A",
            "summary": "ok",
            "sealed_overall": 95.0,
            "sealed": {"overall_score": 95.0, "metrics": []},
            "latency_ms": 1,
            "model_used": "stub-model",
            "user_tier": kwargs.get("user_tier", "free"),
            "raw_scores_0_1": {},
            "raw_scores_0_100": {},
        }

    monkeypatch.setattr("routers.geo_router._supabase_rpc_consume_tokens", fake_consume_tokens)
    monkeypatch.setattr("routers.geo_router.geo_score_pipeline", fake_geo_score_pipeline)

    app.dependency_overrides[deps.require_authed_user] = fake_require_user

    payload = {
        "user_question": "Question",
        "article_title": "Title",
        "source_text": "Original text",
        "rewritten_text": "Rewritten text",
        "model_ui": "groq",
        "model_name": "llama-3.3-70b-versatile",
        "samples": 1,
    }

    response = client.post("/api/geo/score", json=payload)

    app.dependency_overrides.pop(deps.require_authed_user, None)

    assert response.status_code == 200
    data = response.json()
    assert "ok" in data
    assert len(consume_calls) >= 1
