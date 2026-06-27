"""Integration tests for POST /v1/coaching/ground.

Overrides the You.com client dependency with a mocked-transport client (grounded
path) and a no-key client (fallback path) -- no live key or network needed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api.v1 import router as v1_router  # noqa: E402
from app.api.v1.deps import get_you_client  # noqa: E402
from app.coaching.you_search import YouComClient  # noqa: E402

_YOU_RESPONSE = {
    "results": {
        "web": [
            {
                "title": "NSCA eccentric tempo guidelines",
                "url": "https://example.org/nsca-tempo",
                "description": "Controlled eccentric loading.",
                "snippets": ["Aim for a 2-4s eccentric."],
            }
        ]
    }
}


def _transport(payload):
    def _t(url, headers, params):
        return payload

    return _t


def _client_with(you_client: YouComClient) -> TestClient:
    app = FastAPI()
    app.include_router(v1_router)
    app.dependency_overrides[get_you_client] = lambda: you_client
    return TestClient(app)


def test_ground_endpoint_grounded_path():
    you = YouComClient(api_key="k", transport=_transport(_YOU_RESPONSE))
    res = _client_with(you).post(
        "/v1/coaching/ground",
        json={
            "exercise_id": "back_squat",
            "faults": [
                {"id": "tempo-fast-eccentric", "title": "Control the lowering phase", "cue": "Slow descent."}
            ],
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["grounding_enabled"] is True
    assert body["backend"] == "you.com"
    cue = body["cues"][0]
    assert cue["grounded"] is True
    assert cue["cue"] == "Slow descent."
    assert cue["citations"][0]["url"].startswith("https://")
    assert "you_com_grounded" in cue["reason_codes"]


def test_ground_endpoint_fallback_without_key():
    you = YouComClient(api_key=None)
    res = _client_with(you).post(
        "/v1/coaching/ground",
        json={
            "exercise_id": "back_squat",
            "faults": [{"id": "vis-low", "title": "Framing", "cue": "Fix lighting."}],
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["grounding_enabled"] is False
    assert body["backend"] == "fallback"
    assert body["cues"][0]["grounded"] is False
    assert body["cues"][0]["cue"] == "Fix lighting."


def test_ground_endpoint_validates_body():
    res = _client_with(YouComClient(api_key=None)).post(
        "/v1/coaching/ground", json={"faults": []}
    )
    assert res.status_code == 422
