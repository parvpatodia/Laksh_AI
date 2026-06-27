"""Integration tests for persistence + ``GET /v1/leaderboard``.

Mounts only the v1 router (no app.main lifespan), overrides the store
dependency with a temp-dir SQLite store, runs the real gym pipeline on the
demo fixture, and asserts the analysis is persisted and ranked.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api.v1 import router as v1_router  # noqa: E402
from app.api.v1.deps import get_store  # noqa: E402
from app.persistence.store import SqliteSessionStore  # noqa: E402

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "fixtures"
    / "demo_squat_frames.json"
)


@pytest.fixture
def client(tmp_path):
    store = SqliteSessionStore(db_path=tmp_path / "lb.db")
    app = FastAPI()
    app.include_router(v1_router)
    app.dependency_overrides[get_store] = lambda: store
    return TestClient(app)


def _frames_body(display_name: str | None = None) -> dict:
    data = json.loads(_FIXTURE.read_text())
    body = {"exercise_id": "back_squat", "fps": data["fps"], "frames": data["frames"]}
    if display_name is not None:
        body["display_name"] = display_name
    return body


def test_analyze_persists_then_leaderboard_returns_it(client):
    r = client.post("/v1/analyze/gym", json=_frames_body("parv"))
    assert r.status_code == 200, r.text

    lb = client.get("/v1/leaderboard", params={"exercise_id": "back_squat"})
    assert lb.status_code == 200
    body = lb.json()
    assert body["backend"] == "sqlite"
    assert body["count"] == 1
    entry = body["entries"][0]
    assert entry["display_name"] == "parv"
    assert entry["rank"] == 1
    assert entry["exercise_id"] == "back_squat"
    assert entry["form_index"] is not None
    assert "uncalibrated" in body["disclaimer"].lower()


def test_leaderboard_empty_before_any_analysis(client):
    lb = client.get("/v1/leaderboard")
    assert lb.status_code == 200
    assert lb.json()["count"] == 0


def test_analysis_without_name_defaults_to_anon(client):
    assert client.post("/v1/analyze/gym", json=_frames_body()).status_code == 200
    entry = client.get("/v1/leaderboard").json()["entries"][0]
    assert entry["display_name"] == "anon"


def test_leaderboard_limit_param_is_validated(client):
    assert client.get("/v1/leaderboard", params={"limit": 0}).status_code == 422
    assert client.get("/v1/leaderboard", params={"limit": 101}).status_code == 422
