"""End-to-end tests for the ``/v1`` HTTP surface.

Uses :class:`fastapi.testclient.TestClient` so no network, no MediaPipe.
The tests exercise the same gym fixture shape a browser would POST
after running @mediapipe/tasks-vision locally.
"""
from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module if FastAPI is missing in the minimal env.
pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api.v1 import V1_PREFIX, router as v1_router  # noqa: E402
from app.api.v1.schema import V1_RESPONSE_SCHEMA_VERSION  # noqa: E402

_SQUAT_JOINTS = [
    "left_wrist", "right_wrist", "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Minimal FastAPI app with only the v1 router.

    We deliberately do NOT import :mod:`app.main` because it pulls in
    the ChromaDB lifespan + Gemini client. The v1 router has no
    lifespan dependencies, so a bare ``FastAPI()`` is sufficient.
    """
    app = FastAPI()
    app.include_router(v1_router)
    return TestClient(app)


def _synthetic_squat_payload(n_frames: int = 90, fps: float = 30.0) -> dict:
    frames = []
    for i in range(n_frames):
        hip_y = 0.5 + 0.15 * np.sin(2 * np.pi * i / 30)
        frame: dict = {}
        for j in _SQUAT_JOINTS:
            if "hip" in j:
                frame[j] = {"x": 0.5, "y": float(hip_y), "z": 0.0, "visibility": 0.9}
            elif "knee" in j:
                frame[j] = {"x": 0.5, "y": float(hip_y + 0.15), "z": 0.0, "visibility": 0.9}
            elif "ankle" in j:
                frame[j] = {"x": 0.5, "y": 0.85, "z": 0.0, "visibility": 0.9}
            else:
                frame[j] = {"x": 0.5, "y": 0.3, "z": 0.0, "visibility": 0.9}
        frames.append(frame)
    return {"exercise_id": "back_squat", "fps": fps, "frames": frames}


# ---------- meta endpoints ------------------------------------------------


def test_health_ok(client: TestClient) -> None:
    res = client.get(f"{V1_PREFIX}/health")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "ok"
    assert body["v1_schema_version"] == V1_RESPONSE_SCHEMA_VERSION
    prov = body["provenance"]
    assert prov["pose_baseline_version"]
    assert len(prov["exercise_manifest_sha"]) == 64  # SHA-256 hex
    assert prov["calibration_manifest_sha"]


def test_sports_lists_basketball_and_gym_as_available(client: TestClient) -> None:
    res = client.get(f"{V1_PREFIX}/sports")
    assert res.status_code == 200, res.text
    rows = res.json()
    by_id = {r["id"]: r for r in rows}
    assert by_id["basketball"]["available"] is True
    assert by_id["gym"]["available"] is True
    # Gym carries the exercise list; basketball does not.
    assert len(by_id["gym"]["exercises"]) >= 8
    assert by_id["basketball"]["exercises"] == []
    # Tennis + golf still advertised but unavailable.
    assert by_id["tennis"]["available"] is False
    assert by_id["golf"]["available"] is False


# ---------- analyze/gym ---------------------------------------------------


def test_analyze_gym_squat_returns_v1_envelope(client: TestClient) -> None:
    payload = _synthetic_squat_payload(n_frames=90, fps=30.0)
    res = client.post(f"{V1_PREFIX}/analyze/gym", json=payload)
    assert res.status_code == 200, res.text
    body = res.json()
    # Envelope contract.
    assert body["schema_version"] == V1_RESPONSE_SCHEMA_VERSION
    assert body["sport_id"] == "gym"
    assert body["exercise_id"] == "back_squat"
    assert body["source"] == "frames_json"
    assert body["analysis_mode"] == "canonical_backend"
    assert body["n_frames"] == 90
    assert body["parity_probe"] is None
    # Provenance is populated.
    prov = body["provenance"]
    assert prov["pose_baseline_version"]
    assert len(prov["exercise_manifest_sha"]) == 64
    assert prov["model"] == "none_frames_json"
    # Feature-vector contract.
    fvs = body["feature_vectors"]
    assert len(fvs) >= 1
    first = fvs[0]
    for key in ("rep_index", "start_frame", "end_frame", "peak_frame", "rep_status", "features"):
        assert key in first
    for fname, fval in first["features"].items():
        assert fval["status"] in ("valid", "degraded", "unknown"), fname
        assert "unit" in fval
        assert "reason_codes" in fval
    # Calibration honesty: v0 ships uncalibrated.
    cal = body["calibration"]
    assert cal["evidence_status"] == "uncalibrated_v0"
    for per_rep in cal["per_rep"]:
        for fname, fcal in per_rep["fields"].items():
            assert fcal["status"] in (
                "no_reference_yet",
                "unavailable",
                "within_reference",
                "outside_reference",
            )


def test_analyze_gym_rejects_unknown_exercise(client: TestClient) -> None:
    payload = _synthetic_squat_payload()
    payload["exercise_id"] = "moonwalk_dance"
    res = client.post(f"{V1_PREFIX}/analyze/gym", json=payload)
    assert res.status_code == 400, res.text


def test_analyze_gym_rejects_reserved_token(client: TestClient) -> None:
    payload = _synthetic_squat_payload()
    payload["exercise_id"] = "mixed"
    res = client.post(f"{V1_PREFIX}/analyze/gym", json=payload)
    assert res.status_code == 400, res.text


def test_analyze_gym_rejects_non_positive_fps(client: TestClient) -> None:
    payload = _synthetic_squat_payload()
    payload["fps"] = 0.0
    res = client.post(f"{V1_PREFIX}/analyze/gym", json=payload)
    # pydantic rejects gt=0.0 with 422 (validation error).
    assert res.status_code == 422, res.text


def test_analyze_gym_empty_frames_emits_zero_reps(client: TestClient) -> None:
    payload = {"exercise_id": "back_squat", "fps": 30.0, "frames": []}
    res = client.post(f"{V1_PREFIX}/analyze/gym", json=payload)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["n_frames"] == 0
    assert body["feature_vectors"] == []
    assert body["calibration"]["evidence_status"] == "uncalibrated_v0"


def test_analyze_gym_rejects_extra_top_level_field(client: TestClient) -> None:
    payload = _synthetic_squat_payload()
    payload["unexpected"] = 1  # extra="forbid" on AnalyzeGymRequest
    res = client.post(f"{V1_PREFIX}/analyze/gym", json=payload)
    assert res.status_code == 422, res.text
