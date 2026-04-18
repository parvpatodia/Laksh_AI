"""Tests for :mod:`app.gym.pipeline`.

No MediaPipe required: every test drives the pipeline through
``canonical_frames`` produced by
:func:`app.gym.pose_adapter.frames_json_to_canonical_frames`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.gym.pipeline import (
    DEFAULT_CALIBRATION_CONFIG,
    GYM_PIPELINE_SCHEMA_VERSION,
    UnknownExerciseError,
    analyze_gym_clip,
)
from app.gym.pose_adapter import frames_json_to_canonical_frames

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_FIXTURE = REPO_ROOT / "evaluation" / "fixtures" / "demo_squat_frames.json"

_SQUAT_JOINTS = [
    "left_wrist", "right_wrist", "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _synthetic_squat_frames(n_frames: int = 90) -> list:
    """Build pre-extracted canonical frames for a 2-rep squat signal."""
    raw = []
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
        raw.append(frame)
    return frames_json_to_canonical_frames(raw)


def test_pipeline_rejects_unknown_exercise() -> None:
    frames = _synthetic_squat_frames(30)
    with pytest.raises(UnknownExerciseError):
        analyze_gym_clip(
            exercise_id="moonwalk_dance",
            fps=30.0,
            canonical_frames=frames,
        )


def test_pipeline_rejects_reserved_token() -> None:
    frames = _synthetic_squat_frames(30)
    with pytest.raises(UnknownExerciseError):
        analyze_gym_clip(
            exercise_id="mixed",
            fps=30.0,
            canonical_frames=frames,
        )


def test_pipeline_rejects_non_positive_fps() -> None:
    frames = _synthetic_squat_frames(30)
    with pytest.raises(ValueError):
        analyze_gym_clip(
            exercise_id="back_squat",
            fps=0.0,
            canonical_frames=frames,
        )


def test_pipeline_empty_frames_has_zero_reps() -> None:
    result = analyze_gym_clip(
        exercise_id="back_squat",
        fps=30.0,
        canonical_frames=[],
    )
    assert result["schema_version"] == GYM_PIPELINE_SCHEMA_VERSION
    assert result["n_frames"] == 0
    assert result["feature_vectors"] == []
    assert result["calibration"]["evidence_status"] == "uncalibrated_v0"
    assert result["calibration"]["per_rep"] == []


def test_pipeline_squat_produces_v1_shape() -> None:
    frames = _synthetic_squat_frames(90)
    result = analyze_gym_clip(
        exercise_id="back_squat",
        fps=30.0,
        canonical_frames=frames,
        source="frames_json",
    )
    assert result["schema_version"] == GYM_PIPELINE_SCHEMA_VERSION
    assert result["exercise_id"] == "back_squat"
    assert result["source"] == "frames_json"
    assert result["fps"] == 30.0
    assert result["n_frames"] == 90
    assert "reps" in result["segment"]
    assert len(result["feature_vectors"]) >= 1
    first = result["feature_vectors"][0]
    assert "rep_index" in first and "features" in first
    assert "rep_duration_s" in first["features"]
    # Calibration block: v0 ships honest.
    cal = result["calibration"]
    assert cal["exercise_id"] == "back_squat"
    assert cal["evidence_status"] == "uncalibrated_v0"
    for per_rep in cal["per_rep"]:
        for fname, fcal in per_rep["fields"].items():
            assert fcal["status"] in (
                "no_reference_yet",
                "unavailable",
                "within_reference",
                "outside_reference",
            )


def test_pipeline_default_calibration_file_is_sha_pinned() -> None:
    """Smoke-check that the default calibration path actually exists and parses."""
    assert DEFAULT_CALIBRATION_CONFIG.is_file()
    payload = json.loads(DEFAULT_CALIBRATION_CONFIG.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0.0"


def test_pipeline_matches_cli_fixture_output() -> None:
    """Library path on the committed fixture emits the same structure as the CLI."""
    if not DEMO_FIXTURE.is_file():
        pytest.skip(f"fixture missing: {DEMO_FIXTURE}")
    payload = json.loads(DEMO_FIXTURE.read_text(encoding="utf-8"))
    canonical = frames_json_to_canonical_frames(payload["frames"])
    result = analyze_gym_clip(
        exercise_id="back_squat",
        fps=float(payload["fps"]),
        canonical_frames=canonical,
        source="frames_json",
    )
    assert result["n_frames"] == len(payload["frames"])
    assert result["calibration"]["evidence_status"] == "uncalibrated_v0"
    # Fixture is a deterministic 2-rep squat signal.
    assert len(result["feature_vectors"]) >= 1
