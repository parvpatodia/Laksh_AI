"""Tests for scripts/analyze_gym_clip.py and app.gym.pose_adapter.

All tests use the --frames-json path so MediaPipe is not required.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from app.gym.pose_adapter import (
    frames_json_to_canonical_frames,
    raw_2d_to_canonical_frames,
)
from app.pose.canonical import JointObservation

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_gym_clip.py"
DEFAULT_CAL = REPO_ROOT / "evaluation" / "gym_calibration_v0.json"

# Joints needed to construct a minimal squat signal (cyclic_vertical: right_hip.y)
_SQUAT_JOINTS = [
    "left_wrist", "right_wrist", "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _squat_frames_payload(n_frames: int = 90, fps: float = 30.0) -> dict:
    """Build a --frames-json payload with synthetic squat hip-y oscillation."""
    frames = []
    for i in range(n_frames):
        # hip y oscillates 0.5 +/- 0.15 at ~1 Hz (period=30 frames @ 30fps)
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
    return {"fps": fps, "frames": frames}


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


# ---------- pose_adapter unit tests ---------------------------------------


def test_raw_2d_to_canonical_frames_basic() -> None:
    """Valid 3-frame raw_2d produces 3 frames with JointObservation values."""
    raw_2d = {
        "left_shoulder": np.array([[0.4, 0.3, 0.9], [0.4, 0.3, 0.9], [0.4, 0.3, 0.9]]),
        "right_shoulder": np.array([[0.6, 0.3, 0.85], [0.6, 0.3, 0.85], [np.nan, np.nan, np.nan]]),
        "left_hip": np.array([[0.45, 0.55, 0.88], [0.45, 0.55, 0.88], [0.45, 0.55, 0.88]]),
    }
    frames = raw_2d_to_canonical_frames(raw_2d)
    assert len(frames) == 3
    # Frame 0: all joints present
    assert frames[0] is not None
    assert "left_shoulder" in frames[0]
    assert isinstance(frames[0]["left_shoulder"], JointObservation)
    # Frame 2: right_shoulder NaN -> absent from dict but others present
    assert frames[2] is not None
    assert "right_shoulder" not in frames[2]
    assert "left_hip" in frames[2]


def test_raw_2d_to_canonical_frames_all_nan_yields_none() -> None:
    """A frame where every joint is NaN produces None."""
    raw_2d = {
        "left_shoulder": np.array([[np.nan, np.nan, np.nan]]),
        "right_shoulder": np.array([[np.nan, np.nan, np.nan]]),
    }
    frames = raw_2d_to_canonical_frames(raw_2d)
    assert frames[0] is None


def test_raw_2d_to_canonical_frames_empty() -> None:
    frames = raw_2d_to_canonical_frames({})
    assert frames == []


def test_frames_json_to_canonical_frames_basic() -> None:
    raw = [
        {"left_shoulder": {"x": 0.4, "y": 0.3, "z": 0.0, "visibility": 0.9}},
        None,
        {"right_hip": {"x": 0.5, "y": 0.6, "z": 0.0, "visibility": 0.8}},
    ]
    frames = frames_json_to_canonical_frames(raw)
    assert len(frames) == 3
    assert frames[0] is not None
    assert "left_shoulder" in frames[0]
    assert frames[1] is None
    assert frames[2] is not None


def test_frames_json_to_canonical_frames_drops_nan() -> None:
    raw = [{"left_shoulder": {"x": float("nan"), "y": 0.3, "z": 0.0, "visibility": 0.9}}]
    frames = frames_json_to_canonical_frames(raw)
    assert frames[0] is None  # nan x -> joint dropped -> empty dict -> None


def test_frames_json_to_canonical_frames_ignores_bad_obs() -> None:
    raw = [{"left_shoulder": "not_a_dict"}]
    frames = frames_json_to_canonical_frames(raw)
    assert frames[0] is None


# ---------- analyze_gym_clip.py CLI tests ---------------------------------


def test_script_requires_exercise_id() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
        json.dump({"fps": 30.0, "frames": []}, f)
        f.flush()
        res = _run_script("--frames-json", f.name)
    assert res.returncode != 0


def test_script_unknown_exercise_id_exits_1() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
        json.dump({"fps": 30.0, "frames": []}, f)
        f.flush()
        res = _run_script(
            "--exercise-id", "moonwalk_dance",
            "--frames-json", f.name,
        )
    assert res.returncode == 1, res.stderr


def test_script_reserved_token_exits_1() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
        json.dump({"fps": 30.0, "frames": []}, f)
        f.flush()
        res = _run_script(
            "--exercise-id", "mixed",
            "--frames-json", f.name,
        )
    assert res.returncode == 1


def test_script_missing_frames_json_exits_2() -> None:
    res = _run_script(
        "--exercise-id", "back_squat",
        "--frames-json", "/tmp/__no_such_file__.json",
    )
    assert res.returncode == 2


def test_script_empty_frames_produces_zero_reps() -> None:
    payload = {"fps": 30.0, "frames": []}
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(payload, f)
        fname = f.name
    try:
        res = _run_script("--exercise-id", "back_squat", "--frames-json", fname)
    finally:
        Path(fname).unlink(missing_ok=True)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    # Without --out, stdout is the full result JSON.
    assert "feature_vectors" in out
    assert len(out["feature_vectors"]) == 0  # no frames -> no reps


def test_script_squat_frames_json_output_shape() -> None:
    """Full pipeline with synthetic squat frames; verify output JSON structure."""
    payload = _squat_frames_payload(n_frames=90, fps=30.0)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(payload, f)
        fname = f.name

    out_path = Path(fname).with_suffix(".out.json")
    try:
        res = _run_script(
            "--exercise-id", "back_squat",
            "--frames-json", fname,
            "--out", str(out_path),
            "--pretty",
        )
        assert res.returncode == 0, res.stderr + res.stdout
        # Script prints a single-line summary to stdout when --out is used.
        summary = json.loads(res.stdout)
        assert summary["ok"] is True
        assert summary["exercise_id"] == "back_squat"

        result = json.loads(out_path.read_text())
        # Top-level keys
        assert result["schema_version"] == "1.0.0"
        assert result["exercise_id"] == "back_squat"
        assert result["source"] == "frames_json"
        assert result["fps"] == 30.0
        assert result["n_frames"] == 90
        # Segment block
        assert "reps" in result["segment"]
        # Feature vectors
        assert isinstance(result["feature_vectors"], list)
        for fv in result["feature_vectors"]:
            assert "rep_index" in fv
            assert "features" in fv
            assert "rep_duration_s" in fv["features"]
        # Calibration block
        assert result["calibration"]["exercise_id"] == "back_squat"
        assert result["calibration"]["evidence_status"] == "uncalibrated_v0"
        for per_rep in result["calibration"]["per_rep"]:
            assert "rep_index" in per_rep
            for field_name, field_cal in per_rep["fields"].items():
                assert field_cal["status"] in (
                    "no_reference_yet", "unavailable", "within_reference", "outside_reference"
                )
    finally:
        Path(fname).unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def test_script_plank_frames_json_runs() -> None:
    """Plank (duration exercise) should succeed with zero cyclic reps."""
    # Flat signal: plank hold => no cyclic peaks => single duration span
    frames = []
    for _ in range(60):
        frame = {j: {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9} for j in _SQUAT_JOINTS}
        frames.append(frame)
    payload = {"fps": 30.0, "frames": frames}

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(payload, f)
        fname = f.name
    try:
        res = _run_script("--exercise-id", "plank", "--frames-json", fname)
        assert res.returncode == 0, res.stderr
        out = json.loads(res.stdout)
        assert out["exercise_id"] == "plank"
    finally:
        Path(fname).unlink(missing_ok=True)


def test_script_stdout_is_valid_json() -> None:
    """When --out is omitted the full result goes to stdout."""
    payload = _squat_frames_payload(n_frames=60, fps=30.0)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(payload, f)
        fname = f.name
    try:
        res = _run_script("--exercise-id", "back_squat", "--frames-json", fname)
        assert res.returncode == 0, res.stderr
        parsed = json.loads(res.stdout)
        assert parsed["schema_version"] == "1.0.0"
    finally:
        Path(fname).unlink(missing_ok=True)
