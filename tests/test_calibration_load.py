"""Gym pose usable-gate calibration JSON."""
import json
from pathlib import Path

import pytest

from app.pose.calibration import (
    DEFAULT_CALIBRATION_PATH,
    GymPoseUsableGate,
    load_gym_pose_usable_gate,
)


def test_default_repo_calibration_loads():
    gate, rec = load_gym_pose_usable_gate()
    assert isinstance(gate, GymPoseUsableGate)
    assert gate.min_detection_rate == 0.25
    assert rec["calibration_file_sha256"] is not None
    assert rec["calibration_source"] == "evaluation/gym_pose_calibration.json"


def test_invalid_json_falls_back(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    gate, rec = load_gym_pose_usable_gate(path=bad)
    assert gate.min_n_frames == 15
    assert "fallback_invalid_json" in rec["calibration_source"]


def test_root_not_dict_falls_back(tmp_path: Path):
    p = tmp_path / "arr.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    gate, rec = load_gym_pose_usable_gate(path=p)
    assert rec["calibration_source"] == "builtin_defaults_fallback_invalid_root_type"
    assert gate == GymPoseUsableGate(0.25, 0.35, 15)


def test_out_of_range_rejected(tmp_path: Path):
    p = tmp_path / "cal.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pose_usable_heuristic": {
                    "min_detection_rate": 2.0,
                    "min_visibility_core_when_detected": 0.5,
                    "min_n_frames": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    gate, rec = load_gym_pose_usable_gate(path=p)
    assert rec["calibration_source"] == "builtin_defaults_fallback_invalid_schema"
    assert gate == GymPoseUsableGate(0.25, 0.35, 15)


@pytest.mark.skipif(not DEFAULT_CALIBRATION_PATH.is_file(), reason="calibration file not in tree")
def test_committed_calibration_matches_builtin_numbers():
    gate, _ = load_gym_pose_usable_gate()
    assert gate.min_detection_rate == 0.25
    assert gate.min_visibility_core_when_detected == 0.35
    assert gate.min_n_frames == 15
