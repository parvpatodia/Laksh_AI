"""Provenance dict shape for pose baseline (no OpenCV required)."""
from pathlib import Path

import pytest

from app.pose.provenance import POSE_BASELINE_SCHEMA_VERSION, build_mediapipe_pose_provenance


def test_provenance_required_keys():
    p = build_mediapipe_pose_provenance(
        ffmpeg_preprocess_applied=False,
        multipass=False,
        pose_usable_gate_applied={"min_detection_rate": 0.25, "min_visibility_core_when_detected": 0.35, "min_n_frames": 15},
        calibration_record={"calibration_source": "test"},
    )
    assert p["pose_baseline_schema_version"] == POSE_BASELINE_SCHEMA_VERSION
    assert p["canonical_joint_schema_version"]
    assert p["canonical_joint_set"] == "coco_17_names"
    assert p["canonical_mapping_id"] == "mediapipe_blazepose33_v1"
    assert p["ffmpeg_preprocess_applied"] is False
    assert p["frame_preprocess_multipass"] is False
    assert "landmarker_options" in p
    assert p["landmarker_options"]["num_poses"] >= 1
    assert p["pose_usable_gate_applied"]["min_n_frames"] == 15
    assert p["calibration"]["calibration_source"] == "test"
    assert "mediapipe_package_version" in p
    assert "pose_model_sha256" in p or p.get("pose_model_status") == "missing"
    assert p["platform_sys"]


def test_provenance_multipass_flag():
    p = build_mediapipe_pose_provenance(ffmpeg_preprocess_applied=True, multipass=True)
    assert p["ffmpeg_preprocess_applied"] is True
    assert p["frame_preprocess_multipass"] is True


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "pose_landmarker_heavy.task").is_file(),
    reason="pose model not present in checkout",
)
def test_provenance_model_hash_when_asset_present():
    p = build_mediapipe_pose_provenance(ffmpeg_preprocess_applied=False, multipass=False)
    assert p.get("pose_model_status") == "ok"
    sha = p.get("pose_model_sha256")
    assert isinstance(sha, str)
    # Full-file hash is 64 hex chars; oversized assets use a truncated digest string.
    assert len(sha) == 64 or ":truncated_" in sha
