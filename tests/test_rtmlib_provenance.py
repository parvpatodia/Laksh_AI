"""RTMPose/rtmlib provenance shape (no model download)."""

from app.pose.provenance import POSE_BASELINE_SCHEMA_VERSION, build_rtmlib_rtmpose_pose_provenance


def test_rtmlib_provenance_keys():
    p = build_rtmlib_rtmpose_pose_provenance(
        ffmpeg_preprocess_applied=True,
        multipass=False,
        rtmlib_mode="lightweight",
        device="cpu",
        to_openpose=False,
        pose_usable_gate_applied={"min_n_frames": 15},
        calibration_record={"calibration_source": "test"},
    )
    assert p["pose_baseline_schema_version"] == POSE_BASELINE_SCHEMA_VERSION
    assert p["canonical_mapping_id"] == "coco17_xy_pixels_normalized_v1"
    assert p["backend_implementation_id"].startswith("rtmlib")
    assert p["rtmlib_mode"] == "lightweight"
    assert p["calibration"]["calibration_source"] == "test"
