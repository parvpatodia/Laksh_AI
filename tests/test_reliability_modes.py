import math
import numpy as np

from app.correction_engine import generate_correction_video
from app.physics_engine import KinematicAnalyzer


def _make_raw_pose(n: int = 6, nan_tail: int = 0):
    joints = ["wrist", "elbow", "shoulder", "hip", "knee", "ankle"]
    sides = ["left", "right"]
    raw_2d = {}
    raw_3d = {}
    for s in sides:
        for j in joints:
            arr2 = []
            arr3 = []
            for i in range(n):
                x = 0.45 + 0.01 * i
                y = 0.65 - 0.02 * i
                if i >= n - nan_tail and j == "wrist":
                    arr2.append([np.nan, np.nan, np.nan])
                else:
                    arr2.append([x, y, 0.8])
                arr3.append([x, y, 0.1])
            raw_2d[f"{s}_{j}"] = np.array(arr2, dtype=np.float64)
            raw_3d[f"{s}_{j}"] = np.array(arr3, dtype=np.float64)
    return raw_3d, raw_2d


def test_fallback_exposes_reason_codes(monkeypatch):
    monkeypatch.setattr(KinematicAnalyzer, "_prepare_video", lambda self: self.video_path)
    monkeypatch.setattr(KinematicAnalyzer, "_init_pose", lambda self: False)

    out = KinematicAnalyzer("missing.mp4").analyze()
    assert out["analysis_mode"] == "fallback"
    assert "pose_init_failed" in out.get("fallback_reason_codes", [])
    assert "metric_status" in out
    assert out["metric_status"]["knee_angle"]["source"] == "unavailable"


def test_partial_mode_returns_metric_status(monkeypatch):
    monkeypatch.setattr(KinematicAnalyzer, "_prepare_video", lambda self: self.video_path)
    monkeypatch.setattr(KinematicAnalyzer, "_init_pose", lambda self: True)

    def fake_extract(self, variant, start_sec=None, end_sec=None, video_path_override=None):
        # Slightly sparse detections to force partial mode.
        raw_3d, raw_2d = _make_raw_pose(n=7, nan_tail=4)
        return 30.0, raw_3d, raw_2d, 1

    monkeypatch.setattr(KinematicAnalyzer, "_extract_frames_with_variant", fake_extract)

    out = KinematicAnalyzer("missing.mp4").analyze()
    assert out["analysis_mode"] == "partial"
    assert isinstance(out.get("metric_status"), dict)
    assert "shot_arc_deg" in out["metric_status"]
    assert out["metric_status"]["shot_arc_deg"]["source"] in {"measured", "predicted", "unavailable"}
    # Sparse detections should produce conservative confidence, not static high values.
    assert out["metric_status"]["release_velocity_mps"]["confidence"] <= 0.7


def test_correction_engine_projected_mode_renders():
    telemetry = {
        "fps": 30.0,
        "frames": [],
        "dip": {
            "time_sec": 0.8,
            "joints": {
                "wrist": [0.55, 0.48],
                "elbow": [0.52, 0.56],
                "shoulder": [0.50, 0.62],
                "hip": [0.50, 0.73],
                "knee": [0.51, 0.84],
                "ankle": [0.52, 0.95],
            },
        },
        "release": {},
    }
    stats = {
        "knee_angle": 145.0,
        "elbow_angle": 165.0,
        "shot_arc_deg": 47.0,
        "balance_index": 82.0,
        "hip_rotation_deg": 6.0,
    }
    out = generate_correction_video(telemetry, stats, athlete_name="Tester")
    assert out is not None
    assert out["render_mode"] == "projected"
    assert len(out["video_bytes"]) > 1024


def _make_minimal_motion_pose(n: int = 32):
    """Free-throw-like: tiny wrist/hip excursion so pipeline picks set_shot."""
    raw_2d = {}
    raw_3d = {}
    joints = ["wrist", "elbow", "shoulder", "hip", "knee", "ankle"]
    sides = ["left", "right"]
    for s in sides:
        for j in joints:
            arr2, arr3 = [], []
            for i in range(n):
                x = 0.45 + 0.001 * i
                if j == "wrist":
                    y = 0.52 + 0.002 * math.sin(2 * math.pi * i / max(1, n - 1))
                elif j == "hip":
                    y = 0.74 + 0.0008 * math.sin(2 * math.pi * i / max(1, n - 1))
                else:
                    y = 0.60 + 0.01 * (i / max(1, n - 1))
                arr2.append([x, y, 0.85])
                arr3.append([x, y, 0.08])
            key = f"{s}_{j}"
            raw_2d[key] = np.array(arr2, dtype=np.float64)
            raw_3d[key] = np.array(arr3, dtype=np.float64)
    return raw_3d, raw_2d


def test_knee_elbow_predicted_when_world_missing_but_2d_valid(monkeypatch):
    monkeypatch.setattr(KinematicAnalyzer, "_prepare_video", lambda self: self.video_path)
    monkeypatch.setattr(KinematicAnalyzer, "_init_pose", lambda self: True)
    monkeypatch.setattr(
        KinematicAnalyzer,
        "apply_filters",
        lambda self, data: {k: v.copy() for k, v in data.items()},
    )

    n = 24
    raw_3d, raw_2d = _make_raw_pose(n=n, nan_tail=0)
    # Non-degenerate 2D angles (hip–knee–ankle and shoulder–elbow–wrist)
    for i in range(n):
        raw_2d["right_hip"][i] = [0.50, 0.70, 0.9]
        raw_2d["right_knee"][i] = [0.52, 0.82, 0.9]
        raw_2d["right_ankle"][i] = [0.54, 0.94, 0.9]
        raw_2d["right_shoulder"][i] = [0.48, 0.55, 0.9]
        raw_2d["right_elbow"][i] = [0.50, 0.62, 0.9]
        raw_2d["right_wrist"][i] = [0.52, 0.50, 0.9]
        raw_3d["right_hip"][i] = [0.50, 0.70, 0.1]
        raw_3d["right_ankle"][i] = [0.54, 0.94, 0.1]
        raw_3d["right_shoulder"][i] = [0.48, 0.55, 0.1]
        raw_3d["right_wrist"][i] = [0.52, 0.50, 0.1]
    raw_3d["right_knee"][:, :] = np.nan
    raw_3d["right_elbow"][:, :] = np.nan

    def fake_extract(self, variant, start_sec=None, end_sec=None, video_path_override=None):
        return 30.0, raw_3d, raw_2d, 1

    monkeypatch.setattr(KinematicAnalyzer, "_extract_frames_with_variant", fake_extract)

    out = KinematicAnalyzer("missing.mp4").analyze()
    assert out["metric_status"]["knee_angle"]["source"] == "predicted"
    assert out["metric_status"]["knee_angle"].get("reason") == "world_depth_unreliable"
    assert out["metric_status"]["elbow_angle"]["source"] == "predicted"
    assert out["metric_status"]["elbow_angle"].get("reason") == "world_depth_unreliable"
    assert out["knee_angle"] is not None
    assert out["elbow_angle"] is not None


def test_set_shot_telemetry_for_minimal_wrist_motion(monkeypatch):
    monkeypatch.setattr(KinematicAnalyzer, "_prepare_video", lambda self: self.video_path)
    monkeypatch.setattr(KinematicAnalyzer, "_init_pose", lambda self: True)

    raw_3d, raw_2d = _make_minimal_motion_pose(n=32)

    def fake_extract(self, variant, start_sec=None, end_sec=None, video_path_override=None):
        return 30.0, raw_3d, raw_2d, 1

    monkeypatch.setattr(KinematicAnalyzer, "_extract_frames_with_variant", fake_extract)

    out = KinematicAnalyzer("missing.mp4").analyze()
    assert out["telemetry"].get("shot_type") == "set_shot"


def test_debug_summary_included_when_env_enabled(monkeypatch):
    monkeypatch.setenv("LAKSH_INCLUDE_DEBUG_SUMMARY", "1")
    monkeypatch.setattr(KinematicAnalyzer, "_prepare_video", lambda self: self.video_path)
    monkeypatch.setattr(KinematicAnalyzer, "_init_pose", lambda self: False)

    out = KinematicAnalyzer("missing.mp4").analyze()
    assert "debug_summary" in out
    assert out["debug_summary"]["analysis_mode"] == "fallback"
    assert "pose_init_failed" in out["debug_summary"]["fallback_reason_codes"]
