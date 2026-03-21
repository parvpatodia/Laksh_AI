import numpy as np

from correction_engine import generate_correction_video
from physics_engine import KinematicAnalyzer


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
