"""Smoke tests for scripts/build_scorecard.py — the regression bundle generator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_scorecard as bs


def _synthetic_rows() -> list[dict]:
    """Two clips: one clean, one degraded. Exercises quantile + reason codes."""
    return [
        {
            "backend": "mediapipe",
            "video_path": "/abs/clips/good.mp4",
            "ok": True,
            "error": None,
            "n_frames": 120,
            "n_frames_with_pose": 120,
            "detection_rate": 1.0,
            "visibility_core_when_detected": 0.95,
            "visibility_core_all_frames": 0.95,
            "hip_mid_displacement_median_norm": 0.003,
            "max_people_seen": 1,
            "selected_pass": "baseline_only",
            "pose_usable_heuristic": True,
            "reason_codes": [],
            "fps": 30.0,
            "ffmpeg_preprocess_applied": True,
        },
        {
            "backend": "mediapipe",
            "video_path": "/abs/clips/bad.mp4",
            "ok": True,
            "error": None,
            "n_frames": 100,
            "n_frames_with_pose": 40,
            "detection_rate": 0.4,
            "visibility_core_when_detected": 0.65,
            "visibility_core_all_frames": 0.26,
            "hip_mid_displacement_median_norm": 0.05,
            "max_people_seen": 2,
            "selected_pass": "baseline_only",
            "pose_usable_heuristic": False,
            "reason_codes": ["multi_person", "low_visibility"],
            "fps": 30.0,
            "ffmpeg_preprocess_applied": False,
        },
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_backend_aggregate_quantiles() -> None:
    agg = bs._backend_aggregate(_synthetic_rows())
    assert agg["n_ok"] == 2
    assert agg["n_rows"] == 2
    assert pytest.approx(agg["detection_rate_mean"], rel=1e-6) == 0.7
    # p10 is closer to the min (0.4), p90 closer to the max (1.0).
    assert agg["detection_rate_p10"] < agg["detection_rate_p50"] < agg["detection_rate_p90"]
    assert agg["usable_rate"] == 0.5
    top_codes = dict(agg["top_reason_codes"])
    assert top_codes == {"multi_person": 1, "low_visibility": 1}


def test_quantile_edge_cases() -> None:
    assert bs._quantile([], 0.5) is None
    assert bs._quantile([0.42], 0.5) == 0.42
    # Monotone
    xs = [0.1, 0.5, 0.9]
    assert bs._quantile(xs, 0.0) == 0.1
    assert bs._quantile(xs, 1.0) == 0.9
    mid = bs._quantile(xs, 0.5)
    assert mid is not None
    assert 0.1 < mid < 0.9


def test_build_scorecard_writes_expected_sections(tmp_path: Path) -> None:
    jsonl = tmp_path / "pose.jsonl"
    _write_jsonl(jsonl, _synthetic_rows())
    out = tmp_path / "scorecard.md"

    written = bs.build(
        manifest=None,
        jsonl_paths=[jsonl],
        out=out,
        requirements_lock=tmp_path / "nonexistent.lock",
    )
    assert written == out
    text = out.read_text(encoding="utf-8")

    # Header
    assert "Release scorecard" in text
    assert '"scorecard_schema_version"' in text
    assert '"pose_jsonl_artifacts"' in text

    # Aggregate block
    assert "Backend: `mediapipe`" in text
    assert "detection_rate" in text
    assert "`good.mp4`" in text
    assert "`bad.mp4`" in text

    # Reason codes propagate
    assert "multi_person" in text
    assert "low_visibility" in text

    # Worst-first ordering: bad.mp4 row must appear before good.mp4 in per-clip table
    bad_idx = text.index("`bad.mp4`")
    good_idx = text.index("`good.mp4`")
    assert bad_idx < good_idx


def test_header_only_when_no_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "scorecard.md"
    bs.build(
        manifest=None,
        jsonl_paths=[],
        out=out,
        requirements_lock=tmp_path / "nonexistent.lock",
    )
    text = out.read_text(encoding="utf-8")
    assert "No JSONL provided" in text
    assert "Aggregate metrics" in text
