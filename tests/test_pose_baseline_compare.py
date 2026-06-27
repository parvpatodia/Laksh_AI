"""Pose baseline JSONL comparison (P1b)."""

import json
from pathlib import Path

import pytest

from app.pose.pose_baseline_compare import (
    compare_pose_baseline_rows,
    load_pose_baseline_rows,
    per_clip_diff_rows,
)


def test_compare_both_ok_and_delta(tmp_path: Path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(
        json.dumps(
            {
                "clip_id": "c1",
                "ok": True,
                "detection_rate": 0.5,
                "pose_usable_heuristic": True,
                "backend": "mediapipe",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    b.write_text(
        json.dumps(
            {
                "clip_id": "c1",
                "ok": True,
                "detection_rate": 0.7,
                "pose_usable_heuristic": False,
                "backend": "rtmpose",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ra = load_pose_baseline_rows(a)
    rb = load_pose_baseline_rows(b)
    s = compare_pose_baseline_rows(ra, rb, label_a="mp", label_b="rtm")
    assert s["clips_both_ok"] == 1
    assert s["intersection_both_ok"]["mean_delta_detection_rate_b_minus_a"] == pytest.approx(0.2)
    assert s["intersection_both_ok"]["usable_heuristic_count_a"] == 1
    assert s["intersection_both_ok"]["usable_heuristic_count_b"] == 0
    assert "c1" in s["intersection_both_ok"]["usable_lost_b_vs_a_clip_ids_sample"]
    assert s["intersection_both_ok"]["median_delta_detection_rate_b_minus_a"] == pytest.approx(0.2)
    assert s["intersection_both_ok"]["clips_ffmpeg_preprocess_mismatch"] == 0
    assert s["confound_notes"] == []


def test_median_min_max_and_ffmpeg_confound(tmp_path: Path):
    """Three clips both_ok: deltas -0.1, 0, 0.3 → median 0, min/max set; ffmpeg mismatch flagged."""
    rows_a = []
    rows_b = []
    for i, (da, db, ff_a, ff_b) in enumerate(
        [
            (0.5, 0.4, True, True),
            (0.6, 0.6, True, True),
            (0.7, 1.0, True, False),
        ]
    ):
        cid = f"c{i}"
        rows_a.append(
            json.dumps(
                {
                    "clip_id": cid,
                    "ok": True,
                    "detection_rate": da,
                    "ffmpeg_preprocess_applied": ff_a,
                }
            )
        )
        rows_b.append(
            json.dumps(
                {
                    "clip_id": cid,
                    "ok": True,
                    "detection_rate": db,
                    "ffmpeg_preprocess_applied": ff_b,
                }
            )
        )
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("\n".join(rows_a) + "\n", encoding="utf-8")
    b.write_text("\n".join(rows_b) + "\n", encoding="utf-8")
    ra = load_pose_baseline_rows(a)
    rb = load_pose_baseline_rows(b)
    s = compare_pose_baseline_rows(ra, rb)
    assert s["clips_both_ok"] == 3
    inter = s["intersection_both_ok"]
    assert inter["mean_delta_detection_rate_b_minus_a"] == pytest.approx(0.0667, abs=1e-3)
    assert inter["median_delta_detection_rate_b_minus_a"] == pytest.approx(0.0)
    assert inter["min_delta_detection_rate_b_minus_a"] == pytest.approx(-0.1)
    assert inter["max_delta_detection_rate_b_minus_a"] == pytest.approx(0.3)
    assert inter["clips_ffmpeg_preprocess_mismatch"] == 1
    assert "c2" in inter["clip_ids_ffmpeg_preprocess_mismatch_sample"]
    assert len(s["confound_notes"]) == 1


def test_per_clip_ffmpeg_mismatch_flag():
    ra = {
        "z": {
            "clip_id": "z",
            "ok": True,
            "detection_rate": 0.5,
            "ffmpeg_preprocess_applied": True,
        }
    }
    rb = {
        "z": {
            "clip_id": "z",
            "ok": True,
            "detection_rate": 0.5,
            "ffmpeg_preprocess_applied": False,
        }
    }
    rows = per_clip_diff_rows(ra, rb)
    assert rows[0]["ffmpeg_preprocess_mismatch"] is True


def test_only_in_one_file(tmp_path: Path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(
        json.dumps({"clip_id": "only_a", "ok": True, "detection_rate": 1.0}) + "\n", encoding="utf-8"
    )
    b.write_text(
        json.dumps({"clip_id": "only_b", "ok": True, "detection_rate": 1.0}) + "\n", encoding="utf-8"
    )
    ra = load_pose_baseline_rows(a)
    rb = load_pose_baseline_rows(b)
    s = compare_pose_baseline_rows(ra, rb)
    assert set(s["clip_ids_only_in_a"]) == {"only_a"}
    assert set(s["clip_ids_only_in_b"]) == {"only_b"}


def test_invalid_jsonl_raises(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_pose_baseline_rows(p)


def test_per_clip_diff_rows():
    ra = {"x": {"clip_id": "x", "ok": True, "detection_rate": 0.1, "backend": "a"}}
    rb = {"x": {"clip_id": "x", "ok": True, "detection_rate": 0.2, "backend": "b"}}
    rows = per_clip_diff_rows(ra, rb)
    assert len(rows) == 1
    assert rows[0]["delta_detection_rate_b_minus_a"] == pytest.approx(0.1)


def test_p2_l0_multi_person_cleared():
    """P2 L0: compare reason_codes multiple_people_detected between two runs."""
    ra = {
        "c1": {
            "clip_id": "c1",
            "ok": True,
            "detection_rate": 0.9,
            "reason_codes": ["multiple_people_detected", "low_visibility_core"],
        },
        "c2": {
            "clip_id": "c2",
            "ok": True,
            "detection_rate": 0.8,
            "reason_codes": [],
        },
    }
    rb = {
        "c1": {
            "clip_id": "c1",
            "ok": True,
            "detection_rate": 0.85,
            "reason_codes": ["low_visibility_core"],
        },
        "c2": {
            "clip_id": "c2",
            "ok": True,
            "detection_rate": 0.8,
            "reason_codes": ["multiple_people_detected"],
        },
    }
    s = compare_pose_baseline_rows(ra, rb, label_a="full", label_b="roi")
    p2 = s["p2_l0"]
    assert p2["clips_both_ok"] == 2
    assert p2["multiple_people_detected_count_a"] == 1
    assert p2["multiple_people_detected_count_b"] == 1
    assert p2["n_cleared_multi_person_b_vs_a"] == 1
    assert p2["cleared_multi_person_b_vs_a_clip_ids_sample"] == ["c1"]
    assert p2["n_introduced_multi_person_b_vs_a"] == 1
    assert p2["introduced_multi_person_b_vs_a_clip_ids_sample"] == ["c2"]


def test_per_clip_includes_p2_fields():
    ra = {
        "z": {
            "clip_id": "z",
            "ok": True,
            "detection_rate": 0.5,
            "max_people_seen": 2,
            "reason_codes": ["multiple_people_detected"],
            "provenance": {"person_isolation": {"haar_detection_attempts": 13}},
        }
    }
    rb = {
        "z": {
            "clip_id": "z",
            "ok": True,
            "detection_rate": 0.5,
            "max_people_seen": 1,
            "reason_codes": [],
            "provenance": {"person_isolation": {"redetect_events": 13}},
        }
    }
    rows = per_clip_diff_rows(ra, rb)
    assert rows[0]["multiple_people_detected_a"] is True
    assert rows[0]["multiple_people_detected_b"] is False
    assert rows[0]["max_people_seen_a"] == 2
    assert rows[0]["max_people_seen_b"] == 1
    assert rows[0]["haar_detection_attempts_a"] == 13
    assert rows[0]["haar_detection_attempts_b"] == 13
