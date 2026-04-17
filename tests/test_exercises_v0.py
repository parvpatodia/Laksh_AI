"""Tests for app/gym/exercises_v0.py + scripts/freeze_exercise_v0.py.

These tests enforce the frozen-artifact contract (GOALS.md Milestone 1):
taxonomy is stable, SHA-committed, and deterministic.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.gym import exercises_v0 as ev0
from app.pose.canonical import CanonicalJointName

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_JSON = REPO_ROOT / "evaluation" / "exercise_v0_manifest.json"


def test_registry_size_in_goals_range() -> None:
    # GOALS.md Milestone 1 bullet 1: "8-15 movements".
    assert 8 <= len(ev0.EXERCISES_V0) <= 15


def test_all_exercise_ids_unique_and_snake_case() -> None:
    ids = ev0.list_exercise_ids()
    assert len(ids) == len(set(ids))
    for eid in ids:
        assert eid == eid.lower()
        assert " " not in eid
        assert "-" not in eid


def test_all_primary_joints_are_canonical() -> None:
    canonical = {j.value for j in CanonicalJointName}
    for ex in ev0.EXERCISES_V0.values():
        assert set(ex.primary_joints).issubset(canonical), ex.exercise_id
        if ex.rep_signal_joint is not None:
            assert ex.rep_signal_joint in canonical, ex.exercise_id


def test_cyclic_types_carry_rep_signal_joint_noncyclic_do_not() -> None:
    for ex in ev0.EXERCISES_V0.values():
        cyclic = ex.rep_signal_type in {"cyclic_angle", "cyclic_vertical"}
        if cyclic:
            assert ex.rep_signal_joint is not None, ex.exercise_id
        else:
            assert ex.rep_signal_joint is None, ex.exercise_id


def test_validate_exercise_id_accepts_known_reserved_and_empty() -> None:
    assert ev0.validate_exercise_id(None) is None
    assert ev0.validate_exercise_id("") is None
    assert ev0.validate_exercise_id("  ") is None
    assert ev0.validate_exercise_id("back_squat") is None
    assert ev0.validate_exercise_id("mixed") is None
    assert ev0.validate_exercise_id("unknown") is None


def test_validate_exercise_id_rejects_unknown() -> None:
    err = ev0.validate_exercise_id("flying_squirrel")
    assert err is not None
    assert "flying_squirrel" in err
    assert "back_squat" in err  # lists the allowed set


def test_construct_exercise_rejects_bad_category() -> None:
    with pytest.raises(ValueError, match="category"):
        ev0.ExerciseV0(
            exercise_id="bad",
            display_name="Bad",
            category="not_a_category",
            camera_view_hint="side",
            rep_signal_type="cyclic_angle",
            rep_signal_joint="right_elbow",
            primary_joints=("right_elbow",),
            framing="upper_body",
            camera_instruction="n/a",
        )


def test_construct_exercise_rejects_cyclic_without_rep_signal_joint() -> None:
    with pytest.raises(ValueError, match="rep_signal_joint"):
        ev0.ExerciseV0(
            exercise_id="bad",
            display_name="Bad",
            category="squat",
            camera_view_hint="side",
            rep_signal_type="cyclic_angle",
            rep_signal_joint=None,
            primary_joints=("right_hip",),
            framing="full_body",
            camera_instruction="n/a",
        )


def test_construct_exercise_rejects_noncyclic_with_rep_signal_joint() -> None:
    with pytest.raises(ValueError, match="non-cyclic"):
        ev0.ExerciseV0(
            exercise_id="bad",
            display_name="Bad",
            category="isometric_core",
            camera_view_hint="side",
            rep_signal_type="duration",
            rep_signal_joint="right_hip",
            primary_joints=("right_hip",),
            framing="full_body",
            camera_instruction="n/a",
        )


def test_sha256_is_stable_across_calls() -> None:
    s1 = ev0.compute_manifest_sha256()
    s2 = ev0.compute_manifest_sha256()
    assert s1 == s2
    assert len(s1) == 64  # sha256 hex


def test_to_manifest_dict_has_required_top_level_keys() -> None:
    m = ev0.to_manifest_dict()
    for k in (
        "schema_version",
        "manifest_version",
        "allowed_categories",
        "allowed_camera_views",
        "allowed_framings",
        "allowed_rep_signal_types",
        "reserved_exercise_tokens",
        "exercises",
    ):
        assert k in m
    assert m["schema_version"] == ev0.EXERCISE_V0_SCHEMA_VERSION
    assert m["manifest_version"] == ev0.EXERCISE_V0_MANIFEST_VERSION
    assert len(m["exercises"]) == len(ev0.EXERCISES_V0)


def test_frozen_json_matches_in_code_registry() -> None:
    """Committed artifact must not drift from the source of truth."""
    assert FROZEN_JSON.is_file(), (
        f"{FROZEN_JSON} missing — regenerate with scripts/freeze_exercise_v0.py"
    )
    on_disk = json.loads(FROZEN_JSON.read_text(encoding="utf-8"))
    expected = ev0.to_manifest_dict()
    expected["sha256"] = ev0.compute_manifest_sha256()
    assert on_disk == expected, (
        "exercise_v0_manifest.json drifted from app/gym/exercises_v0.py; "
        "run: python scripts/freeze_exercise_v0.py"
    )


def test_freeze_script_verify_mode_passes() -> None:
    """End-to-end: --verify on the committed JSON returns exit 0."""
    cmd = [sys.executable, str(REPO_ROOT / "scripts/freeze_exercise_v0.py"), "--verify"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True


def test_freeze_script_print_mode_is_valid_json() -> None:
    cmd = [sys.executable, str(REPO_ROOT / "scripts/freeze_exercise_v0.py"), "--print"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout)
    assert parsed["schema_version"] == ev0.EXERCISE_V0_SCHEMA_VERSION
    assert parsed["sha256"] == ev0.compute_manifest_sha256()
