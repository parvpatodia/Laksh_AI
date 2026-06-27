"""Tests for app.gym.calibration_v0 and scripts/freeze_calibration_v0.py.

Coverage targets:
  * CalibrationEntry.__post_init__ enforces the GOALS.md calibration policy.
  * load_calibration_v0_from_dict enforces schema + coverage guards.
  * apply_calibration returns the documented four statuses.
  * The shipped v0 JSON on disk parses cleanly and is all-uncalibrated.
  * freeze_calibration_v0.py CLI: verify, --print, --expected-sha, --show-versions.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.gym.calibration_v0 import (
    CALIBRATION_V0_MANIFEST_VERSION,
    CALIBRATION_V0_SCHEMA_VERSION,
    CalibrationEntry,
    apply_calibration,
    compute_manifest_sha256,
    load_calibration_v0,
    load_calibration_v0_from_dict,
    manifest_to_dict,
)
from app.gym.exercises_v0 import EXERCISES_V0

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "evaluation" / "gym_calibration_v0.json"
FREEZE_SCRIPT = REPO_ROOT / "scripts" / "freeze_calibration_v0.py"


# ---------- CalibrationEntry.__post_init__ policy guards ------------------


def test_uncalibrated_v0_with_ranges_is_rejected() -> None:
    with pytest.raises(ValueError, match="uncalibrated_v0"):
        CalibrationEntry(
            exercise_id="back_squat",
            comparable_fields=("rep_duration_s",),
            reference_ranges={"rep_duration_s": (1.0, 3.0)},
            evidence_status="uncalibrated_v0",
            evidence_source=None,
        )


def test_uncalibrated_v0_with_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="uncalibrated_v0"):
        CalibrationEntry(
            exercise_id="back_squat",
            comparable_fields=("rep_duration_s",),
            reference_ranges={},
            evidence_status="uncalibrated_v0",
            evidence_source="scorecards/2026-04.json#row7",
        )


def test_cited_without_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="cited"):
        CalibrationEntry(
            exercise_id="back_squat",
            comparable_fields=("rep_duration_s",),
            reference_ranges={"rep_duration_s": (1.0, 3.0)},
            evidence_status="cited",
            evidence_source="",
        )


def test_cited_without_ranges_is_rejected() -> None:
    with pytest.raises(ValueError, match="cited"):
        CalibrationEntry(
            exercise_id="back_squat",
            comparable_fields=("rep_duration_s",),
            reference_ranges={},
            evidence_status="cited",
            evidence_source="scorecards/2026-04.json#row7",
        )


def test_cited_valid_entry_constructs() -> None:
    entry = CalibrationEntry(
        exercise_id="back_squat",
        comparable_fields=("rep_duration_s", "signal_amplitude"),
        reference_ranges={"rep_duration_s": (1.0, 4.0)},
        evidence_status="cited",
        evidence_source="scorecards/2026-04.json#row7",
    )
    assert entry.reference_ranges["rep_duration_s"] == (1.0, 4.0)


def test_range_lo_must_be_less_than_hi() -> None:
    with pytest.raises(ValueError, match="lo < hi"):
        CalibrationEntry(
            exercise_id="back_squat",
            comparable_fields=("rep_duration_s",),
            reference_ranges={"rep_duration_s": (3.0, 3.0)},
            evidence_status="cited",
            evidence_source="scorecards/2026-04.json#row7",
        )


def test_range_field_must_be_in_comparable_fields() -> None:
    with pytest.raises(ValueError, match="must also be listed"):
        CalibrationEntry(
            exercise_id="back_squat",
            comparable_fields=("rep_duration_s",),
            reference_ranges={"signal_amplitude": (0.05, 0.2)},
            evidence_status="cited",
            evidence_source="scorecards/2026-04.json#row7",
        )


def test_comparable_field_not_in_allowlist_rejected() -> None:
    with pytest.raises(ValueError, match="comparable_fields"):
        CalibrationEntry(
            exercise_id="back_squat",
            comparable_fields=("not_a_real_field",),
            reference_ranges={},
            evidence_status="uncalibrated_v0",
            evidence_source=None,
        )


def test_unknown_exercise_id_rejected() -> None:
    with pytest.raises(ValueError, match="unknown exercise_id"):
        CalibrationEntry(
            exercise_id="moon_walk",
            comparable_fields=("rep_duration_s",),
            reference_ranges={},
            evidence_status="uncalibrated_v0",
            evidence_source=None,
        )


def test_duplicate_comparable_fields_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        CalibrationEntry(
            exercise_id="back_squat",
            comparable_fields=("rep_duration_s", "rep_duration_s"),
            reference_ranges={},
            evidence_status="uncalibrated_v0",
            evidence_source=None,
        )


# ---------- load_calibration_v0_from_dict schema + coverage guards --------


def _v0_all_uncalibrated_payload() -> dict:
    """Minimal valid v0 payload covering every EXERCISES_V0 entry."""
    return {
        "schema_version": CALIBRATION_V0_SCHEMA_VERSION,
        "manifest_version": CALIBRATION_V0_MANIFEST_VERSION,
        "evidence_source": None,
        "exercises": [
            {
                "exercise_id": eid,
                "comparable_fields": ["rep_duration_s"],
                "reference_ranges": {},
                "evidence_status": "uncalibrated_v0",
                "evidence_source": None,
                "notes": "",
            }
            for eid in sorted(EXERCISES_V0.keys())
        ],
    }


def test_load_minimal_v0_payload_succeeds() -> None:
    m = load_calibration_v0_from_dict(_v0_all_uncalibrated_payload())
    assert len(m.entries) == len(EXERCISES_V0)
    assert all(e.evidence_status == "uncalibrated_v0" for e in m.entries.values())


def test_schema_version_mismatch_rejected() -> None:
    payload = _v0_all_uncalibrated_payload()
    payload["schema_version"] = "9.9.9"
    with pytest.raises(ValueError, match="schema_version"):
        load_calibration_v0_from_dict(payload)


def test_missing_entries_rejected() -> None:
    payload = _v0_all_uncalibrated_payload()
    payload["exercises"] = payload["exercises"][:-1]  # drop one
    with pytest.raises(ValueError, match="missing entries"):
        load_calibration_v0_from_dict(payload)


def test_duplicate_exercise_id_rejected() -> None:
    payload = _v0_all_uncalibrated_payload()
    payload["exercises"].append(payload["exercises"][0])
    with pytest.raises(ValueError, match="duplicate"):
        load_calibration_v0_from_dict(payload)


def test_unknown_exercise_id_in_payload_rejected() -> None:
    payload = _v0_all_uncalibrated_payload()
    payload["exercises"][0]["exercise_id"] = "moon_walk"
    # CalibrationEntry constructor catches this before coverage guard.
    with pytest.raises(ValueError, match="unknown exercise_id"):
        load_calibration_v0_from_dict(payload)


# ---------- apply_calibration four-status matrix --------------------------


@dataclass(frozen=True)
class _FV:
    value: float | None
    status: str


@dataclass(frozen=True)
class _RFV:
    features: dict


def test_apply_calibration_no_reference_yet() -> None:
    entry = CalibrationEntry(
        exercise_id="back_squat",
        comparable_fields=("rep_duration_s",),
        reference_ranges={},
        evidence_status="uncalibrated_v0",
        evidence_source=None,
    )
    rfv = _RFV(features={"rep_duration_s": _FV(value=2.0, status="valid")})
    out = apply_calibration(entry, rfv)
    assert out["rep_duration_s"]["status"] == "no_reference_yet"
    assert out["rep_duration_s"]["value"] == 2.0
    assert out["rep_duration_s"]["range"] is None
    assert out["rep_duration_s"]["evidence_status"] == "uncalibrated_v0"


def test_apply_calibration_unavailable_when_feature_unknown() -> None:
    entry = CalibrationEntry(
        exercise_id="back_squat",
        comparable_fields=("rep_duration_s",),
        reference_ranges={},
        evidence_status="uncalibrated_v0",
        evidence_source=None,
    )
    rfv = _RFV(features={"rep_duration_s": _FV(value=None, status="unknown")})
    out = apply_calibration(entry, rfv)
    assert out["rep_duration_s"]["status"] == "unavailable"
    assert out["rep_duration_s"]["value"] is None


def test_apply_calibration_within_and_outside_reference() -> None:
    entry = CalibrationEntry(
        exercise_id="back_squat",
        comparable_fields=("rep_duration_s",),
        reference_ranges={"rep_duration_s": (1.5, 3.5)},
        evidence_status="cited",
        evidence_source="scorecards/2026-04.json#row7",
    )
    inside = _RFV(features={"rep_duration_s": _FV(value=2.0, status="valid")})
    outside = _RFV(features={"rep_duration_s": _FV(value=5.0, status="valid")})
    boundary = _RFV(features={"rep_duration_s": _FV(value=3.5, status="valid")})  # inclusive hi
    assert apply_calibration(entry, inside)["rep_duration_s"]["status"] == "within_reference"
    assert apply_calibration(entry, outside)["rep_duration_s"]["status"] == "outside_reference"
    assert apply_calibration(entry, boundary)["rep_duration_s"]["status"] == "within_reference"


def test_apply_calibration_handles_missing_field() -> None:
    entry = CalibrationEntry(
        exercise_id="back_squat",
        comparable_fields=("rep_duration_s",),
        reference_ranges={},
        evidence_status="uncalibrated_v0",
        evidence_source=None,
    )
    rfv = _RFV(features={})  # no such key
    out = apply_calibration(entry, rfv)
    assert out["rep_duration_s"]["status"] == "unavailable"


# ---------- shipped v0 file integration guards ----------------------------


def test_shipped_v0_config_parses_cleanly() -> None:
    manifest = load_calibration_v0(SHIPPED_CONFIG)
    assert manifest.schema_version == CALIBRATION_V0_SCHEMA_VERSION
    assert set(manifest.entries.keys()) == set(EXERCISES_V0.keys())


def test_shipped_v0_config_is_all_cited_v0_2_0() -> None:
    """Policy guard (v0.2.0+): every entry MUST be cited with a verifiable
    evidence_source. v0.1.0 shipped uncalibrated; v0.2.0 swapped that for
    literature-cited bands sourced from
    ``evaluation/calibration_evidence_v0/literature_bundle_v0.md``.

    The invariant flips intentionally: any future regression to
    ``uncalibrated_v0`` for a registered exercise is a policy violation, and
    so is shipping ``evidence_status='cited'`` without an evidence_source
    file that actually exists on disk.
    """
    manifest = load_calibration_v0(SHIPPED_CONFIG)
    assert manifest.manifest_version == "v0.2.0", (
        "shipped calibration must be on the v0.2.0 cited-bands manifest"
    )
    for eid, entry in manifest.entries.items():
        assert entry.evidence_status == "cited", (
            f"{eid}: v0.2.0 ships cited ranges; uncalibrated_v0 is a regression"
        )
        assert entry.reference_ranges, f"{eid}: cited entries must carry ranges"
        assert entry.evidence_source, f"{eid}: cited entries must name a source"
        # The shipped source must be a real file in the repo so a reviewer
        # can verify the citation. We allow only the literature bundle path
        # in v0.2.0; later milestones may add scorecard paths alongside.
        src_path = REPO_ROOT / entry.evidence_source
        assert src_path.is_file(), (
            f"{eid}: evidence_source {entry.evidence_source!r} must exist on disk"
        )


def test_shipped_v0_sha_is_deterministic() -> None:
    m1 = load_calibration_v0(SHIPPED_CONFIG)
    m2 = load_calibration_v0(SHIPPED_CONFIG)
    assert compute_manifest_sha256(m1) == compute_manifest_sha256(m2)


def test_shipped_v0_round_trip() -> None:
    m = load_calibration_v0(SHIPPED_CONFIG)
    round_tripped = load_calibration_v0_from_dict(manifest_to_dict(m))
    assert compute_manifest_sha256(m) == compute_manifest_sha256(round_tripped)


# ---------- freeze_calibration_v0.py CLI ----------------------------------


def _run_freeze(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(FREEZE_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_freeze_cli_verify_default_config_ok() -> None:
    res = _run_freeze()
    assert res.returncode == 0, res.stderr + res.stdout
    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    assert payload["n_entries"] == len(EXERCISES_V0)
    # v0.2.0+: every entry is literature-cited (see
    # evaluation/calibration_evidence_v0/literature_bundle_v0.md).
    assert payload["n_cited"] == len(EXERCISES_V0)
    assert payload["n_uncalibrated_v0"] == 0


def test_freeze_cli_expected_sha_match() -> None:
    m = load_calibration_v0(SHIPPED_CONFIG)
    sha = compute_manifest_sha256(m)
    res = _run_freeze("--expected-sha", sha)
    assert res.returncode == 0, res.stderr + res.stdout


def test_freeze_cli_expected_sha_mismatch() -> None:
    res = _run_freeze("--expected-sha", "0" * 64)
    assert res.returncode == 1, res.stderr + res.stdout
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "sha_mismatch"


def test_freeze_cli_show_versions() -> None:
    res = _run_freeze("--show-versions")
    assert res.returncode == 0, res.stderr + res.stdout
    payload = json.loads(res.stdout)
    assert payload["schema_version"] == CALIBRATION_V0_SCHEMA_VERSION
    assert payload["manifest_version"] == CALIBRATION_V0_MANIFEST_VERSION


def test_freeze_cli_print_emits_canonical_json() -> None:
    res = _run_freeze("--print")
    assert res.returncode == 0, res.stderr + res.stdout
    parsed = json.loads(res.stdout)
    assert parsed["schema_version"] == CALIBRATION_V0_SCHEMA_VERSION
    assert len(parsed["exercises"]) == len(EXERCISES_V0)
