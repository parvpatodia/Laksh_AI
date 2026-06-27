"""Tests for the leaderboard form-index scorer.

The honesty contract forbids claiming an uncalibrated value sits inside a
reference band. The leaderboard sidesteps that by ranking on a *transparent,
relative* index built only from MEASURED (status=valid) quantities, tagged
``uncalibrated_demo_index`` and never sold as a validated form grade. These
tests pin that behavior: measured-only inputs, honest status/reason_codes, and
graceful degradation when inputs are missing.
"""
from app.persistence.scoring import FORM_INDEX_UNIT, compute_form_index


def _feat(value, status="valid"):
    return {"value": value, "unit": "x", "status": status, "reason_codes": []}


def _rep(rep_index, rep_status, tempo, visibility, tempo_status="valid", vis_status="valid"):
    return {
        "rep_index": rep_index,
        "rep_status": rep_status,
        "features": {
            "tempo_ratio_ecc_over_con": _feat(tempo, tempo_status),
            "primary_joints_min_visibility": _feat(visibility, vis_status),
        },
    }


def test_no_reps_yields_unknown():
    fi = compute_form_index([])
    assert fi.value is None
    assert fi.status == "unknown"
    assert "no_valid_reps" in fi.reason_codes
    assert "uncalibrated_demo_index" in fi.reason_codes


def test_no_valid_reps_yields_unknown():
    reps = [_rep(0, "degraded", 1.0, 0.9), _rep(1, "unknown", 1.0, 0.9)]
    fi = compute_form_index(reps)
    assert fi.value is None
    assert fi.status == "unknown"


def test_full_valid_session_is_valid_and_bounded():
    reps = [
        _rep(0, "valid", tempo=2.0, visibility=0.95),
        _rep(1, "valid", tempo=2.0, visibility=0.93),
        _rep(2, "valid", tempo=2.0, visibility=0.94),
    ]
    fi = compute_form_index(reps)
    assert fi.status == "valid"
    assert 0.0 <= fi.value <= 100.0
    assert fi.unit == FORM_INDEX_UNIT
    assert "uncalibrated_demo_index" in fi.reason_codes
    # Consistent tempo + high tracking + all reps valid -> high index.
    assert fi.value > 80.0
    assert set(fi.components) >= {"valid_rep_ratio", "tracking_quality", "tempo_consistency"}


def test_single_valid_rep_degrades_tempo_consistency():
    fi = compute_form_index([_rep(0, "valid", tempo=2.0, visibility=0.9)])
    assert fi.status == "degraded"
    assert fi.value is not None
    assert "tempo_consistency_unavailable_single_rep" in fi.reason_codes
    assert "tempo_consistency" not in fi.components


def test_inconsistent_tempo_lowers_index_vs_consistent():
    consistent = compute_form_index(
        [_rep(i, "valid", tempo=2.0, visibility=0.95) for i in range(3)]
    )
    inconsistent = compute_form_index(
        [
            _rep(0, "valid", tempo=0.5, visibility=0.95),
            _rep(1, "valid", tempo=3.5, visibility=0.95),
            _rep(2, "valid", tempo=1.0, visibility=0.95),
        ]
    )
    assert inconsistent.value < consistent.value


def test_only_measured_fields_count_toward_tracking():
    # visibility field marked unknown must be excluded from tracking_quality.
    reps = [
        _rep(0, "valid", tempo=2.0, visibility=0.95, vis_status="valid"),
        _rep(1, "valid", tempo=2.0, visibility=0.10, vis_status="unknown"),
    ]
    fi = compute_form_index(reps)
    # The 0.10 visibility (status=unknown) is ignored, so tracking stays high.
    assert fi.components["tracking_quality"] > 0.9
