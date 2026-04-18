"""Tests for :mod:`app.parity.realtime`.

All tests are pure (no I/O, no network, no MediaPipe).
"""
from __future__ import annotations

import pytest

from app.parity.realtime import (
    DEFAULT_MAX_TOLERANCE,
    compare_feature_vectors,
    probe_reps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fv(value: float | None, status: str = "valid") -> dict:
    """Build a minimal FieldValueModel-like dict."""
    return {"value": value, "unit": "s", "status": status, "reason_codes": []}


def _rep(rep_index: int, features: dict) -> dict:
    """Build a minimal RepVectorModel-like dict."""
    return {
        "rep_index": rep_index,
        "start_frame": 0,
        "end_frame": 30,
        "peak_frame": 15,
        "rep_status": "valid",
        "features": features,
    }


# ---------------------------------------------------------------------------
# compare_feature_vectors
# ---------------------------------------------------------------------------


def test_perfect_agreement_is_within_tolerance() -> None:
    rt = {
        "rep_duration_s": _fv(2.0),
        "eccentric_duration_s": _fv(1.0),
    }
    can = {
        "rep_duration_s": _fv(2.0),
        "eccentric_duration_s": _fv(1.0),
    }
    result = compare_feature_vectors(rt, can)
    assert result["status"] == "within_tolerance"
    assert result["max_abs_delta"] == pytest.approx(0.0)
    assert result["p90_abs_delta"] == pytest.approx(0.0)
    assert set(result["fields_compared"]) == {"rep_duration_s", "eccentric_duration_s"}


def test_small_delta_within_tolerance() -> None:
    rt = {
        "rep_duration_s": _fv(2.1),
        "eccentric_duration_s": _fv(1.05),
    }
    can = {
        "rep_duration_s": _fv(2.0),
        "eccentric_duration_s": _fv(1.0),
    }
    result = compare_feature_vectors(rt, can)
    assert result["status"] == "within_tolerance"
    assert result["max_abs_delta"] <= DEFAULT_MAX_TOLERANCE


def test_large_delta_is_outside_tolerance() -> None:
    rt = {
        "rep_duration_s": _fv(2.0),
        "eccentric_duration_s": _fv(0.1),  # canonical is 1.5s -> delta 1.4s
    }
    can = {
        "rep_duration_s": _fv(2.0),
        "eccentric_duration_s": _fv(1.5),
    }
    result = compare_feature_vectors(rt, can)
    assert result["status"] == "outside_tolerance"
    assert result["max_abs_delta"] == pytest.approx(1.4, abs=1e-5)


def test_only_one_valid_pair_is_insufficient() -> None:
    rt = {"rep_duration_s": _fv(2.0)}
    can = {"rep_duration_s": _fv(2.0)}
    result = compare_feature_vectors(rt, can)
    assert result["status"] == "insufficient_data"
    assert result["fields_compared"] == ["rep_duration_s"]


def test_zero_valid_pairs_is_insufficient() -> None:
    result = compare_feature_vectors({}, {})
    assert result["status"] == "insufficient_data"
    assert result["fields_compared"] == []


def test_degraded_field_excluded_from_comparison() -> None:
    """Degraded status on either side must not count."""
    rt = {
        "rep_duration_s": _fv(2.0, status="valid"),
        "eccentric_duration_s": _fv(1.0, status="degraded"),  # degraded
    }
    can = {
        "rep_duration_s": _fv(2.0, status="valid"),
        "eccentric_duration_s": _fv(0.5, status="valid"),  # large delta but excluded
    }
    result = compare_feature_vectors(rt, can)
    # Only rep_duration_s compared -> insufficient_data (only 1 field).
    assert result["status"] == "insufficient_data"
    assert result["fields_compared"] == ["rep_duration_s"]


def test_none_value_excluded() -> None:
    """A None value on either side must be skipped."""
    rt = {
        "rep_duration_s": _fv(None, status="valid"),  # None value
        "eccentric_duration_s": _fv(1.0),
        "concentric_duration_s": _fv(0.8),
    }
    can = {
        "rep_duration_s": _fv(2.0, status="valid"),
        "eccentric_duration_s": _fv(1.0),
        "concentric_duration_s": _fv(0.8),
    }
    result = compare_feature_vectors(rt, can)
    # rep_duration_s excluded (rt None); eccentric and concentric included.
    assert "rep_duration_s" not in result["fields_compared"]
    assert result["status"] == "within_tolerance"


def test_fields_not_in_canonical_are_skipped() -> None:
    rt = {
        "rep_duration_s": _fv(2.0),
        "eccentric_duration_s": _fv(1.0),
        "realtime_only_metric": _fv(99.0),  # not in canonical
    }
    can = {
        "rep_duration_s": _fv(2.0),
        "eccentric_duration_s": _fv(1.0),
    }
    result = compare_feature_vectors(rt, can)
    assert "realtime_only_metric" not in result["fields_compared"]
    assert result["status"] == "within_tolerance"


def test_fields_compared_is_sorted() -> None:
    rt = {
        "zzz_field": _fv(1.0),
        "aaa_field": _fv(1.0),
    }
    can = {
        "zzz_field": _fv(1.0),
        "aaa_field": _fv(1.0),
    }
    result = compare_feature_vectors(rt, can)
    assert result["fields_compared"] == sorted(result["fields_compared"])


def test_custom_tolerance() -> None:
    """A stricter tolerance should flip within->outside."""
    rt = {
        "rep_duration_s": _fv(2.1),
        "eccentric_duration_s": _fv(1.05),
    }
    can = {
        "rep_duration_s": _fv(2.0),
        "eccentric_duration_s": _fv(1.0),
    }
    # With very tight tolerance (0.001) the p90 delta (0.05) should fail.
    result = compare_feature_vectors(rt, can, p90_tolerance=0.001, max_tolerance=0.001)
    assert result["status"] == "outside_tolerance"


# ---------------------------------------------------------------------------
# probe_reps
# ---------------------------------------------------------------------------


def test_probe_reps_single_rep_agreement() -> None:
    rt_vecs = [
        _rep(0, {
            "rep_duration_s": _fv(2.0),
            "eccentric_duration_s": _fv(1.0),
        }),
    ]
    can_vecs = [
        _rep(0, {
            "rep_duration_s": _fv(2.0),
            "eccentric_duration_s": _fv(1.0),
        }),
    ]
    result = probe_reps(rt_vecs, can_vecs)
    assert result["status"] == "within_tolerance"
    assert result["max_abs_delta"] == pytest.approx(0.0)


def test_probe_reps_multiple_reps_pooled() -> None:
    """Deltas across 3 reps must be pooled into a single p90/max."""
    rt_vecs = [
        _rep(0, {"rep_duration_s": _fv(2.0), "eccentric_duration_s": _fv(1.0)}),
        _rep(1, {"rep_duration_s": _fv(2.1), "eccentric_duration_s": _fv(1.1)}),
        _rep(2, {"rep_duration_s": _fv(1.9), "eccentric_duration_s": _fv(0.9)}),
    ]
    can_vecs = [
        _rep(0, {"rep_duration_s": _fv(2.0), "eccentric_duration_s": _fv(1.0)}),
        _rep(1, {"rep_duration_s": _fv(2.0), "eccentric_duration_s": _fv(1.0)}),
        _rep(2, {"rep_duration_s": _fv(2.0), "eccentric_duration_s": _fv(1.0)}),
    ]
    result = probe_reps(rt_vecs, can_vecs)
    # Largest delta is 0.1s; all within default tolerance.
    assert result["status"] == "within_tolerance"
    assert result["max_abs_delta"] == pytest.approx(0.1, abs=1e-5)


def test_probe_reps_unmatched_rep_skipped() -> None:
    """Realtime has rep 0 only; canonical has reps 0 + 1. Rep 1 skipped."""
    rt_vecs = [
        _rep(0, {"rep_duration_s": _fv(2.0), "eccentric_duration_s": _fv(1.0)}),
    ]
    can_vecs = [
        _rep(0, {"rep_duration_s": _fv(2.0), "eccentric_duration_s": _fv(1.0)}),
        _rep(1, {"rep_duration_s": _fv(3.0), "eccentric_duration_s": _fv(2.0)}),
    ]
    result = probe_reps(rt_vecs, can_vecs)
    # Only rep 0 compared -> within_tolerance (delta 0).
    assert result["status"] == "within_tolerance"


def test_probe_reps_no_overlap_is_insufficient() -> None:
    rt_vecs = [_rep(0, {"rep_duration_s": _fv(2.0), "eccentric_duration_s": _fv(1.0)})]
    can_vecs = [_rep(1, {"rep_duration_s": _fv(2.0), "eccentric_duration_s": _fv(1.0)})]
    result = probe_reps(rt_vecs, can_vecs)
    assert result["status"] == "insufficient_data"


def test_probe_reps_outside_tolerance_aggregate() -> None:
    """If any rep has a huge delta the aggregate must be outside_tolerance."""
    rt_vecs = [
        _rep(0, {"rep_duration_s": _fv(2.0), "eccentric_duration_s": _fv(1.0)}),
        _rep(1, {"rep_duration_s": _fv(2.0), "eccentric_duration_s": _fv(0.01)}),
    ]
    can_vecs = [
        _rep(0, {"rep_duration_s": _fv(2.0), "eccentric_duration_s": _fv(1.0)}),
        _rep(1, {"rep_duration_s": _fv(2.0), "eccentric_duration_s": _fv(1.5)}),
    ]
    result = probe_reps(rt_vecs, can_vecs)
    assert result["status"] == "outside_tolerance"
    assert result["max_abs_delta"] == pytest.approx(1.49, abs=1e-4)
