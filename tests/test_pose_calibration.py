"""Smoke tests for scripts/pose_calibration.py (reliability-diagram bins)."""
from __future__ import annotations

import pytest

from scripts import pose_calibration as pc


def _row(ok: bool, vis: float | None, usable: bool | None) -> dict:
    return {
        "ok": ok,
        "visibility_core_when_detected": vis,
        "pose_usable_heuristic": usable,
    }


def test_bins_cover_unit_interval() -> None:
    buckets = pc.reliability_bins([(0.1, True), (0.9, False)], n_bins=10)
    assert len(buckets) == 10
    assert buckets[0]["bin_lo"] == 0.0
    assert buckets[-1]["bin_hi"] == pytest.approx(1.0)


def test_final_bin_inclusive_of_one() -> None:
    # Predicted=1.0 must land in the last bin, not be dropped.
    buckets = pc.reliability_bins([(1.0, True)], n_bins=5)
    assert buckets[-1]["n"] == 1
    assert sum(b["n"] for b in buckets) == 1


def test_perfect_calibration_has_zero_ece() -> None:
    # Two tight bins at ~0.2 and ~0.8 where empirical matches predicted.
    # 10 samples at p=0.2 with 2 successes → rate=0.2.
    # 10 samples at p=0.8 with 8 successes → rate=0.8.
    pairs: list[tuple[float, bool]] = []
    pairs += [(0.2, True), (0.2, True)] + [(0.2, False)] * 8
    pairs += [(0.8, True)] * 8 + [(0.8, False), (0.8, False)]
    buckets = pc.reliability_bins(pairs, n_bins=10)
    ece = pc.expected_calibration_error(buckets, total_n=len(pairs))
    assert ece is not None
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_miscalibration_positive_ece() -> None:
    # All predictions 0.9, but only half succeed. Expected gap |0.9 - 0.5| = 0.4.
    pairs = [(0.9, True)] * 5 + [(0.9, False)] * 5
    buckets = pc.reliability_bins(pairs, n_bins=10)
    ece = pc.expected_calibration_error(buckets, total_n=len(pairs))
    assert ece == pytest.approx(0.4, abs=1e-9)


def test_analyse_filters_invalid_rows() -> None:
    rows = [
        _row(True, 0.9, True),
        _row(True, None, True),       # missing confidence
        _row(True, 0.3, None),        # missing label
        _row(False, 0.5, True),       # failed row
        _row(True, 1.2, True),        # clamps above 1.0
    ]
    report = pc.analyse(rows, n_bins=5)
    assert report["n_rows"] == 5
    assert report["n_ok"] == 4
    assert report["n_pairs"] == 2  # only first and last after filtering
    # Both pairs are positives so any non-empty bin has empirical_rate = 1.0
    non_empty = [b for b in report["bins"] if b["n"] > 0]
    assert all(b["empirical_rate"] == 1.0 for b in non_empty)


def test_empty_input_safe() -> None:
    report = pc.analyse([], n_bins=10)
    assert report["n_pairs"] == 0
    assert report["expected_calibration_error"] is None
