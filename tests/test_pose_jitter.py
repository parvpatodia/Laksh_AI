"""Smoke tests for scripts/pose_jitter.py (temporal sanity report)."""
from __future__ import annotations

import pytest

from scripts import pose_jitter as pj


def _row(ok: bool, jitter: float | None, det: float | None) -> dict:
    return {
        "ok": ok,
        "hip_mid_displacement_median_norm": jitter,
        "detection_rate": det,
        "pose_usable_heuristic": ok,
        "video_path": "/clips/x.mp4",
        "reason_codes": [],
    }


def test_tukey_fence_small_sample_returns_none() -> None:
    assert pj.tukey_upper_fence([]) is None
    assert pj.tukey_upper_fence([0.1]) is None


def test_tukey_fence_basic() -> None:
    xs = [0.01, 0.02, 0.03, 0.04, 0.05]
    fence = pj.tukey_upper_fence(xs, k=1.5)
    assert fence is not None
    # Q3 = 0.04, Q1 = 0.02, IQR = 0.02, upper = 0.04 + 1.5*0.02 = 0.07
    assert pytest.approx(fence, rel=1e-6) == 0.07


def test_summarise_flags_outlier() -> None:
    # Realistic spread (non-zero IQR) so the Tukey fence is above the bulk.
    jitter_values = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050]
    rows = [_row(True, v, 0.95) for v in jitter_values]
    # Planted outlier far above the distribution.
    rows.append(_row(True, 1.5, 0.3))

    report = pj.summarise(rows, k=1.5)
    assert report["n_rows"] == 10
    assert report["n_ok"] == 10
    assert report["hip_mid_displacement_median_norm"]["n"] == 10

    outliers = report["outliers_above_fence"]
    paths = [o["hip_mid_displacement_median_norm"] for o in outliers]
    assert 1.5 in paths
    # The planted extreme must be caught; bulk points should not be.
    bulk_values = set(jitter_values)
    assert not (bulk_values & set(paths))


def test_summarise_skips_failed_rows() -> None:
    rows = [
        _row(True, 0.03, 0.95),
        {"ok": False, "error": "decode failed"},
        _row(True, 0.04, 0.92),
    ]
    report = pj.summarise(rows)
    assert report["n_rows"] == 3
    assert report["n_ok"] == 2
    assert report["hip_mid_displacement_median_norm"]["n"] == 2


def test_summarise_empty_is_safe() -> None:
    report = pj.summarise([])
    assert report["n_rows"] == 0
    assert report["n_ok"] == 0
    assert report["hip_mid_displacement_median_norm"]["upper_fence"] is None
    assert report["outliers_above_fence"] == []
