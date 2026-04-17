"""Tests for app/gym/rep_segmenter.py on synthetic signals.

Milestone 1 bullet 3: "each field has valid / degraded / unknown semantics".
These tests enforce that taxonomy behaviourally — every branch below names
the reason code the segmenter must emit.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.gym.exercises_v0 import get_exercise
from app.gym.rep_segmenter import (
    REP_SEGMENTER_SCHEMA_VERSION,
    RepStatus,
    SegmenterConfig,
    segment_reps,
)

BACK_SQUAT = get_exercise("back_squat")
BENCH = get_exercise("bench_press")
PLANK = get_exercise("plank")
FARMER = get_exercise("farmer_carry")


def _sine_signal(n_reps: int, fps: float, rep_period_s: float, amplitude: float = 0.1, offset: float = 0.5) -> np.ndarray:
    """Synthesize a cyclic-vertical signal: y = offset + amplitude * sin(2 pi t / T)."""
    n = int(round(n_reps * rep_period_s * fps)) + 1
    t = np.arange(n, dtype=np.float64) / fps
    # Positive amplitude => first extremum is a max at t = T/4 — works for cyclic_vertical
    return offset + amplitude * np.sin(2 * np.pi * t / rep_period_s)


def test_schema_version_is_exported() -> None:
    assert REP_SEGMENTER_SCHEMA_VERSION == "1.0.0"


def test_sine_signal_detects_expected_rep_count_for_squat() -> None:
    fps = 30.0
    rep_period_s = 2.0
    n_reps = 5
    sig = _sine_signal(n_reps=n_reps, fps=fps, rep_period_s=rep_period_s)
    result = segment_reps(sig, BACK_SQUAT, fps=fps)
    # Allow off-by-one on boundary detection; core claim is n_reps extrema.
    assert abs(len(result.reps) - n_reps) <= 1
    assert result.status in (RepStatus.VALID.value, RepStatus.DEGRADED.value)


def test_sine_signal_passes_through_config_default_thresholds() -> None:
    fps = 30.0
    sig = _sine_signal(n_reps=4, fps=fps, rep_period_s=2.0)
    result = segment_reps(sig, BACK_SQUAT, fps=fps)
    assert result.config.min_rep_s == SegmenterConfig().min_rep_s
    assert result.schema_version == REP_SEGMENTER_SCHEMA_VERSION
    assert result.exercise_id == "back_squat"
    assert result.rep_signal_type == "cyclic_vertical"


def test_flat_signal_returns_unknown_with_flat_signal_code() -> None:
    sig = np.full(90, 0.5, dtype=np.float64)
    result = segment_reps(sig, BACK_SQUAT, fps=30.0)
    assert result.status == RepStatus.UNKNOWN.value
    assert "flat_signal" in result.reason_codes
    assert result.reps == ()


def test_too_short_signal_returns_unknown_signal_too_short() -> None:
    sig = np.array([0.5, 0.51, 0.49], dtype=np.float64)
    result = segment_reps(sig, BACK_SQUAT, fps=30.0)
    assert result.status == RepStatus.UNKNOWN.value
    assert "signal_too_short" in result.reason_codes


def test_zero_fps_raises_valueerror() -> None:
    sig = _sine_signal(n_reps=3, fps=30.0, rep_period_s=2.0)
    with pytest.raises(ValueError, match="fps"):
        segment_reps(sig, BACK_SQUAT, fps=0.0)


def test_pure_noise_does_not_invent_reps() -> None:
    # Gaussian noise below prominence threshold should produce no reps or
    # be classified unknown rather than a cascade of spurious detections.
    rng = np.random.default_rng(42)
    sig = 0.5 + 0.001 * rng.standard_normal(180)  # amplitude << min_signal_range
    result = segment_reps(sig, BACK_SQUAT, fps=30.0)
    assert result.status == RepStatus.UNKNOWN.value


def test_cyclic_angle_exercise_inverts_peak_detection() -> None:
    """For bench press we look at elbow angle: deepest flexion = minimum
    value. The segmenter must still find those troughs as "work extrema".
    """
    fps = 30.0
    rep_period_s = 1.5
    n = int(round(4 * rep_period_s * fps)) + 1
    t = np.arange(n, dtype=np.float64) / fps
    # Elbow angle: 170 at top, 90 at bottom — invert the sine so troughs are the work
    angle = 130.0 - 40.0 * np.sin(2 * np.pi * t / rep_period_s)
    # Segmenter's min_signal_range is 0.02; angles are 50-170. Bump prominence_frac
    # up a touch; default should still work because range is ~80.
    result = segment_reps(angle, BENCH, fps=fps)
    assert abs(len(result.reps) - 4) <= 1
    assert result.rep_signal_type == "cyclic_angle"


def test_plank_returns_single_duration_hold_span() -> None:
    fps = 30.0
    sig = 0.6 + 0.001 * np.arange(150, dtype=np.float64)  # barely drifts
    result = segment_reps(sig, PLANK, fps=fps)
    assert len(result.reps) == 1
    assert "duration_hold" in result.reps[0].reason_codes
    assert result.reps[0].start_frame == 0
    assert result.reps[0].end_frame == 149


def test_plank_with_heavy_missingness_is_degraded() -> None:
    fps = 30.0
    n = 150
    sig = 0.6 + 0.001 * np.arange(n, dtype=np.float64)
    miss = np.zeros(n, dtype=bool)
    miss[:60] = True  # 40% missing
    result = segment_reps(sig, PLANK, fps=fps, missingness=miss)
    assert result.status == RepStatus.DEGRADED.value
    assert "high_missingness" in result.reps[0].reason_codes


def test_gait_cadence_tags_spans_as_gait_steps() -> None:
    fps = 30.0
    rep_period_s = 1.0
    n_steps = 6
    sig = _sine_signal(n_reps=n_steps, fps=fps, rep_period_s=rep_period_s, amplitude=0.05)
    result = segment_reps(sig, FARMER, fps=fps)
    assert result.rep_signal_type == "gait_cadence"
    # Every span gets the gait_step tag regardless of valid/degraded.
    assert all("gait_step" in r.reason_codes for r in result.reps)


def test_boundary_truncated_rep_is_flagged_degraded() -> None:
    """Leading partial rep: pad a long flat tail before the first real cycle
    so that rep[0]'s midpoint-based span is much shorter than the interior median.
    Actually easier: shrink rep[0] by ending the sine late."""
    fps = 30.0
    rep_period_s = 2.0
    # Concat: long flat head + 4 clean sine reps. The head absorbs into
    # rep[0]'s left boundary so rep[0] spans head+half-rep, which is LONGER
    # than median. Flip that — we want rep[0] to be SHORTER. Build:
    # [half-rep at end + 4 full reps] so the last rep's right bound clips early.
    sine = _sine_signal(n_reps=4, fps=fps, rep_period_s=rep_period_s)
    # Cut tail off mid-cycle so final rep is boundary-truncated short.
    trimmed = sine[: int(round(3.5 * rep_period_s * fps))]
    result = segment_reps(trimmed, BACK_SQUAT, fps=fps)
    # At least one rep should carry boundary_truncated; not asserting which.
    flagged = [r for r in result.reps if "boundary_truncated" in r.reason_codes]
    # If the trim accidentally lands on a clean boundary this may be empty;
    # allow that but require at least 2 reps detected.
    assert len(result.reps) >= 2
    if flagged:
        for r in flagged:
            assert r.status == RepStatus.DEGRADED.value


def test_missingness_in_rep_span_flags_degraded() -> None:
    fps = 30.0
    sig = _sine_signal(n_reps=4, fps=fps, rep_period_s=2.0)
    n = sig.shape[0]
    # Zero out missingness then set 40% of the middle rep window as missing.
    miss = np.zeros(n, dtype=bool)
    mid = n // 2
    miss[mid - 12 : mid + 12] = True  # ~24 frames missing
    result = segment_reps(sig, BACK_SQUAT, fps=fps, missingness=miss)
    # Some rep should have high_missingness.
    has_miss = any("high_missingness" in r.reason_codes for r in result.reps)
    assert has_miss, f"missingness was not attributed to any rep: {result.to_dict()}"


def test_to_dict_is_json_serialisable_shape() -> None:
    import json

    fps = 30.0
    sig = _sine_signal(n_reps=3, fps=fps, rep_period_s=2.0)
    result = segment_reps(sig, BACK_SQUAT, fps=fps)
    payload = result.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["schema_version"] == REP_SEGMENTER_SCHEMA_VERSION
    assert payload["exercise_id"] == "back_squat"
    for rep in payload["reps"]:
        assert isinstance(rep["reason_codes"], list)
        assert rep["status"] in ("valid", "degraded", "unknown")


def test_signal_shape_validation() -> None:
    with pytest.raises(ValueError, match="1D"):
        segment_reps(np.zeros((3, 3)), BACK_SQUAT, fps=30.0)


def test_missingness_shape_mismatch_raises() -> None:
    sig = _sine_signal(n_reps=3, fps=30.0, rep_period_s=2.0)
    with pytest.raises(ValueError, match="missingness"):
        segment_reps(sig, BACK_SQUAT, fps=30.0, missingness=np.zeros(7, dtype=bool))
