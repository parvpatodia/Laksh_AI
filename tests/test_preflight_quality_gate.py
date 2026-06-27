"""Deterministic tests for :mod:`app.preflight.quality_gate`.

Four fixtures:

1. ``test_full_visibility_in_frame_passes`` -- all joints visible, all in
   frame, fps=30. Result: passed=True, empty reason_codes.
2. ``test_low_visibility_fails`` -- mean visibility = 0.3 < 0.5 threshold.
   Result: passed=False, ``preflight_visibility_failed`` in reason_codes.
3. ``test_low_in_frame_ratio_fails`` -- joints outside the 0.05 margin on
   50% of frames. Result: passed=False, ``preflight_in_frame_failed``.
4. ``test_low_fps_fails`` -- fps_observed=20 < 25 threshold.
   Result: passed=False, ``preflight_fps_failed``.
5. ``test_empty_frames_gives_zero_ifr`` -- all frames None. No crash; both
   visibility and in_frame_ratio checks fail gracefully.
6. ``test_all_three_failing_codes_reported`` -- visibility + in_frame +
   fps all fail simultaneously; all three reason_codes present.
7. ``test_per_signal_actuals_present`` -- actuals dict has all three keys.
8. ``test_to_dict_is_json_compatible`` -- to_dict() has no nan/inf at
   threshold level (passed clip).

All inputs are synthetic plain dicts -- no MediaPipe import needed.
"""
from __future__ import annotations

import math

import pytest

from app.preflight.quality_gate import (
    FPS_FLOOR,
    IN_FRAME_RATIO_MIN,
    VISIBILITY_CORE_MIN,
    compute_preflight_metrics,
)

# ---------------------------------------------------------------------------
# Landmark builder helpers
# ---------------------------------------------------------------------------

# Core landmark indices (must match quality_gate._CORE_LANDMARK_INDICES).
_CORE_INDICES = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26)


def _lm(x: float = 0.5, y: float = 0.5, visibility: float = 0.9) -> dict[str, float]:
    """Build a plain-dict landmark compatible with ``_get_attr``."""
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def _full_frame(x: float = 0.5, y: float = 0.5, vis: float = 0.9) -> list[dict[str, float]]:
    """33-landmark list with core joints at (x, y) and given visibility."""
    frame: list[dict[str, float] | None] = [None] * 33
    for idx in _CORE_INDICES:
        frame[idx] = _lm(x, y, vis)  # type: ignore[call-overload]
    return frame  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Fixture 1: all-good clip passes
# ---------------------------------------------------------------------------


def test_full_visibility_in_frame_passes() -> None:
    """30 frames, all visible (0.9), all in frame (0.5, 0.5), fps=30."""
    frames = [_full_frame(0.5, 0.5, 0.9) for _ in range(30)]
    result = compute_preflight_metrics(frames, fps_observed=30.0)
    assert result.passed is True, result
    assert result.reason_codes == ()
    assert result.visibility_core >= VISIBILITY_CORE_MIN
    assert result.in_frame_ratio >= IN_FRAME_RATIO_MIN
    assert result.fps_observed == 30.0


# ---------------------------------------------------------------------------
# Fixture 2: low visibility fails
# ---------------------------------------------------------------------------


def test_low_visibility_fails() -> None:
    """Visibility = 0.3 < 0.5 threshold -> preflight_visibility_failed."""
    frames = [_full_frame(0.5, 0.5, 0.3) for _ in range(30)]
    result = compute_preflight_metrics(frames, fps_observed=30.0)
    assert result.passed is False
    assert "preflight_visibility_failed" in result.reason_codes
    assert result.visibility_core < VISIBILITY_CORE_MIN


# ---------------------------------------------------------------------------
# Fixture 3: low in-frame ratio fails
# ---------------------------------------------------------------------------


def test_low_in_frame_ratio_fails() -> None:
    """50% of frames have joints outside the 0.05 margin -> fails."""
    in_frame = [_full_frame(0.5, 0.5, 0.9) for _ in range(15)]
    # x=0.01 is outside the 0.05 margin -> not in-frame.
    out_frame = [_full_frame(0.01, 0.5, 0.9) for _ in range(15)]
    frames = in_frame + out_frame
    result = compute_preflight_metrics(frames, fps_observed=30.0)
    assert result.passed is False
    assert "preflight_in_frame_failed" in result.reason_codes
    assert result.in_frame_ratio == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# Fixture 4: low fps fails
# ---------------------------------------------------------------------------


def test_low_fps_fails() -> None:
    """fps_observed=20 < 25 -> preflight_fps_failed."""
    frames = [_full_frame() for _ in range(30)]
    result = compute_preflight_metrics(frames, fps_observed=20.0)
    assert result.passed is False
    assert "preflight_fps_failed" in result.reason_codes
    assert result.fps_observed == 20.0


# ---------------------------------------------------------------------------
# Fixture 5: all-None frames -> graceful zero
# ---------------------------------------------------------------------------


def test_empty_frames_gives_zero_ifr() -> None:
    """All frames are None -> no crash, visibility is nan, ifr=0."""
    result = compute_preflight_metrics([None] * 20, fps_observed=30.0)
    assert result.passed is False
    assert math.isnan(result.visibility_core)
    assert result.in_frame_ratio == 0.0
    assert "preflight_visibility_failed" in result.reason_codes
    assert "preflight_in_frame_failed" in result.reason_codes


# ---------------------------------------------------------------------------
# Fixture 6: all three signals fail simultaneously
# ---------------------------------------------------------------------------


def test_all_three_failing_codes_reported() -> None:
    """vis=0.3, out_frame 100%, fps=10 -> three reason codes."""
    frames = [_full_frame(0.01, 0.5, 0.3) for _ in range(30)]
    result = compute_preflight_metrics(frames, fps_observed=10.0)
    assert result.passed is False
    assert "preflight_visibility_failed" in result.reason_codes
    assert "preflight_in_frame_failed" in result.reason_codes
    assert "preflight_fps_failed" in result.reason_codes
    assert len(result.reason_codes) == 3


# ---------------------------------------------------------------------------
# Fixture 7: per_signal_actuals dict is complete
# ---------------------------------------------------------------------------


def test_per_signal_actuals_present() -> None:
    """All three actuals keys present on both pass and fail."""
    frames = [_full_frame() for _ in range(30)]
    result = compute_preflight_metrics(frames, fps_observed=30.0)
    assert "visibility_core" in result.per_signal_actuals
    assert "in_frame_ratio" in result.per_signal_actuals
    assert "fps_observed" in result.per_signal_actuals


# ---------------------------------------------------------------------------
# Fixture 8: to_dict() produces JSON-compatible output on pass
# ---------------------------------------------------------------------------


def test_to_dict_is_json_compatible() -> None:
    """to_dict() returns no nan/inf and has all expected keys."""
    import json

    frames = [_full_frame() for _ in range(30)]
    result = compute_preflight_metrics(frames, fps_observed=30.0)
    d = result.to_dict()
    # Must be JSON-serialisable (no NaN, no infinity).
    serialised = json.dumps(d)
    assert "NaN" not in serialised
    assert "Infinity" not in serialised
    assert "passed" in d
    assert "thresholds" in d
    assert d["thresholds"]["visibility_core"] == VISIBILITY_CORE_MIN
    assert d["thresholds"]["in_frame_ratio"] == IN_FRAME_RATIO_MIN
    assert d["thresholds"]["fps_floor"] == FPS_FLOOR


# ---------------------------------------------------------------------------
# Fixture 9: threshold constants match JSON file
# ---------------------------------------------------------------------------


def test_thresholds_match_json_file() -> None:
    """Python constants must equal evaluation/preflight_thresholds.json."""
    import json
    import pathlib

    json_path = (
        pathlib.Path(__file__).parent.parent
        / "evaluation"
        / "preflight_thresholds.json"
    )
    assert json_path.exists(), f"Missing: {json_path}"
    with json_path.open() as fh:
        data = json.load(fh)
    assert data["visibility_core"] == VISIBILITY_CORE_MIN, data
    assert data["in_frame_ratio"] == IN_FRAME_RATIO_MIN, data
    assert data["fps_floor"] == FPS_FLOOR, data


# ---------------------------------------------------------------------------
# Fixture 10: threshold overrides work
# ---------------------------------------------------------------------------


def test_threshold_overrides_respected() -> None:
    """Caller-supplied threshold overrides must be used."""
    # Visibility = 0.6, normally passes 0.5 threshold.
    # With override of 0.7, it must fail.
    frames = [_full_frame(vis=0.6) for _ in range(30)]
    result = compute_preflight_metrics(
        frames, fps_observed=30.0, visibility_min=0.7
    )
    assert result.passed is False
    assert "preflight_visibility_failed" in result.reason_codes
