"""Tests for app/gym/rep_features.py on synthetic canonical frames.

Covers the per-field valid / degraded / unknown taxonomy end-to-end:
every assertion below names the exact status + reason_codes the module
must emit. Synthetic frames keep these tests free of MediaPipe / ffmpeg
so they run in the fast test-pose-core subset.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from app.gym.exercises_v0 import get_exercise
from app.gym.rep_features import (
    REP_FEATURES_SCHEMA_VERSION,
    RepFeaturesConfig,
    compute_rep_features,
    extract_rep_signal,
    feature_vectors_from_segment,
)
from app.gym.rep_segmenter import RepSpan, segment_reps

BACK_SQUAT = get_exercise("back_squat")
BENCH = get_exercise("bench_press")
PLANK = get_exercise("plank")
FARMER = get_exercise("farmer_carry")


def _obs(x: float, y: float, visibility: float = 0.95) -> dict[str, float]:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def _squat_frame(t: float, rep_period_s: float) -> dict[str, Any]:
    """Frame for a back squat: hip oscillates vertically, knee + ankle fixed-ish."""
    # hip.y moves between 0.45 (top) and 0.60 (bottom) with sine
    hip_y = 0.525 + 0.075 * math.sin(2 * math.pi * t / rep_period_s)
    return {
        "right_hip": _obs(0.50, hip_y),
        "left_hip": _obs(0.48, hip_y),
        "right_knee": _obs(0.50, 0.75),
        "left_knee": _obs(0.48, 0.75),
        "right_ankle": _obs(0.50, 0.95),
        "left_ankle": _obs(0.48, 0.95),
    }


def _bench_frame(t: float, rep_period_s: float) -> dict[str, Any]:
    """Frame for bench press: elbow angle oscillates between 90 and 170 deg."""
    # Build shoulder(fixed), elbow(fixed), wrist(moves) so angle varies.
    # Simpler: move elbow y to achieve target angle between S=(0.5,0.5) and W=(0.5,0.6).
    # Instead, keep geometry simple — place wrist at varying y so the
    # interior angle at elbow cycles.
    target = 130.0 - 40.0 * math.sin(2 * math.pi * t / rep_period_s)
    # Place S at (0.4, 0.5), E at (0.5, 0.5). Wrist sits at angle `target` from E.
    rad = math.radians(target)
    # Make wrist position so that angle S-E-W == target.
    # Vector from E to S is (-1, 0). Rotate by `target` around E.
    wx = 0.5 + math.cos(math.pi - rad) * 0.15
    wy = 0.5 + math.sin(math.pi - rad) * 0.15
    return {
        "right_shoulder": _obs(0.40, 0.50),
        "right_elbow": _obs(0.50, 0.50),
        "right_wrist": _obs(wx, wy),
        "left_shoulder": _obs(0.40, 0.50),
        "left_elbow": _obs(0.50, 0.50),
        "left_wrist": _obs(wx, wy),
    }


def _plank_frame(drift: float = 0.0) -> dict[str, Any]:
    return {
        "right_shoulder": _obs(0.25, 0.50 + drift),
        "left_shoulder": _obs(0.25, 0.50 + drift),
        "right_hip": _obs(0.50, 0.52 + drift),
        "left_hip": _obs(0.50, 0.52 + drift),
        "right_ankle": _obs(0.80, 0.54 + drift),
        "left_ankle": _obs(0.80, 0.54 + drift),
    }


def _make_squat_clip(n_reps: int, fps: float, rep_period_s: float) -> tuple[list[Any], np.ndarray]:
    n = int(round(n_reps * rep_period_s * fps)) + 1
    frames = [_squat_frame(i / fps, rep_period_s) for i in range(n)]
    sig, _miss = extract_rep_signal(frames, BACK_SQUAT)
    return frames, sig


# ----- extract_rep_signal --------------------------------------------------


def test_extract_signal_for_squat_matches_hip_y() -> None:
    frames, sig = _make_squat_clip(n_reps=4, fps=30.0, rep_period_s=2.0)
    # First frame hip_y == 0.525 + 0.075*sin(0) = 0.525
    assert sig[0] == pytest.approx(0.525, abs=1e-6)
    # Signal should have real variation
    assert float(np.max(sig) - np.min(sig)) > 0.1


def test_extract_signal_for_bench_returns_degrees() -> None:
    n = 60
    fps = 30.0
    frames = [_bench_frame(i / fps, 2.0) for i in range(n)]
    sig, miss = extract_rep_signal(frames, BENCH)
    assert not miss.any()
    # Degrees, so values should be between 90 and 170 roughly.
    usable = sig[np.isfinite(sig)]
    assert usable.min() > 60.0
    assert usable.max() < 180.0


def test_extract_signal_tolerates_enum_keys_and_joint_observation_values() -> None:
    """Non-string keys + JointObservation values must work without conversion."""
    from app.pose.canonical import CanonicalJointName, JointObservation

    frame = {
        CanonicalJointName.RIGHT_HIP: JointObservation(x=0.5, y=0.5, z=0.0, visibility=0.9),
        CanonicalJointName.LEFT_HIP: JointObservation(x=0.48, y=0.5, z=0.0, visibility=0.9),
        CanonicalJointName.RIGHT_KNEE: JointObservation(x=0.5, y=0.75, z=0.0, visibility=0.9),
        CanonicalJointName.LEFT_KNEE: JointObservation(x=0.48, y=0.75, z=0.0, visibility=0.9),
        CanonicalJointName.RIGHT_ANKLE: JointObservation(x=0.5, y=0.95, z=0.0, visibility=0.9),
        CanonicalJointName.LEFT_ANKLE: JointObservation(x=0.48, y=0.95, z=0.0, visibility=0.9),
    }
    sig, miss = extract_rep_signal([frame, frame], BACK_SQUAT)
    assert not miss.any()
    assert sig[0] == pytest.approx(0.5, abs=1e-6)


def test_extract_signal_missing_joints_produces_missingness() -> None:
    """Dropping rep_signal_joint from a frame must mark it missing."""
    good = _squat_frame(0.0, 2.0)
    bad = {k: v for k, v in good.items() if k != "right_hip"}
    sig, miss = extract_rep_signal([good, bad], BACK_SQUAT)
    assert miss[0] is np.False_ or not miss[0]
    assert bool(miss[1])


# ----- compute_rep_features shape ------------------------------------------


def test_feature_vector_has_all_v0_fields() -> None:
    frames, _ = _make_squat_clip(n_reps=3, fps=30.0, rep_period_s=2.0)
    rep = RepSpan(start_frame=0, end_frame=len(frames) - 1, peak_frame=len(frames) // 2, status="valid", reason_codes=())
    fv = compute_rep_features(rep, 0, frames, BACK_SQUAT, fps=30.0)
    expected = {
        "rep_duration_s",
        "eccentric_duration_s",
        "concentric_duration_s",
        "tempo_ratio_ecc_over_con",
        "signal_amplitude",
        "primary_joints_min_visibility",
        "primary_joints_missing_frac",
    }
    assert set(fv.features.keys()) == expected
    assert fv.schema_version == REP_FEATURES_SCHEMA_VERSION


def test_to_dict_is_json_serialisable() -> None:
    import json

    frames, _ = _make_squat_clip(n_reps=2, fps=30.0, rep_period_s=2.0)
    rep = RepSpan(start_frame=0, end_frame=30, peak_frame=15, status="valid", reason_codes=())
    fv = compute_rep_features(rep, 0, frames, BACK_SQUAT, fps=30.0)
    json.dumps(fv.to_dict())


# ----- duration / phase semantics ------------------------------------------


def test_peak_at_start_degrades_eccentric_phase() -> None:
    frames, _ = _make_squat_clip(n_reps=2, fps=30.0, rep_period_s=2.0)
    rep = RepSpan(start_frame=0, end_frame=60, peak_frame=0, status="valid", reason_codes=())
    fv = compute_rep_features(rep, 0, frames, BACK_SQUAT, fps=30.0)
    assert fv.features["eccentric_duration_s"].status == "unknown"
    assert "eccentric_missing" in fv.features["eccentric_duration_s"].reason_codes


def test_peak_at_end_degrades_concentric_phase() -> None:
    frames, _ = _make_squat_clip(n_reps=2, fps=30.0, rep_period_s=2.0)
    rep = RepSpan(start_frame=0, end_frame=60, peak_frame=60, status="valid", reason_codes=())
    fv = compute_rep_features(rep, 0, frames, BACK_SQUAT, fps=30.0)
    assert fv.features["concentric_duration_s"].status == "unknown"


def test_tempo_ratio_valid_when_both_phases_valid() -> None:
    frames, _ = _make_squat_clip(n_reps=2, fps=30.0, rep_period_s=2.0)
    rep = RepSpan(start_frame=0, end_frame=60, peak_frame=30, status="valid", reason_codes=())
    fv = compute_rep_features(rep, 0, frames, BACK_SQUAT, fps=30.0)
    tempo = fv.features["tempo_ratio_ecc_over_con"]
    assert tempo.status == "valid"
    assert tempo.value is not None
    assert 0.5 < tempo.value < 2.0


# ----- visibility / missingness semantics ----------------------------------


def test_low_visibility_degrades_visibility_field() -> None:
    fps = 30.0
    frames = [_squat_frame(i / fps, 2.0) for i in range(30)]
    # Drop all visibility to 0.3 (below threshold).
    low_frames: list[Any] = []
    for fr in frames:
        low_frames.append({k: {**v, "visibility": 0.3} for k, v in fr.items()})
    rep = RepSpan(start_frame=0, end_frame=29, peak_frame=15, status="valid", reason_codes=())
    fv = compute_rep_features(rep, 0, low_frames, BACK_SQUAT, fps=fps)
    vis = fv.features["primary_joints_min_visibility"]
    assert vis.status == "degraded"
    assert "low_visibility" in vis.reason_codes


def test_high_missingness_flags_missing_field() -> None:
    fps = 30.0
    # 30 frames; blank out the last 15 entirely.
    frames: list[Any] = [_squat_frame(i / fps, 2.0) for i in range(30)]
    frames[15:] = [None] * 15
    rep = RepSpan(start_frame=0, end_frame=29, peak_frame=15, status="valid", reason_codes=())
    fv = compute_rep_features(rep, 0, frames, BACK_SQUAT, fps=fps)
    miss = fv.features["primary_joints_missing_frac"]
    assert miss.status == "degraded"
    assert "high_missingness" in miss.reason_codes
    assert miss.value is not None and miss.value >= 0.5


# ----- amplitude semantics -------------------------------------------------


def test_low_amplitude_is_degraded_not_invalid() -> None:
    """A shallow rep is still a rep -- degrade the amplitude field only."""
    fps = 30.0
    # Hips barely move: amplitude way below min_amplitude_normalized default.
    n = 60
    frames = [
        {
            "right_hip": _obs(0.5, 0.525 + 0.002 * math.sin(2 * math.pi * i / 30)),
            "left_hip": _obs(0.48, 0.525 + 0.002 * math.sin(2 * math.pi * i / 30)),
            "right_knee": _obs(0.5, 0.75),
            "left_knee": _obs(0.48, 0.75),
            "right_ankle": _obs(0.5, 0.95),
            "left_ankle": _obs(0.48, 0.95),
        }
        for i in range(n)
    ]
    rep = RepSpan(start_frame=0, end_frame=n - 1, peak_frame=n // 2, status="valid", reason_codes=())
    fv = compute_rep_features(rep, 0, frames, BACK_SQUAT, fps=fps)
    amp = fv.features["signal_amplitude"]
    assert amp.status == "degraded"
    assert "low_amplitude" in amp.reason_codes
    assert amp.unit == "normalized_y"


def test_plank_amplitude_is_stable_when_drift_small() -> None:
    fps = 30.0
    # Drift 0.0001/frame over 60 frames -> total range ~0.006 << 0.02 threshold.
    frames = [_plank_frame(drift=0.0001 * i) for i in range(60)]
    rep = RepSpan(start_frame=0, end_frame=59, peak_frame=30, status="valid", reason_codes=("duration_hold",))
    fv = compute_rep_features(rep, 0, frames, PLANK, fps=fps)
    amp = fv.features["signal_amplitude"]
    assert amp.status == "valid"
    assert "stable_hold" in amp.reason_codes


def test_plank_amplitude_flags_unstable_hold() -> None:
    fps = 30.0
    # Large drift -> high amplitude -> unstable.
    frames = [_plank_frame(drift=0.01 * i) for i in range(60)]
    rep = RepSpan(start_frame=0, end_frame=59, peak_frame=30, status="valid", reason_codes=("duration_hold",))
    fv = compute_rep_features(rep, 0, frames, PLANK, fps=fps)
    amp = fv.features["signal_amplitude"]
    assert amp.status == "degraded"
    assert "unstable_hold" in amp.reason_codes


# ----- end-to-end with segmenter ------------------------------------------


def test_feature_vectors_from_segment_matches_rep_count() -> None:
    fps = 30.0
    frames, sig = _make_squat_clip(n_reps=4, fps=fps, rep_period_s=2.0)
    segment = segment_reps(sig, BACK_SQUAT, fps=fps)
    fvs = feature_vectors_from_segment(segment, frames, BACK_SQUAT)
    assert len(fvs) == len(segment.reps)
    for i, fv in enumerate(fvs):
        assert fv.rep_index == i
        assert fv.exercise_id == "back_squat"


# ----- config surface ------------------------------------------------------


def test_config_rides_with_result_and_is_overridable() -> None:
    frames, _ = _make_squat_clip(n_reps=2, fps=30.0, rep_period_s=2.0)
    rep = RepSpan(start_frame=0, end_frame=30, peak_frame=15, status="valid", reason_codes=())
    override = RepFeaturesConfig(visibility_degraded_threshold=0.99, min_amplitude_normalized=0.99)
    fv = compute_rep_features(rep, 0, frames, BACK_SQUAT, fps=30.0, config=override)
    assert fv.config.visibility_degraded_threshold == 0.99
    # With threshold 0.99, our 0.95-visibility frames degrade.
    assert fv.features["primary_joints_min_visibility"].status == "degraded"


# ----- validation ---------------------------------------------------------


def test_zero_fps_raises() -> None:
    rep = RepSpan(0, 10, 5, "valid", ())
    with pytest.raises(ValueError, match="fps"):
        compute_rep_features(rep, 0, [], BACK_SQUAT, fps=0.0)


def test_invalid_span_raises() -> None:
    rep = RepSpan(start_frame=10, end_frame=5, peak_frame=7, status="valid", reason_codes=())
    with pytest.raises(ValueError, match="RepSpan"):
        compute_rep_features(rep, 0, [], BACK_SQUAT, fps=30.0)


# ----- eccentric/concentric phase polarity ---------------------------------


def test_phase_polarity_squat_vs_pull() -> None:
    """start->peak is eccentric for a squat but concentric for a pull (curl).

    The segmenter peak is the bottom for a squat (eccentric end) but the
    contracted top for a curl (concentric end), so the phase labels must flip.
    """
    curl = get_exercise("dumbbell_bicep_curl")
    # peak near the start: start->peak = 5 frames, peak->end = 25 frames.
    rep = RepSpan(start_frame=0, end_frame=30, peak_frame=5, status="valid", reason_codes=())
    frames = [None] * 31
    fps = 30.0

    sq = compute_rep_features(rep, 0, frames, BACK_SQUAT, fps).features
    # Squat: start->peak (short, 5 frames) is the eccentric lowering.
    assert sq["eccentric_duration_s"].value == pytest.approx(5 / 30)
    assert sq["concentric_duration_s"].value == pytest.approx(25 / 30)

    cu = compute_rep_features(rep, 0, frames, curl, fps).features
    # Curl: start->peak is the concentric lift, so phases are swapped.
    assert cu["concentric_duration_s"].value == pytest.approx(5 / 30)
    assert cu["eccentric_duration_s"].value == pytest.approx(25 / 30)

    # tempo_ratio = ecc/con therefore inverts between the two movements.
    assert sq["tempo_ratio_ecc_over_con"].value == pytest.approx(5 / 25)
    assert cu["tempo_ratio_ecc_over_con"].value == pytest.approx(25 / 5)
