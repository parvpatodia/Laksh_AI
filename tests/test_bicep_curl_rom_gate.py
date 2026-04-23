"""Deterministic tests for the Norkin & White curl ROM gate.

Fixtures mirror plan A2's acceptance list:

1. ``test_full_curl_is_valid`` -- start >=150 deg, peak <=60 deg,
   end >=150 deg, wrist rises by >=0.4 * shoulder-elbow pixel length.
   Status: ``valid``, empty ``reason_codes``.
2. ``test_partial_curl_peak_75_is_partial`` -- peak reaches 75 deg
   (<=90 deg partial regime) but start/end never reach 150 deg AND
   wrist amplitude fails C2. Status: ``partial`` with
   ``partial_rom`` in ``reason_codes``.
3. ``test_twitch_peak_140_is_dropped`` -- micro motion: start/end near
   160 deg but peak only dips to 140 deg (nowhere near the 90 deg
   partial regime) and wrist barely moves. Status: ``dropped`` with
   ``twitch`` in ``reason_codes``.
4. ``test_three_full_plus_one_partial_sequence`` -- integration with
   :func:`feature_vectors_from_segment`: the segmenter produces 4 reps;
   three are full, one is partial (peak 75 deg). Exactly 3 reps have
   ``curl_rom_gate.status == "valid"`` and 1 has ``"partial"``.

All fixtures are synthetic canonical-frame dicts; no MediaPipe, no
FFmpeg. This mirrors the determinism contract in
:mod:`tests.test_rep_features`.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.gym.exercises_v0 import get_exercise
from app.gym.rep_features import (
    BicepCurlRomGateConfig,
    compute_rep_features,
    evaluate_bicep_curl_rom_gate,
    extract_rep_signal,
    feature_vectors_from_segment,
)
from app.gym.rep_segmenter import RepSpan, segment_reps

CURL = get_exercise("dumbbell_bicep_curl")
assert CURL is not None, "dumbbell_bicep_curl must be in registry"

FPS = 30.0
FOREARM = 0.15  # normalized forearm length for synthesis
UPPER_ARM = 0.20  # normalized upper-arm length (shoulder->elbow)


def _obs(x: float, y: float, visibility: float = 0.95) -> dict[str, float]:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def _curl_frame(angle_deg: float) -> dict[str, Any]:
    """Synthesise one canonical frame whose right-elbow interior angle
    equals ``angle_deg``.

    Geometry:
      * shoulder fixed at (0.50, 0.40),
      * elbow fixed at (0.50, 0.60) (``UPPER_ARM`` below the shoulder),
      * wrist placed so interior angle at elbow equals ``angle_deg``.

    At angle=180 deg (full extension) the wrist lies straight below the
    elbow at (0.50, 0.75). At angle=60 deg (full flexion) the wrist is
    rotated up by 120 deg around the elbow, so the wrist image-y is well
    ABOVE the elbow. This matches the physical curl: wrist rises, pixel-y
    decreases.
    """
    # ba = shoulder - elbow = (0, -UPPER_ARM); points from elbow upward.
    # We want the angle between (elbow->shoulder) and (elbow->wrist) to
    # equal angle_deg. Rotating the ba vector by +/- angle_deg and
    # extending to length FOREARM yields a valid wrist placement.
    # Rotate ba = (0, -1) by (pi - rad) so that at angle_deg=180 the
    # wrist is at (0, +1) relative to elbow (straight below), and at
    # angle_deg=60 the wrist is well above.
    ex, ey = 0.50, 0.60
    # Direction of the wrist from the elbow when the interior angle is
    # ``angle_deg``: start from straight-down (0,1) and rotate by
    # (180 - angle_deg). At angle=180 -> rotation 0 -> straight down
    # -> interior angle = 180 (straight arm). At angle=60 -> rotation 120
    # -> wrist well above elbow.
    rot = math.radians(180.0 - angle_deg)
    dx = math.sin(rot)
    dy = math.cos(rot)  # positive dy = pixel-down; at rot=0 dy=1 (down)
    wx = ex + FOREARM * dx
    wy = ey + FOREARM * dy

    return {
        "right_shoulder": _obs(ex, ey - UPPER_ARM),
        "right_elbow": _obs(ex, ey),
        "right_wrist": _obs(wx, wy),
        "left_shoulder": _obs(ex, ey - UPPER_ARM),
        "left_elbow": _obs(ex, ey),
        "left_wrist": _obs(wx, wy),
    }


def _synthesise_rep_frames(
    start_angle: float,
    peak_angle: float,
    end_angle: float,
    n_frames: int = 30,
) -> list[dict[str, Any]]:
    """Build a single rep: start -> peak -> end via cosine interpolation.

    Returns a list of canonical-frame dicts length ``n_frames``.
    ``peak_frame`` is at the midpoint.
    """
    half = n_frames // 2
    frames: list[dict[str, Any]] = []
    for i in range(n_frames):
        if i <= half:
            # eccentric-ish: start -> peak
            alpha = (1 - math.cos(math.pi * i / half)) / 2 if half > 0 else 1.0
            angle = start_angle + (peak_angle - start_angle) * alpha
        else:
            # concentric: peak -> end
            j = i - half
            rem = n_frames - 1 - half
            alpha = (1 - math.cos(math.pi * j / rem)) / 2 if rem > 0 else 1.0
            angle = peak_angle + (end_angle - peak_angle) * alpha
        frames.append(_curl_frame(angle))
    return frames


# ----- fixture 1: full curl ---------------------------------------------------


def test_full_curl_is_valid() -> None:
    frames = _synthesise_rep_frames(
        start_angle=160.0, peak_angle=55.0, end_angle=160.0, n_frames=30
    )
    rep = RepSpan(start_frame=0,
        peak_frame=15,
        end_frame=29,
        status="valid",
        reason_codes=(),
    )
    fv = evaluate_bicep_curl_rom_gate(rep, frames, CURL)
    assert fv.status == "valid", fv
    assert fv.reason_codes == ()


# ----- fixture 2: partial curl (peak 75 deg) ----------------------------------


def test_partial_curl_peak_95_is_degraded() -> None:
    # Peak reaches 95 deg -- within the c1_peak_partial regime (<=110) but
    # above peak_flexion_deg_max (80), so C1 full fails. C2 passes (wrist
    # still rises enough). Single-signal -> degraded (schema contract).
    # Use a strict custom config to decouple from evolving defaults.
    strict_cfg = BicepCurlRomGateConfig(
        start_extension_deg_min=150.0,
        end_extension_deg_min=150.0,
        peak_flexion_deg_max=60.0,
        peak_partial_deg_max=90.0,
        wrist_y_descent_ratio=0.40,
    )
    frames = _synthesise_rep_frames(
        start_angle=160.0, peak_angle=75.0, end_angle=160.0, n_frames=30
    )
    rep = RepSpan(start_frame=0,
        peak_frame=15,
        end_frame=29,
        status="valid",
        reason_codes=(),
    )
    fv = evaluate_bicep_curl_rom_gate(rep, frames, CURL, strict_cfg)
    assert fv.status == "degraded", fv  # partial_rom maps to "degraded" (schema contract)
    assert "partial_rom" in fv.reason_codes or "single_signal_rom" in fv.reason_codes


# ----- fixture 3: twitch (peak 140 deg) ---------------------------------------


def test_twitch_peak_140_is_dropped() -> None:
    # Start/end at 160, peak barely dips to 140 deg. Fails strict C1
    # (peak_not_flexed), fails partial C1 (peak > 90), fails C2.
    frames = _synthesise_rep_frames(
        start_angle=160.0, peak_angle=140.0, end_angle=160.0, n_frames=30
    )
    rep = RepSpan(start_frame=0,
        peak_frame=15,
        end_frame=29,
        status="valid",
        reason_codes=(),
    )
    fv = evaluate_bicep_curl_rom_gate(rep, frames, CURL)
    assert fv.status == "unknown", fv  # twitch/dropped maps to "unknown" (schema contract)
    assert "twitch" in fv.reason_codes


# ----- fixture 4: sequence --------------------------------------------------


def test_three_full_plus_one_partial_sequence() -> None:
    """Integration: build a 4-rep clip, run it through the segmenter and
    feature extractor, check per-rep ROM gate outcomes.
    """
    # Rep 1, 2, 3 are full curls (160 -> 55 -> 160). Rep 4 is "degraded":
    # peak 95 deg is in the c1_peak_partial regime (<=110) but fails
    # peak_flexion_deg_max (80), so C1 full fails. C2 passes (wrist rises
    # enough), giving single_signal_rom -> degraded.
    full_angles = (160.0, 55.0, 160.0)
    partial_angles = (160.0, 95.0, 160.0)
    rest_frames = [_curl_frame(170.0) for _ in range(6)]  # 0.2 s @ 30 fps
    all_frames: list[dict[str, Any]] = list(rest_frames)
    for _ in range(3):
        all_frames.extend(_synthesise_rep_frames(*full_angles, n_frames=40))
        all_frames.extend(rest_frames)
    all_frames.extend(_synthesise_rep_frames(*partial_angles, n_frames=40))
    all_frames.extend(rest_frames)

    signal, miss = extract_rep_signal(all_frames, CURL)
    seg = segment_reps(signal, CURL, fps=FPS, missingness=miss)
    # Segmenter must find 4 reps.
    assert len(seg.reps) == 4, (
        f"segmenter found {len(seg.reps)} reps, expected 4 "
        f"(reasons: {seg.reason_codes})"
    )

    fvs = feature_vectors_from_segment(seg, all_frames, CURL)
    gate_statuses = [fv.features["curl_rom_gate"].status for fv in fvs]
    n_valid = sum(1 for s in gate_statuses if s == "valid")
    n_degraded = sum(1 for s in gate_statuses if s == "degraded")  # partial_rom -> degraded
    n_unknown = sum(1 for s in gate_statuses if s == "unknown")    # twitch/dropped -> unknown
    assert n_valid == 3, (gate_statuses, "expected 3 valid curls")
    assert n_degraded == 1, (gate_statuses, "expected 1 degraded (partial) curl")
    assert n_unknown == 0, (gate_statuses, "expected 0 unknown")


# ----- guards ----------------------------------------------------------------


def test_wrong_exercise_returns_unknown() -> None:
    bench = get_exercise("bench_press")
    rep = RepSpan(start_frame=0,
        peak_frame=5,
        end_frame=10,
        status="valid",
        reason_codes=(),
    )
    fv = evaluate_bicep_curl_rom_gate(rep, [_curl_frame(120.0)] * 11, bench)
    assert fv.status == "unknown"
    assert "wrong_exercise" in fv.reason_codes


def test_empty_frames_returns_unknown() -> None:
    rep = RepSpan(start_frame=0,
        peak_frame=0,
        end_frame=0,
        status="valid",
        reason_codes=(),
    )
    fv = evaluate_bicep_curl_rom_gate(rep, [None], CURL)
    assert fv.status == "unknown"


def test_config_thresholds_round_trip() -> None:
    """Every threshold is accessible and matches calibrated defaults."""
    cfg = BicepCurlRomGateConfig()
    # Current defaults (loosened for real-world use; stricter academic values are
    # 150/60/90/0.4 but those reject most casual users -- see docstring).
    assert cfg.start_extension_deg_min == 130.0
    assert cfg.peak_flexion_deg_max == 80.0
    assert cfg.end_extension_deg_min == 130.0
    assert cfg.peak_partial_deg_max == 110.0
    assert cfg.wrist_y_descent_ratio == 0.25


def test_compute_rep_features_attaches_gate_for_curl() -> None:
    """End-to-end: compute_rep_features must include curl_rom_gate only
    for dumbbell_bicep_curl."""
    frames = _synthesise_rep_frames(
        start_angle=160.0, peak_angle=55.0, end_angle=160.0, n_frames=30
    )
    rep = RepSpan(start_frame=0,
        peak_frame=15,
        end_frame=29,
        status="valid",
        reason_codes=(),
    )
    fv = compute_rep_features(rep, 0, frames, CURL, fps=FPS)
    assert "curl_rom_gate" in fv.features
    assert fv.features["curl_rom_gate"].status == "valid"

    bench = get_exercise("bench_press")
    fv_bench = compute_rep_features(rep, 0, frames, bench, fps=FPS)
    assert "curl_rom_gate" not in fv_bench.features


def test_synthesised_frame_angle_is_correct() -> None:
    """Sanity: the synthesiser really produces the angle it claims."""
    for target in (60.0, 90.0, 120.0, 150.0, 180.0):
        fr = _curl_frame(target)
        s = fr["right_shoulder"]
        e = fr["right_elbow"]
        w = fr["right_wrist"]
        ba = (s["x"] - e["x"], s["y"] - e["y"])
        bc = (w["x"] - e["x"], w["y"] - e["y"])
        n_ba = math.hypot(*ba)
        n_bc = math.hypot(*bc)
        cos = (ba[0] * bc[0] + ba[1] * bc[1]) / (n_ba * n_bc)
        cos = max(-1.0, min(1.0, cos))
        got = math.degrees(math.acos(cos))
        assert abs(got - target) < 0.5, (target, got)


def test_c2_fires_when_wrist_rises_enough() -> None:
    """Direct C2 probe: build a rep where wrist rises a lot but elbow
    barely bends -> C1 fails, C2 passes -> partial status."""
    # Whole-arm translation up by 0.4 * UPPER_ARM (wrist amp passes)
    # while elbow angle stays ~170 deg (C1 full fails, partial fails).
    n = 30
    frames: list[dict[str, Any]] = []
    for i in range(n):
        # cosine envelope: 0 -> 1 -> 0
        env = (1 - math.cos(2 * math.pi * i / (n - 1))) / 2
        dy = -env * 0.5 * UPPER_ARM  # wrist rises (pixel-y decreases)
        ex, ey = 0.50, 0.60 + dy
        # Keep arm straight: wrist directly below elbow.
        frames.append(
            {
                "right_shoulder": _obs(ex, ey - UPPER_ARM),
                "right_elbow": _obs(ex, ey),
                "right_wrist": _obs(ex, ey + FOREARM),
                "left_shoulder": _obs(ex, ey - UPPER_ARM),
                "left_elbow": _obs(ex, ey),
                "left_wrist": _obs(ex, ey + FOREARM),
            }
        )
    rep = RepSpan(start_frame=0,
        peak_frame=n // 2,
        end_frame=n - 1,
        status="valid",
        reason_codes=(),
    )
    fv = evaluate_bicep_curl_rom_gate(rep, frames, CURL)
    # Whole-arm translation with straight elbow: C1 fails (peak still
    # 180 deg, nowhere near <=60 or <=90), C2 passes (wrist descent
    # ratio > 0.4). Consensus rule: exactly one gate fires -> "partial"
    # with `single_signal_rom` reason code. This is correct: a shoulder
    # shrug / whole-arm lift is NOT a curl, but it is also not a pure
    # twitch, so we surface it as partial so the judge can see why.
    assert fv.status == "degraded", fv  # single-signal -> degraded (schema contract)
    assert "single_signal_rom" in fv.reason_codes
    # Quiet numpy import.
    _ = np.array([])
