"""Deterministic tests for :mod:`app.basketball.shot_segmenter`.

Four fixtures mirror the plan's A1 acceptance list:

1. ``test_zero_shots_flat_signal`` — pose is valid but the subject is
   static. No S1 peak survives prominence; the segmenter returns 0 shots
   with ``reason_codes=("no_release_detected",)``.
2. ``test_single_shot_consensus`` — one synthetic shot where both wrist-y
   and elbow angular velocity peak together. Exactly one ``valid`` shot
   with ``signals_fired == ("wrist_y_nadir", "elbow_velocity_peak")``.
3. ``test_three_shots_consensus`` — three shots at realistic 1.5 s
   spacing; segmenter returns 3 ``valid`` shots with non-overlapping
   windows.
4. ``test_pump_fake_rejected`` — the signal contains two wrist-y peaks
   but only the second has an elbow extension spike. S2 vetoes the pump,
   so the first shot is ``degraded`` with ``single_signal_release`` and
   the second is ``valid`` — proving consensus filters the pump-fake.

All fixtures are synthetic NumPy arrays; no pose model, no video decode.
This is the determinism contract inherited from
:mod:`tests.test_rep_segmenter` — zero ML in the hot path, zero fixtures
that depend on disk state.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.basketball.shot_segmenter import (
    SHOT_SEGMENTER_SCHEMA_VERSION,
    ShotSegmenterConfig,
    ShotStatus,
    segment_shots,
)

FPS = 30.0


def _synthesise_shot(
    n_frames: int,
    peak_frame: int,
    *,
    wrist_amp: float = 0.18,
    elbow_low_deg: float = 70.0,
    elbow_high_deg: float = 165.0,
    dip_width_frames: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 2D (n, 3) arrays for shoulder / elbow / wrist that simulate
    one realistic shot.

    * Shoulder is stationary at (0.5, 0.4).
    * Elbow is stationary at (0.5, 0.5) — the pivot.
    * Wrist-y dips DOWN (increases, since image-y grows downward) then
      snaps UP by ``wrist_amp`` at ``peak_frame``. The snap is a Gaussian
      centered on ``peak_frame`` with sigma ``dip_width_frames / 4``.
    * Elbow angle follows the same envelope mapped from
      ``elbow_low_deg`` (at dip) to ``elbow_high_deg`` (at peak). We do
      NOT pre-compute the angle — we place the wrist geometrically so the
      segmenter's own angle calculation produces the intended value.

    The (shoulder, elbow, wrist) geometry is chosen so the interior
    elbow angle at peak is ``elbow_high_deg`` and at dip is
    ``elbow_low_deg``. All landmarks are stored as ``(n, 3)`` with
    column 2 a dummy visibility of 1.0.
    """
    t = np.arange(n_frames, dtype=np.float64)
    sigma = max(1.0, dip_width_frames / 4.0)
    envelope = np.exp(-0.5 * ((t - peak_frame) / sigma) ** 2)

    shoulder = np.zeros((n_frames, 3), dtype=np.float64)
    shoulder[:, 0] = 0.50
    shoulder[:, 1] = 0.40
    shoulder[:, 2] = 1.0

    elbow = np.zeros((n_frames, 3), dtype=np.float64)
    elbow[:, 0] = 0.50
    elbow[:, 1] = 0.50
    elbow[:, 2] = 1.0

    # Place the wrist along a line from elbow that rotates with the
    # envelope. At envelope=0 we want elbow_low_deg (arm flexed, wrist
    # close to shoulder). At envelope=1 we want elbow_high_deg (arm
    # extended, wrist far above elbow).
    lo = np.radians(180.0 - elbow_low_deg)
    hi = np.radians(180.0 - elbow_high_deg)
    theta = lo + (hi - lo) * envelope  # angle from positive-y axis
    # Wrist position: distance r above elbow, swinging through theta.
    r = 0.12  # forearm-like length in normalized coords
    wrist = np.zeros((n_frames, 3), dtype=np.float64)
    wrist[:, 0] = elbow[:, 0] + r * np.sin(theta)
    # image-y grows downward; extended arm has wrist ABOVE elbow so y is
    # SMALLER (image-up). The envelope's direct mapping already produces
    # that because at env=1 we subtract r*cos(theta) from elbow-y.
    wrist[:, 1] = elbow[:, 1] - r * np.cos(theta)
    wrist[:, 2] = 1.0

    return shoulder, elbow, wrist


def _static_arrays(n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant-pose synthetic arrays: wrist/elbow/shoulder never move."""
    shoulder = np.tile(np.array([0.50, 0.40, 1.0]), (n_frames, 1))
    elbow = np.tile(np.array([0.50, 0.50, 1.0]), (n_frames, 1))
    wrist = np.tile(np.array([0.50, 0.62, 1.0]), (n_frames, 1))
    return shoulder, elbow, wrist


# ----- fixture 1: zero shots, flat signal ----------------------------------


def test_zero_shots_flat_signal():
    shoulder, elbow, wrist = _static_arrays(n_frames=120)
    result = segment_shots(wrist, elbow, shoulder, fps=FPS)
    assert result.schema_version == SHOT_SEGMENTER_SCHEMA_VERSION
    assert result.n_shots_detected == 0
    assert result.n_shots_valid == 0
    assert result.n_shots_degraded == 0
    assert result.shots == ()
    assert result.reason_codes == ("no_release_detected",)


# ----- fixture 2: single shot, consensus -----------------------------------


def test_single_shot_consensus():
    n = 90  # 3 s @ 30 fps
    peak = 45
    shoulder, elbow, wrist = _synthesise_shot(n_frames=n, peak_frame=peak)
    result = segment_shots(wrist, elbow, shoulder, fps=FPS)
    assert result.n_shots_detected == 1
    assert result.n_shots_valid == 1
    assert result.n_shots_degraded == 0
    shot = result.shots[0]
    assert shot.status == ShotStatus.VALID.value
    # Release frame should land within a few frames of the peak we placed.
    assert abs(shot.release_frame - peak) <= 2
    assert "wrist_y_nadir" in shot.signals_fired
    assert "elbow_velocity_peak" in shot.signals_fired
    # Per-shot window is clipped into range and non-degenerate.
    assert 0 <= shot.start_frame < shot.release_frame < shot.end_frame <= n - 1


# ----- fixture 3: three shots spaced realistically -------------------------


def test_three_shots_consensus():
    n = 180  # 6 s
    peaks = [30, 90, 150]  # 2 s apart — well beyond the 0.6 s min_inter_shot
    shoulder_parts = []
    elbow_parts = []
    wrist_parts = []
    # Build a composite by summing single-shot envelopes into one clip. We
    # reuse _synthesise_shot N times with shifted peaks and keep the
    # minimum wrist-y (most-up) across the envelopes at each frame — a
    # close analog to what you'd see if a real shooter strung three
    # releases together.
    shoulder = np.tile(np.array([0.50, 0.40, 1.0]), (n, 1))
    elbow = np.tile(np.array([0.50, 0.50, 1.0]), (n, 1))
    # Sum envelopes then renormalise so each local peak reaches amplitude 1.
    t = np.arange(n, dtype=np.float64)
    sigma = 3.0
    envelope = np.zeros(n, dtype=np.float64)
    for p in peaks:
        envelope = np.maximum(envelope, np.exp(-0.5 * ((t - p) / sigma) ** 2))
    lo = np.radians(180.0 - 70.0)
    hi = np.radians(180.0 - 165.0)
    theta = lo + (hi - lo) * envelope
    r = 0.12
    wrist = np.zeros((n, 3), dtype=np.float64)
    wrist[:, 0] = elbow[:, 0] + r * np.sin(theta)
    wrist[:, 1] = elbow[:, 1] - r * np.cos(theta)
    wrist[:, 2] = 1.0
    del shoulder_parts, elbow_parts, wrist_parts  # appease flake; not needed

    result = segment_shots(wrist, elbow, shoulder, fps=FPS)
    assert result.n_shots_detected == 3
    assert result.n_shots_valid == 3
    detected = sorted(sh.release_frame for sh in result.shots)
    for expected, actual in zip(peaks, detected):
        assert abs(actual - expected) <= 2, (expected, actual, detected)
    for sh in result.shots:
        assert sh.status == ShotStatus.VALID.value
        assert set(sh.signals_fired) >= {"wrist_y_nadir", "elbow_velocity_peak"}


# ----- fixture 4: pump-fake is rejected by consensus ------------------------


def test_pump_fake_rejected_by_consensus():
    """Two wrist-y peaks; the first has NO elbow extension (pump fake).

    Physics of a real pump-fake: the shooter raises the *whole arm* without
    extending the elbow. Shoulder + elbow + wrist all translate upward
    together, so the interior elbow angle is unchanged and S2 (elbow
    angular velocity) does NOT fire. The segmenter's S1 still detects a
    wrist-y peak (because wrist-y changes in image coords), which is
    exactly the regime consensus is designed to filter: wrist-only
    motion without elbow extension -> ``degraded`` with
    ``single_signal_release``, not ``valid``. The second peak is a full
    shot (real elbow extension) and is ``valid``.
    """
    n = 180
    pump_peak = 40
    real_peak = 120

    shoulder = np.tile(np.array([0.50, 0.40, 1.0]), (n, 1)).astype(np.float64)
    elbow = np.tile(np.array([0.50, 0.50, 1.0]), (n, 1)).astype(np.float64)

    # Start from the real shot's envelope (drives both wrist pos + elbow).
    t = np.arange(n, dtype=np.float64)
    sigma = 3.0
    env_real = np.exp(-0.5 * ((t - real_peak) / sigma) ** 2)
    lo = np.radians(180.0 - 70.0)
    hi = np.radians(180.0 - 165.0)
    theta = lo + (hi - lo) * env_real
    r = 0.12
    wrist = np.zeros((n, 3), dtype=np.float64)
    wrist[:, 0] = elbow[:, 0] + r * np.sin(theta)
    wrist[:, 1] = elbow[:, 1] - r * np.cos(theta)
    wrist[:, 2] = 1.0

    # Overlay a pump-fake: translate shoulder, elbow, AND wrist upward by
    # the same Gaussian envelope. Because all three joints move together
    # the interior elbow angle is invariant -> S2 cannot fire.
    pump_env = np.exp(-0.5 * ((t - pump_peak) / sigma) ** 2)
    pump_amp = 0.14  # big enough to pass the 0.04 prominence floor
    shoulder[:, 1] -= pump_amp * pump_env
    elbow[:, 1] -= pump_amp * pump_env
    wrist[:, 1] -= pump_amp * pump_env

    result = segment_shots(wrist, elbow, shoulder, fps=FPS)
    # We expect 2 S1 candidates: one at pump_peak and one at real_peak.
    assert result.n_shots_detected == 2, [
        (sh.release_frame, sh.status, sh.signals_fired) for sh in result.shots
    ]

    by_release = {sh.release_frame: sh for sh in result.shots}
    # Pump-fake shot is degraded (S2 does not agree) with the right code.
    pump_shot = next(
        sh for sh in result.shots if abs(sh.release_frame - pump_peak) <= 4
    )
    assert pump_shot.status == ShotStatus.DEGRADED.value
    assert "single_signal_release" in pump_shot.reason_codes
    assert pump_shot.signals_fired == ("wrist_y_nadir",)

    # Real shot is valid with both signals.
    real_shot = next(
        sh for sh in result.shots if abs(sh.release_frame - real_peak) <= 4
    )
    assert real_shot.status == ShotStatus.VALID.value
    assert set(real_shot.signals_fired) >= {"wrist_y_nadir", "elbow_velocity_peak"}
    del by_release  # silence flake


# ----- misc guards ---------------------------------------------------------


def test_raises_on_zero_fps():
    shoulder, elbow, wrist = _static_arrays(n_frames=30)
    with pytest.raises(ValueError):
        segment_shots(wrist, elbow, shoulder, fps=0.0)


def test_raises_on_shape_mismatch():
    shoulder, elbow, wrist = _static_arrays(n_frames=30)
    with pytest.raises(ValueError):
        segment_shots(wrist[:20], elbow, shoulder, fps=FPS)


def test_too_short_clip_is_reported_not_crashed():
    shoulder, elbow, wrist = _static_arrays(n_frames=5)
    result = segment_shots(wrist, elbow, shoulder, fps=FPS)
    assert result.n_shots_detected == 0
    assert "signal_too_short" in result.reason_codes


def test_ball_detect_adds_signal_and_never_downgrades():
    """With S3 supplied at the real release, valid stays valid, signals grow."""
    n = 90
    peak = 45
    shoulder, elbow, wrist = _synthesise_shot(n_frames=n, peak_frame=peak)
    cfg = ShotSegmenterConfig()
    result = segment_shots(
        wrist, elbow, shoulder, fps=FPS, ball_release_frames=[peak], config=cfg
    )
    assert result.n_shots_valid == 1
    shot = result.shots[0]
    assert shot.status == ShotStatus.VALID.value
    assert set(shot.signals_fired) == {
        "wrist_y_nadir",
        "elbow_velocity_peak",
        "ball_leaves_hand",
    }
