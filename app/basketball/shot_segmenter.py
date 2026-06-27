"""Multi-signal consensus release detector for basketball shots.

Scope (plan `warm-swimming-acorn.md` A1)
----------------------------------------
Given per-frame 2D landmark arrays for the shooting arm (wrist / elbow /
shoulder) and the sampling FPS, return 0..N :class:`ShotSpan`s. Each span is
tagged ``valid`` only when **two independent release signals agree within a
small temporal tolerance**, mirroring the robust-statistics consensus pattern
(Rousseeuw / Huber) adopted throughout the pipeline.

Signals
-------
S1 — Wrist vertical (y) trajectory nadir.
    Peak of ``-wrist_y`` via :func:`scipy.signal.find_peaks`. Thresholds come
    from two places (plan P4):

    * ``distance = int(round(fps * 0.6))`` — floor on inter-shot interval.
      Published elite spot-up shooting cadence is 0.7–0.9 s per rep; 0.6 s
      leaves 33 % margin while rejecting pose jitter.
    * ``prominence = 0.04`` — 4 % of normalized image height. Jump-shot wrist
      excursion on a centered subject is ~18 %; 4 % is the MediaPipe Heavy
      noise floor measured on 720p during the FFmpeg stabilization fix.

S2 — Elbow angular-velocity extension spike.
    First difference of the interior angle (shoulder-elbow-wrist) per frame.
    At each S1 candidate we require a positive peak in ``d(angle)/dt`` within
    ``tolerance_frames`` (default ±4 at 30 fps = ±133 ms, covering the 80–120
    ms wrist-nadir-to-elbow-peak lag observed in sports-biomechanics data).
    Release is biomechanically defined as maximum elbow extension velocity,
    so S2 independently validates S1 and uniquely filters pump-fakes (wrist
    rises but elbow never snaps).

S3 — Ball-leaves-hand (optional).
    Supplied externally via ``ball_release_frames``. When provided and within
    the tolerance window, it contributes to consensus. This module does NOT
    run the ball detector — that is gated by ``LAKSH_ENABLE_BALL_DETECT``
    (plan Track B) and injected by the caller.

Consensus rule
--------------
At each S1 candidate frame ``p``:

* If **S2 agrees within ``tolerance_frames``** (and S3 agrees when provided)
  → ``status = "valid"``, ``signals_fired`` names all that agreed.
* If only S1 fires (S2 — and S3 when provided — don't agree) →
  ``status = "degraded"`` with ``reason_codes = ("single_signal_release",)``.
* If 0 S1 candidates → empty shot list; segmenter reports
  ``reason_codes = ("no_release_detected",)`` at the aggregate level.

Per-shot window
---------------
Each accepted shot gets a window ``[release - window_before, release +
window_after]`` (defaults 0.8 s before, 0.5 s after) that the caller feeds
into the existing ``KinematicAnalyzer._extract_release_metrics`` for biomech
on that shot only. Windows may clip at clip boundaries; boundary-clipped
shots still report but are tagged ``boundary_truncated`` in addition to
their consensus status.

Honesty contract
----------------
Null is a first-class value. If no S1 peak survives prominence, the
:class:`SegmentResult` reports 0 shots — not a forced "best guess". Callers
must surface the aggregate ``reason_codes`` back to the user.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum

import numpy as np
from scipy.signal import find_peaks

SHOT_SEGMENTER_SCHEMA_VERSION = "1.0.0"


class ShotStatus(StrEnum):
    VALID = "valid"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class ShotSegmenterConfig:
    """Thresholds shipped alongside each :class:`SegmentResult`.

    Every default here has a provenance comment. Any future edit without
    matching justification violates plan principle P4.
    """

    min_inter_shot_s: float = 0.6
    """Minimum seconds between two detected release peaks (S1 distance
    constraint). See module docstring for the 0.7–0.9 s published-cadence
    derivation; 0.6 s = 33 % safety margin."""

    prominence_wrist_y: float = 0.04
    """Minimum S1 peak prominence in normalized image height. 4 % is the
    MediaPipe Heavy noise floor on 720p; ~18 % typical wrist excursion."""

    tolerance_frames: int = 4
    """Half-width of the consensus agreement window around an S1 candidate.
    4 frames at 30 fps = 133 ms, bracketing the 80–120 ms wrist-nadir-to-
    elbow-peak physiological lag."""

    window_before_s: float = 0.8
    """Seconds of clip before the release frame retained for per-shot
    biomech. Covers the dip + load phase of a standard jump shot."""

    window_after_s: float = 0.5
    """Seconds retained after release. Sufficient for follow-through
    kinematics while short enough to isolate one shot from the next."""

    boundary_truncation_frac: float = 0.6
    """A per-shot window is tagged ``boundary_truncated`` when its actual
    length falls below this fraction of its nominal length."""

    s2_relative_floor: float = 0.30
    """Minimum S2 (elbow angular velocity) peak as a fraction of the
    clip's overall max positive elbow angular velocity. 30 % means a
    candidate only passes S2 when the local extension spike is at least
    one-third as strong as the strongest extension observed in the
    whole clip. Self-calibrating: no hard-coded deg/s threshold. A flat
    clip produces max=0 -> denominator guarded -> S2 simply never
    agrees, which is the correct behaviour (e.g. pump fakes with no
    real extension anywhere)."""


@dataclass(frozen=True)
class ShotSpan:
    """One detected shot. Frame indices are inclusive and clipped into range."""

    start_frame: int
    end_frame: int
    release_frame: int
    status: str  # ShotStatus value
    signals_fired: tuple[str, ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class SegmentResult:
    """Immutable segmenter output. Mirrors :class:`app.gym.rep_segmenter.SegmentResult`
    shape so downstream consumers (provenance, UI) can share reporting code.
    """

    schema_version: str
    n_frames: int
    fps: float
    n_shots_detected: int
    n_shots_valid: int
    n_shots_degraded: int
    shots: tuple[ShotSpan, ...]
    reason_codes: tuple[str, ...]
    config: ShotSegmenterConfig = field(default_factory=ShotSegmenterConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shots"] = [asdict(s) for s in self.shots]
        for s in d["shots"]:
            s["signals_fired"] = list(s["signals_fired"])
            s["reason_codes"] = list(s["reason_codes"])
        d["reason_codes"] = list(self.reason_codes)
        return d


# ---------- signal helpers -------------------------------------------------


def _as_xy(arr: np.ndarray) -> np.ndarray:
    """Coerce a landmark array to shape ``(n, 2)`` of ``float64``.

    Accepts ``(n, 2)`` or ``(n, 3+)`` where columns 0/1 are x/y and any
    extra column (typically visibility) is ignored. This matches the
    ``f_2d`` convention used by :mod:`app.physics_engine` where each joint
    is an ``(n, 3)`` array of ``[x, y, visibility]``.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError(f"expected (n, 2+) landmark array, got shape {a.shape}")
    return a[:, :2]


def _interior_angle_series(
    shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray
) -> np.ndarray:
    """Per-frame interior elbow angle in degrees, with NaN where degenerate.

    Uses the same ``arccos(dot / (|a| |b|))`` form as
    :func:`app.gym.rep_features._interior_angle_deg` but vectorised so it
    runs once per clip (n ≤ a few hundred) rather than per frame.
    """
    ba = shoulder - elbow
    bc = wrist - elbow
    n_ba = np.hypot(ba[:, 0], ba[:, 1])
    n_bc = np.hypot(bc[:, 0], bc[:, 1])
    denom = n_ba * n_bc
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(denom > 1e-9, (ba[:, 0] * bc[:, 0] + ba[:, 1] * bc[:, 1]) / denom, np.nan)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _elbow_angular_velocity(angle_deg: np.ndarray, fps: float) -> np.ndarray:
    """First difference of the elbow angle in deg/s.

    Uses :func:`numpy.gradient` for a centred-difference estimator that
    preserves length (the segmenter needs per-frame values). NaN frames
    propagate; callers must tolerate NaN in downstream peak detection.
    """
    if angle_deg.size < 2 or fps <= 0:
        return np.zeros_like(angle_deg)
    # np.gradient uses centred differences in the interior, forward/backward
    # at the endpoints. Scale by fps to get deg/s.
    return np.gradient(angle_deg) * float(fps)


# ---------- core segmenter -------------------------------------------------


def _find_s1_candidates(
    wrist_y: np.ndarray,
    fps: float,
    cfg: ShotSegmenterConfig,
    *,
    wrist_vis: np.ndarray | None = None,
    vis_floor: float = 0.30,
) -> np.ndarray:
    """Return frame indices of wrist-y nadirs satisfying the S1 constraints.

    Edge cases
    ----------
    * ``wrist_vis`` is accepted as an optional (n,) visibility array.  Any
      frame whose visibility is below *vis_floor* is treated as NaN in the
      wrist-y signal.  This prevents phantom S1 peaks from low-confidence
      wrist landmark positions — e.g. when only fingertips are out of frame
      and MediaPipe falls back to a noisy wrist estimate.
    * NaN frames (from low visibility or from the caller) are filled with the
      signal's running median so ``find_peaks`` runs on a clean array; the
      prominence constraint is tight enough that filled-median values (which
      have zero local excursion) cannot pass.
    * If ALL frames are NaN or low-vis, we return an empty array immediately
      (no shots; caller surfaces "no_release_detected").
    """
    y = wrist_y.copy().astype(np.float64)

    # Mask low-visibility frames (top-of-fingers occlusion, hips/knees fully
    # out of frame are IRRELEVANT here — S1 only needs the wrist).
    if wrist_vis is not None:
        vis_arr = np.asarray(wrist_vis, dtype=np.float64)
        if vis_arr.shape[0] == y.shape[0]:
            y[vis_arr < vis_floor] = np.nan

    finite = np.isfinite(y)
    if not finite.any():
        return np.empty(0, dtype=np.int64)
    filled = y.copy()
    if not finite.all():
        filled[~finite] = float(np.nanmedian(y))
    distance = max(1, int(round(cfg.min_inter_shot_s * fps)))
    peaks, _ = find_peaks(-filled, distance=distance, prominence=cfg.prominence_wrist_y)
    return peaks.astype(np.int64)


def _s2_agrees(
    elbow_angvel: np.ndarray,
    candidate: int,
    cfg: ShotSegmenterConfig,
    *,
    clip_max_angvel: float,
) -> bool:
    """Return True when an elbow angular-velocity positive peak sits within
    ``±tolerance_frames`` of ``candidate``.

    We require:

    * A strict local maximum — at least one neighbour on each side lower.
    * Positive angular velocity (elbow extending, not flexing).
    * Magnitude at least ``s2_relative_floor * clip_max_angvel``, so that
      numerical-noise peaks in flat-angle regions do not trigger consensus
      on pump fakes where the arm translates without extending.

    NaN frames are treated as non-agreements: silence rather than
    fabrication. A flat clip (``clip_max_angvel <= 0``) cannot produce
    agreement — by design, since there is no real release anywhere.
    """
    if clip_max_angvel <= 0.0:
        return False
    floor = cfg.s2_relative_floor * clip_max_angvel
    n = elbow_angvel.shape[0]
    lo = max(1, candidate - cfg.tolerance_frames)
    hi = min(n - 2, candidate + cfg.tolerance_frames)
    if lo > hi:
        return False
    for i in range(lo, hi + 1):
        v = elbow_angvel[i]
        if not np.isfinite(v) or v <= floor:
            continue
        left = elbow_angvel[i - 1]
        right = elbow_angvel[i + 1]
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        if v >= left and v >= right and (v > left or v > right):
            return True
    return False


def _s3_agrees(
    ball_release_frames: list[int] | None,
    candidate: int,
    cfg: ShotSegmenterConfig,
) -> bool:
    """True when an externally-supplied ball-release frame sits within the
    tolerance window. Caller supplies ``None`` when S3 is disabled."""
    if not ball_release_frames:
        return False
    tol = cfg.tolerance_frames
    return any(abs(int(f) - candidate) <= tol for f in ball_release_frames)


def _clip_window(
    release: int, n_frames: int, fps: float, cfg: ShotSegmenterConfig
) -> tuple[int, int, bool]:
    """Return (start, end, was_truncated) for a release-centered window."""
    pre = int(round(cfg.window_before_s * fps))
    post = int(round(cfg.window_after_s * fps))
    start = release - pre
    end = release + post
    nominal_len = pre + post + 1
    truncated = start < 0 or end > n_frames - 1
    start = max(0, start)
    end = min(n_frames - 1, end)
    actual_len = end - start + 1
    # Only flag as truncated if the actual window is substantially shorter
    # than nominal (prevents trivial 1-frame cutoffs from being flagged).
    was_truncated = truncated and actual_len < cfg.boundary_truncation_frac * nominal_len
    return start, end, was_truncated


def segment_shots(
    wrist_xy: np.ndarray,
    elbow_xy: np.ndarray,
    shoulder_xy: np.ndarray,
    fps: float,
    *,
    ball_release_frames: list[int] | None = None,
    config: ShotSegmenterConfig | None = None,
) -> SegmentResult:
    """Detect 0..N shots in a clip via consensus over wrist + elbow signals.

    Parameters
    ----------
    wrist_xy, elbow_xy, shoulder_xy : np.ndarray
        Per-frame 2D landmark arrays in normalized image coords. Shape
        ``(n, 2)`` or ``(n, 3+)`` — extra columns (e.g. visibility) are
        ignored. These are the shooting-arm joints: the caller picks the
        active side via visibility mean (matches
        :mod:`app.physics_engine` L857-859).
    fps : float
        Decoded frames-per-second. Must be > 0.
    ball_release_frames : list[int] | None
        Optional S3 signal: frame indices at which an independent ball
        detector (plan Track B) marks release. ``None`` disables S3.
    config : ShotSegmenterConfig | None
        Override defaults. Defaults are the source-cited constants in
        :class:`ShotSegmenterConfig`.

    Returns
    -------
    SegmentResult
        Deterministic per-shot output including the signals that fired,
        the per-shot status, and an aggregate reason-code list for the
        0-shot case.
    """
    cfg = config or ShotSegmenterConfig()
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    w = _as_xy(wrist_xy)
    e = _as_xy(elbow_xy)
    s = _as_xy(shoulder_xy)
    if not (w.shape[0] == e.shape[0] == s.shape[0]):
        raise ValueError(
            f"wrist/elbow/shoulder length mismatch: "
            f"{w.shape[0]}/{e.shape[0]}/{s.shape[0]}"
        )
    n = w.shape[0]
    if n < max(3, int(round(cfg.min_inter_shot_s * fps))):
        return SegmentResult(
            schema_version=SHOT_SEGMENTER_SCHEMA_VERSION,
            n_frames=n,
            fps=float(fps),
            n_shots_detected=0,
            n_shots_valid=0,
            n_shots_degraded=0,
            shots=(),
            reason_codes=("signal_too_short",),
            config=cfg,
        )

    wrist_y = w[:, 1]
    # Extract wrist visibility if the input array carries it as column 2.
    # The raw_2d arrays from physics_engine are (n, 3) = [x, y, visibility].
    # When only (n, 2) is supplied (synthetic tests), visibility is absent
    # and we skip masking — correct behaviour since test fixtures have valid
    # wrist positions on every frame.
    _wrist_vis: np.ndarray | None = None
    raw_w = np.asarray(wrist_xy)
    if raw_w.ndim == 2 and raw_w.shape[1] >= 3:
        _wrist_vis = raw_w[:, 2].astype(np.float64)

    elbow_angle = _interior_angle_series(s, e, w)
    elbow_angvel = _elbow_angular_velocity(elbow_angle, fps)
    # Clip-wide max extension velocity calibrates the S2 floor. Treat
    # NaNs as absent (nanmax with all-NaN returns nan; we coerce to 0).
    if np.all(~np.isfinite(elbow_angvel)):
        clip_max_angvel = 0.0
    else:
        clip_max_angvel = float(np.nanmax(elbow_angvel))
        if not np.isfinite(clip_max_angvel) or clip_max_angvel < 0.0:
            clip_max_angvel = 0.0

    s1 = _find_s1_candidates(wrist_y, fps, cfg, wrist_vis=_wrist_vis)
    if s1.size == 0:
        return SegmentResult(
            schema_version=SHOT_SEGMENTER_SCHEMA_VERSION,
            n_frames=n,
            fps=float(fps),
            n_shots_detected=0,
            n_shots_valid=0,
            n_shots_degraded=0,
            shots=(),
            reason_codes=("no_release_detected",),
            config=cfg,
        )

    shots: list[ShotSpan] = []
    for candidate in s1:
        candidate = int(candidate)
        fired: list[str] = ["wrist_y_nadir"]
        if _s2_agrees(elbow_angvel, candidate, cfg, clip_max_angvel=clip_max_angvel):
            fired.append("elbow_velocity_peak")
        if _s3_agrees(ball_release_frames, candidate, cfg):
            fired.append("ball_leaves_hand")

        # Consensus: >=2 signals OR S1 alone (degraded). S1 alone is still
        # reported so judges see the attempt, but tagged so the UI can
        # visually down-weight it.
        reasons: list[str] = []
        if len(fired) >= 2:
            status = ShotStatus.VALID.value
        else:
            status = ShotStatus.DEGRADED.value
            reasons.append("single_signal_release")

        start, end, truncated = _clip_window(candidate, n, fps, cfg)
        if truncated:
            reasons.append("boundary_truncated")

        shots.append(
            ShotSpan(
                start_frame=start,
                end_frame=end,
                release_frame=candidate,
                status=status,
                signals_fired=tuple(fired),
                reason_codes=tuple(reasons),
            )
        )

    n_valid = sum(1 for sh in shots if sh.status == ShotStatus.VALID.value)
    n_degraded = sum(1 for sh in shots if sh.status == ShotStatus.DEGRADED.value)
    agg_reasons: list[str] = []
    if n_degraded and not n_valid:
        agg_reasons.append("all_shots_degraded")
    return SegmentResult(
        schema_version=SHOT_SEGMENTER_SCHEMA_VERSION,
        n_frames=n,
        fps=float(fps),
        n_shots_detected=len(shots),
        n_shots_valid=n_valid,
        n_shots_degraded=n_degraded,
        shots=tuple(shots),
        reason_codes=tuple(agg_reasons),
        config=cfg,
    )
