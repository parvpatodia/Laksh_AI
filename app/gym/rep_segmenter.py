"""Deterministic rep segmenter over a 1D rep-signal (GOALS.md Milestone 1).

Scope of this module (intentional narrow cut)
---------------------------------------------
Input  : a 1D ``np.ndarray`` rep-signal, an :class:`~app.gym.exercises_v0.ExerciseV0`
         metadata row, and the sampling FPS. Optionally a boolean ``missingness``
         mask the same length as the signal.
Output : a :class:`SegmentResult` containing zero or more :class:`RepSpan`s,
         each tagged ``valid`` / ``degraded`` / ``unknown`` with explicit
         ``reason_codes`` — mirroring today's ``metric_status`` pattern
         (GOALS.md Milestone 1 bullet 3).

What this module deliberately does NOT do
-----------------------------------------
* Extract the 1D signal from a pose sequence. That mapping
  (``rep_signal_joint`` + ``rep_signal_type`` -> 1D series) belongs in a
  thin adapter that depends on the canonical frame type. Keeping the
  segmenter signal-agnostic means the whole algorithm is trivially unit
  testable on synthetic data before any real pose manifest is labeled.
* Embed per-exercise magic numbers. Thresholds live in
  :class:`SegmenterConfig` and ride with the result so a coaching response
  can attribute "why this rep was flagged" back to the actual knobs.

Algorithmic choices (v0, deterministic, no ML)
----------------------------------------------
1. Smooth with a centered moving average of ``smoothing_window`` frames.
2. Compute the signal's valid-range amplitude. If below
   ``min_signal_range`` emit ``unknown`` with ``reason_codes=["flat_signal"]``
   instead of inventing reps from noise.
3. Detect "work extrema" via :func:`scipy.signal.find_peaks`:
     * ``cyclic_vertical`` -> maxima of ``+signal`` (bottom of squat is the
       highest image-y in normalized image space),
     * ``cyclic_angle``    -> maxima of ``-signal`` (deepest flexion is the
       minimum joint angle).
   Peak separation is floored at ``min_rep_s * fps``; prominence floored at
   ``prominence_frac * signal_range`` to reject oscillations from tracker
   jitter rather than real reps.
4. For ``k`` detected extrema, emit ``k`` reps. Each rep's boundaries are
   the midpoint between neighbouring extrema (or ``0`` / ``n-1`` at the
   ends). Boundary-truncated reps are marked ``degraded`` with
   ``reason_codes=["boundary_truncated"]`` when their span is noticeably
   shorter than the median interior rep.
5. ``duration`` exercises (plank) return one span covering the whole clip
   tagged ``duration_hold``; ``gait_cadence`` (farmer carry) reuses the
   peak detector on a gait-rate signal but tags spans ``gait_step``.
6. Per-span missingness fraction above ``max_missingness_per_span`` flips
   that rep to ``degraded`` / ``high_missingness``.

Stability contract
------------------
The output schema is pinned by :data:`REP_SEGMENTER_SCHEMA_VERSION`. Any
change to field names or semantics requires a schema bump.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum

import numpy as np
from scipy.signal import find_peaks

from app.gym.exercises_v0 import ExerciseV0

REP_SEGMENTER_SCHEMA_VERSION = "1.0.0"


class RepStatus(StrEnum):
    VALID = "valid"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SegmenterConfig:
    """Thresholds shipped alongside each :class:`SegmentResult`."""

    smoothing_window: int = 5  # must be odd + >=3; clipped at signal length
    min_rep_s: float = 0.4
    max_rep_s: float = 8.0
    min_signal_range: float = 0.02
    prominence_frac: float = 0.15
    max_missingness_per_span: float = 0.25
    # A rep is boundary_truncated when its length < this fraction of the
    # median interior-rep length. 0.6 is permissive enough that a slow first
    # rep doesn't get flagged, strict enough that a quarter-rep does.
    boundary_truncation_frac: float = 0.6


@dataclass(frozen=True)
class RepSpan:
    """One rep within a :class:`SegmentResult`. Frame indices are inclusive."""

    start_frame: int
    end_frame: int
    peak_frame: int
    status: str  # RepStatus value
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class SegmentResult:
    """Immutable segmenter output. Fields match the stability contract."""

    schema_version: str
    exercise_id: str
    rep_signal_type: str
    rep_signal_joint: str | None
    n_frames: int
    fps: float
    status: str  # RepStatus value (aggregate)
    reason_codes: tuple[str, ...]
    reps: tuple[RepSpan, ...]
    config: SegmenterConfig = field(default_factory=SegmenterConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reps"] = [asdict(r) for r in self.reps]
        for r in d["reps"]:
            r["reason_codes"] = list(r["reason_codes"])
        d["reason_codes"] = list(self.reason_codes)
        return d


def _sanitise_signal(
    signal: np.ndarray, missingness: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (float signal with NaN for missing samples, boolean missingness mask).

    The returned missingness mask is True where the sample is unusable
    (caller-supplied mask ∪ NaN ∪ inf).
    """
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {arr.shape}")
    finite = np.isfinite(arr)
    if missingness is None:
        miss = ~finite
    else:
        m = np.asarray(missingness, dtype=bool)
        if m.shape != arr.shape:
            raise ValueError(
                f"missingness shape {m.shape} != signal shape {arr.shape}"
            )
        miss = m | ~finite
    out = arr.copy()
    out[miss] = np.nan
    return out, miss


def _smooth(signal: np.ndarray, window: int) -> np.ndarray:
    """Centered moving-average smoother that tolerates NaN by filling with the
    nan-aware mean on the fly. Output shape == input shape.
    """
    w = max(3, int(window) | 1)  # force odd
    n = signal.shape[0]
    if n == 0:
        return signal.copy()
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 3:
        # Signal too short to smooth; return NaN-filled copy
        filled = signal.copy()
        if np.any(~np.isfinite(filled)):
            mean = np.nanmean(filled) if np.any(np.isfinite(filled)) else 0.0
            filled[~np.isfinite(filled)] = mean
        return filled
    half = w // 2
    # Interpolate NaNs linearly so the moving average is stable; pad edges
    # with the nearest finite sample.
    finite_idx = np.flatnonzero(np.isfinite(signal))
    if finite_idx.size == 0:
        return np.zeros_like(signal)
    filled = signal.copy()
    nan_idx = np.flatnonzero(~np.isfinite(filled))
    if nan_idx.size:
        filled[nan_idx] = np.interp(nan_idx, finite_idx, signal[finite_idx])
    # Reflect-pad then convolve with uniform kernel.
    padded = np.pad(filled, half, mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(padded, kernel, mode="valid")


def _signal_range(smoothed: np.ndarray, miss: np.ndarray) -> float:
    usable = smoothed[~miss] if miss.any() else smoothed
    if usable.size == 0:
        return 0.0
    return float(np.nanmax(usable) - np.nanmin(usable))


def _detect_work_extrema(
    smoothed: np.ndarray,
    rep_signal_type: str,
    fps: float,
    cfg: SegmenterConfig,
    signal_range: float,
) -> np.ndarray:
    """Return indices of the "work" extrema for this rep_signal_type.

    The sign flip for ``cyclic_angle`` (where deep flexion = small angle)
    keeps a single peak-detector path handling both signal shapes.
    """
    if signal_range <= 0.0:
        return np.empty(0, dtype=np.int64)
    distance = max(1, int(round(cfg.min_rep_s * fps)))
    prominence = max(1e-9, cfg.prominence_frac * signal_range)
    if rep_signal_type == "cyclic_vertical":
        probe = smoothed
    elif rep_signal_type == "cyclic_angle":
        probe = -smoothed
    elif rep_signal_type == "gait_cadence":
        # Same peak math; caller will tag the spans as gait steps instead.
        probe = smoothed
    else:
        return np.empty(0, dtype=np.int64)
    peaks, _ = find_peaks(probe, distance=distance, prominence=prominence)
    return peaks.astype(np.int64)


def _rep_bounds_from_extrema(
    extrema: np.ndarray, n_frames: int
) -> list[tuple[int, int]]:
    """Midpoint partitioning: each rep occupies frames [left_mid, right_mid]."""
    if extrema.size == 0:
        return []
    bounds: list[tuple[int, int]] = []
    for i, p in enumerate(extrema):
        left = 0 if i == 0 else int((extrema[i - 1] + p) // 2)
        right = (
            n_frames - 1
            if i == extrema.size - 1
            else int((p + extrema[i + 1]) // 2)
        )
        bounds.append((left, right))
    return bounds


def _missingness_fraction(miss: np.ndarray, start: int, end: int) -> float:
    if end < start:
        return 0.0
    window = miss[start : end + 1]
    if window.size == 0:
        return 0.0
    return float(window.sum()) / float(window.size)


def _classify_rep(
    length_s: float,
    miss_frac: float,
    is_boundary_truncated: bool,
    cfg: SegmenterConfig,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if length_s < cfg.min_rep_s:
        reasons.append("short_cycle")
    if length_s > cfg.max_rep_s:
        reasons.append("long_cycle")
    if miss_frac > cfg.max_missingness_per_span:
        reasons.append("high_missingness")
    if is_boundary_truncated:
        reasons.append("boundary_truncated")
    status = RepStatus.VALID.value if not reasons else RepStatus.DEGRADED.value
    return status, tuple(reasons)


def _aggregate_status(reps: tuple[RepSpan, ...]) -> tuple[str, tuple[str, ...]]:
    """Overall status = worst of the per-rep statuses (unknown > degraded > valid)
    while still aggregating the union of per-rep reason codes for summary consumers.
    """
    if not reps:
        return RepStatus.UNKNOWN.value, ("no_reps_detected",)
    has_unknown = any(r.status == RepStatus.UNKNOWN.value for r in reps)
    has_degraded = any(r.status == RepStatus.DEGRADED.value for r in reps)
    all_reasons = tuple(
        sorted({code for r in reps for code in r.reason_codes})
    )
    if has_unknown:
        return RepStatus.UNKNOWN.value, all_reasons
    if has_degraded:
        return RepStatus.DEGRADED.value, all_reasons
    return RepStatus.VALID.value, all_reasons


def _segment_duration_hold(
    exercise: ExerciseV0,
    n: int,
    fps: float,
    miss: np.ndarray,
    cfg: SegmenterConfig,
) -> SegmentResult:
    """One span covering the whole clip; status reflects missingness only."""
    miss_frac = float(miss.sum()) / float(max(1, miss.size))
    reasons: list[str] = ["duration_hold"]
    if miss_frac > cfg.max_missingness_per_span:
        reasons.append("high_missingness")
        status = RepStatus.DEGRADED.value
    else:
        status = RepStatus.VALID.value
    span = RepSpan(
        start_frame=0,
        end_frame=max(0, n - 1),
        peak_frame=max(0, n // 2),
        status=status,
        reason_codes=tuple(reasons),
    )
    return SegmentResult(
        schema_version=REP_SEGMENTER_SCHEMA_VERSION,
        exercise_id=exercise.exercise_id,
        rep_signal_type=exercise.rep_signal_type,
        rep_signal_joint=exercise.rep_signal_joint,
        n_frames=n,
        fps=fps,
        status=status,
        reason_codes=tuple(reasons),
        reps=(span,),
        config=cfg,
    )


def segment_reps(
    signal: np.ndarray,
    exercise: ExerciseV0,
    fps: float,
    missingness: np.ndarray | None = None,
    config: SegmenterConfig | None = None,
) -> SegmentResult:
    """Segment ``signal`` into reps under the semantics of ``exercise``.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        1D rep-signal. Missing samples may be ``NaN``; see ``missingness``.
    exercise : ExerciseV0
        Drives ``rep_signal_type`` dispatch (``cyclic_vertical``,
        ``cyclic_angle``, ``duration``, ``gait_cadence``).
    fps : float
        Sampling rate in frames per second. Used only to convert
        :attr:`SegmenterConfig.min_rep_s` / ``max_rep_s`` into frame counts.
    missingness : np.ndarray[bool] | None
        Optional mask same-shape as ``signal`` where ``True`` = unusable
        sample (tracker miss, low-confidence joint, preprocess drop, ...).
    config : SegmenterConfig | None
        Override thresholds. Defaults are conservative for CPU pose traces.
    """
    cfg = config or SegmenterConfig()
    arr, miss = _sanitise_signal(signal, missingness)
    n = arr.shape[0]
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    if exercise.rep_signal_type == "duration":
        return _segment_duration_hold(exercise, n, fps, miss, cfg)

    if n < max(3, int(round(cfg.min_rep_s * fps))):
        return SegmentResult(
            schema_version=REP_SEGMENTER_SCHEMA_VERSION,
            exercise_id=exercise.exercise_id,
            rep_signal_type=exercise.rep_signal_type,
            rep_signal_joint=exercise.rep_signal_joint,
            n_frames=n,
            fps=fps,
            status=RepStatus.UNKNOWN.value,
            reason_codes=("signal_too_short",),
            reps=(),
            config=cfg,
        )

    smoothed = _smooth(arr, cfg.smoothing_window)
    signal_range = _signal_range(smoothed, miss)
    if signal_range < cfg.min_signal_range:
        return SegmentResult(
            schema_version=REP_SEGMENTER_SCHEMA_VERSION,
            exercise_id=exercise.exercise_id,
            rep_signal_type=exercise.rep_signal_type,
            rep_signal_joint=exercise.rep_signal_joint,
            n_frames=n,
            fps=fps,
            status=RepStatus.UNKNOWN.value,
            reason_codes=("flat_signal",),
            reps=(),
            config=cfg,
        )

    extrema = _detect_work_extrema(
        smoothed, exercise.rep_signal_type, fps, cfg, signal_range
    )
    if extrema.size == 0:
        return SegmentResult(
            schema_version=REP_SEGMENTER_SCHEMA_VERSION,
            exercise_id=exercise.exercise_id,
            rep_signal_type=exercise.rep_signal_type,
            rep_signal_joint=exercise.rep_signal_joint,
            n_frames=n,
            fps=fps,
            status=RepStatus.UNKNOWN.value,
            reason_codes=("no_reps_detected",),
            reps=(),
            config=cfg,
        )

    bounds = _rep_bounds_from_extrema(extrema, n)
    lengths = np.array([right - left + 1 for left, right in bounds], dtype=np.float64)
    # Use interior reps (exclude first and last if we have >=3) for a
    # median baseline so boundary truncation is detected against real reps.
    if lengths.size >= 3:
        interior_median = float(np.median(lengths[1:-1]))
    else:
        interior_median = float(np.median(lengths))
    boundary_limit = cfg.boundary_truncation_frac * interior_median

    reps: list[RepSpan] = []
    tag = "gait_step" if exercise.rep_signal_type == "gait_cadence" else "rep"
    for i, (left, right) in enumerate(bounds):
        length_frames = right - left + 1
        length_s = length_frames / fps
        miss_frac = _missingness_fraction(miss, left, right)
        is_boundary = (i == 0 or i == len(bounds) - 1) and length_frames < boundary_limit
        status, reasons = _classify_rep(length_s, miss_frac, is_boundary, cfg)
        reasons = (tag, *reasons) if tag == "gait_step" else reasons
        reps.append(
            RepSpan(
                start_frame=int(left),
                end_frame=int(right),
                peak_frame=int(extrema[i]),
                status=status,
                reason_codes=reasons,
            )
        )

    reps_tuple = tuple(reps)
    agg_status, agg_reasons = _aggregate_status(reps_tuple)
    return SegmentResult(
        schema_version=REP_SEGMENTER_SCHEMA_VERSION,
        exercise_id=exercise.exercise_id,
        rep_signal_type=exercise.rep_signal_type,
        rep_signal_joint=exercise.rep_signal_joint,
        n_frames=n,
        fps=fps,
        status=agg_status,
        reason_codes=agg_reasons,
        reps=reps_tuple,
        config=cfg,
    )
