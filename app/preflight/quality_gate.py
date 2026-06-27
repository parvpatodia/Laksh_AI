"""Pre-flight pose-quality gate for the Laksh.ai analysis pipeline.

Validates that a clip's landmark stream is good enough to justify running
the full 25-40 s MediaPipe Heavy analysis.  The gate is conservative: a
false-positive rejection (good clip flagged as bad) wastes one upload
attempt; a false-negative (bad clip allowed through) wastes 40 s of CPU
and produces null biomech fields anyway.

Thresholds
----------
All thresholds are mirrored in ``evaluation/preflight_thresholds.json``
(single source of truth shared with the client-side ring buffer in
``web/app/[sport]/page.tsx``).

``visibility_core >= 0.50``
    MediaPipe documents 0.5 as the lower bound of the "confident" detection
    band for landmark visibility (MediaPipe Pose Landmarker docs, 2024).
    Below 0.5 the landmark estimate is unreliable by the vendor's own
    specification.

``in_frame_ratio >= 0.80``
    At least 80 % of frames must have all core landmarks strictly inside the
    frame (0.05 <= x,y <= 0.95).  80 % tolerates brief occlusions while
    rejecting clips where the subject is persistently cut off.

``fps_observed >= 25``
    25 fps provides ~7.5 x the Nyquist sampling rate for the 0.6 s minimum
    inter-rep/inter-shot interval used by ``scipy.signal.find_peaks``.
    Accepting clips below 25 fps risks aliasing the rep peak.
    Justification: Nyquist requires >= 2/T sampling; 25 fps / (1/0.6) = 15 x
    margin. FPS floor of 25 is deliberately below the 30 fps target to
    tolerate 17 % VFR jitter without false rejections.

Design constraints
------------------
* Pure function: no file I/O, no MediaPipe import at module level.
* Deterministic: same inputs -> same outputs.
* Monotone: failing one metric does not affect the others.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Core landmark indices (MediaPipe 33-point model, 0-indexed).
# Shoulders (11,12), elbows (13,14), wrists (15,16), hips (23,24),
# knees (25,26).  These 10 joints must be visible for rep/shot detection.
# ---------------------------------------------------------------------------
_CORE_LANDMARK_INDICES: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26)

# Fraction of frame width/height that counts as "in frame" on each edge.
_INFRAME_MARGIN: float = 0.05

# Published default thresholds (also written to evaluation/preflight_thresholds.json).
VISIBILITY_CORE_MIN: float = 0.50
IN_FRAME_RATIO_MIN: float = 0.80
FPS_FLOOR: float = 25.0


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreflightResult:
    """Immutable result of a pre-flight quality check.

    Attributes
    ----------
    passed:
        True iff ALL three checks pass their thresholds.
    visibility_core:
        Mean visibility of the 10 core landmarks across valid frames.
        ``float('nan')`` when no valid frames exist.
    in_frame_ratio:
        Fraction of frames where all core landmarks are inside the frame.
    fps_observed:
        Frame rate as passed by the caller (not recomputed here).
    reason_codes:
        One code per failing check.  Empty when ``passed`` is True.
    per_signal_actuals:
        Raw measured value for each signal so the UI can tell the user how
        far from the threshold they were.
    """

    passed: bool
    visibility_core: float
    in_frame_ratio: float
    fps_observed: float
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    per_signal_actuals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict for the API response."""
        return {
            "passed": self.passed,
            "visibility_core": self.visibility_core,
            "in_frame_ratio": self.in_frame_ratio,
            "fps_observed": self.fps_observed,
            "reason_codes": list(self.reason_codes),
            "per_signal_actuals": dict(self.per_signal_actuals),
            "thresholds": {
                "visibility_core": VISIBILITY_CORE_MIN,
                "in_frame_ratio": IN_FRAME_RATIO_MIN,
                "fps_floor": FPS_FLOOR,
            },
        }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def compute_preflight_metrics(
    landmarks_per_frame: list[list[Any] | None],
    fps_observed: float,
    *,
    visibility_min: float = VISIBILITY_CORE_MIN,
    in_frame_min: float = IN_FRAME_RATIO_MIN,
    fps_floor: float = FPS_FLOOR,
) -> PreflightResult:
    """Compute pre-flight quality metrics from a sequence of landmark lists.

    Parameters
    ----------
    landmarks_per_frame:
        Per-frame landmark objects.  Each element is either ``None``
        (no pose detected) or a list/sequence of landmark objects with
        ``.x``, ``.y``, and ``.visibility`` attributes.  Dict-style
        access (``["x"]``, etc.) is also accepted, enabling use with
        both MediaPipe C-extension objects and plain dicts (tests, JSON).
    fps_observed:
        Frame rate as measured by the caller.  Passed through unchanged.
    visibility_min:
        Override for the visibility threshold.
    in_frame_min:
        Override for the in-frame ratio threshold.
    fps_floor:
        Override for the FPS threshold.

    Returns
    -------
    PreflightResult
        Immutable result with per-check actuals and reason codes.
    """
    vis_sum = 0.0
    vis_count = 0
    in_frame_count = 0
    total_frames = 0

    for frame_landmarks in landmarks_per_frame:
        if frame_landmarks is None:
            continue

        # Collect the 10 core landmarks for this frame.
        core: list[Any] = []
        for idx in _CORE_LANDMARK_INDICES:
            try:
                lm = frame_landmarks[idx]
            except (IndexError, TypeError):
                lm = None
            if lm is not None:
                core.append(lm)

        if not core:
            continue

        total_frames += 1

        # Visibility: mean across available core joints for this frame.
        frame_vis = sum(_get_attr(lm, "visibility", 0.0) for lm in core) / len(core)
        vis_sum += frame_vis
        vis_count += 1

        # In-frame: ALL core joints must be within the margin on both axes.
        in_frame = all(
            _INFRAME_MARGIN <= _get_attr(lm, "x", 0.0) <= (1.0 - _INFRAME_MARGIN)
            and _INFRAME_MARGIN <= _get_attr(lm, "y", 0.0) <= (1.0 - _INFRAME_MARGIN)
            for lm in core
        )
        if in_frame:
            in_frame_count += 1

    vis_core = vis_sum / vis_count if vis_count > 0 else math.nan
    ifr = in_frame_count / total_frames if total_frames > 0 else 0.0

    vis_ok = not math.isnan(vis_core) and vis_core >= visibility_min
    ifr_ok = ifr >= in_frame_min
    fps_ok = fps_observed >= fps_floor

    reason_codes: list[str] = []
    if not vis_ok:
        reason_codes.append("preflight_visibility_failed")
    if not ifr_ok:
        reason_codes.append("preflight_in_frame_failed")
    if not fps_ok:
        reason_codes.append("preflight_fps_failed")

    vis_out = round(vis_core, 4) if not math.isnan(vis_core) else math.nan

    return PreflightResult(
        passed=len(reason_codes) == 0,
        visibility_core=vis_out,
        in_frame_ratio=round(ifr, 4),
        fps_observed=fps_observed,
        reason_codes=tuple(reason_codes),
        per_signal_actuals={
            "visibility_core": vis_out,
            "in_frame_ratio": round(ifr, 4),
            "fps_observed": fps_observed,
        },
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _get_attr(obj: Any, attr: str, default: float) -> float:
    """Get attribute from a landmark object or dict, returning default on miss."""
    if isinstance(obj, dict):
        return float(obj.get(attr, default))
    return float(getattr(obj, attr, default))
