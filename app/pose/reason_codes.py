"""
Stable machine-readable strings for pose baseline JSONL.

- ``ok=False`` rows use ``error`` (human-readable) plus ``reason_codes`` beginning with
  a *failure* code from FAILURE_CODES.
- ``ok=True`` rows use ``reason_codes`` for *diagnostics* (DETECTION_QUALITY_CODES plus
  optional ``pose_not_usable_heuristic`` appended in the runner).

**Do not rename** existing codes without bumping ``pose_baseline_schema_version`` in
``app.pose.provenance`` and documenting the migration in ``docs/POSE_EVALUATION_PROTOCOL.md``.
"""
from __future__ import annotations

# Hard failures — backend did not produce a full metric vector.
FAILURE_CODES: frozenset[str] = frozenset({"decode_error", "pose_init_failed"})

# Diagnostics and soft flags — ok may still be True.
DETECTION_QUALITY_CODES: frozenset[str] = frozenset(
    {
        "short_clip",
        "very_low_detection",
        "low_detection",
        "low_visibility_core",
        "multiple_people_detected",
        "pose_not_usable_heuristic",
    }
)

REASON_CODE_DESCRIPTIONS: dict[str, str] = {
    "decode_error": "Video could not be opened or decoded (see error string).",
    "pose_init_failed": "MediaPipe landmarker or model asset failed to initialize.",
    "short_clip": "n_frames < 3 — metrics are noisy or undefined for segmentation.",
    "very_low_detection": "detection_rate < 0.05",
    "low_detection": "0.05 <= detection_rate < 0.15",
    "low_visibility_core": "visibility_core_when_detected < 0.25 on frames with pose.",
    "multiple_people_detected": "At least one frame returned more than one pose track — first pose only is used.",
    "pose_not_usable_heuristic": "Did not pass versioned gym usable gate (calibration JSON / defaults).",
}


def merge_reason_codes(
    detection_rate: float,
    n_frames: int,
    vis: float,
    *,
    max_people_seen: int = 0,
) -> list[str]:
    """
    Diagnostic taxonomy for manifests and gates (not user-facing product copy).

    ``max_people_seen`` is the maximum concurrent poses in any frame (landmarker config
    may allow >1); values >1 imply ambiguous single-subject assumption.
    """
    codes: list[str] = []
    if n_frames < 3:
        codes.append("short_clip")
    if detection_rate < 0.05:
        codes.append("very_low_detection")
    elif detection_rate < 0.15:
        codes.append("low_detection")
    if vis < 0.25:
        codes.append("low_visibility_core")
    if max_people_seen > 1:
        codes.append("multiple_people_detected")
    return codes
