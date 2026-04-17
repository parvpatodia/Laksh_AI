"""
Feature flags for basketball / gym kinematic path (ADR 0002 P3).

Default off so production metrics stay stable until manifest-backed parity review.
"""
from __future__ import annotations

import os


def use_canonical_joint_trace() -> bool:
    """
    When true, `_extract_frames_with_variant` records per-frame canonical joint maps
    (MediaPipe 33 → COCO-17) for parity telemetry; does not change default metrics.
    """
    v = (os.environ.get("LAKSH_USE_CANONICAL_JOINTS") or "").strip().lower()
    return v in ("1", "true", "yes", "on")
