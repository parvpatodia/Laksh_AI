"""
MediaPipe BlazePose **33** image landmarks → canonical COCO-17-style joints.

Index reference: Google MediaPipe pose landmarker landmark table (33 landmarks).
Eye sub-indices: we use **center** eye landmarks (2 = left eye, 5 = right eye), not inner/outer corners.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.pose.canonical import CanonicalJointName, JointObservation

# BlazePose 33 landmark index per joint (must stay aligned with MediaPipe docs).
MEDIAPIPE_BLAZEPOSE33_TO_CANONICAL: dict[CanonicalJointName, int] = {
    CanonicalJointName.NOSE: 0,
    CanonicalJointName.LEFT_EYE: 2,
    CanonicalJointName.RIGHT_EYE: 5,
    CanonicalJointName.LEFT_EAR: 7,
    CanonicalJointName.RIGHT_EAR: 8,
    CanonicalJointName.LEFT_SHOULDER: 11,
    CanonicalJointName.RIGHT_SHOULDER: 12,
    CanonicalJointName.LEFT_ELBOW: 13,
    CanonicalJointName.RIGHT_ELBOW: 14,
    CanonicalJointName.LEFT_WRIST: 15,
    CanonicalJointName.RIGHT_WRIST: 16,
    CanonicalJointName.LEFT_HIP: 23,
    CanonicalJointName.RIGHT_HIP: 24,
    CanonicalJointName.LEFT_KNEE: 25,
    CanonicalJointName.RIGHT_KNEE: 26,
    CanonicalJointName.LEFT_ANKLE: 27,
    CanonicalJointName.RIGHT_ANKLE: 28,
}


def _obs_from_landmark(lm: Any) -> JointObservation:
    vis = getattr(lm, "visibility", None)
    if vis is None:
        vis = getattr(lm, "presence", float("nan"))
    return JointObservation(
        x=float(lm.x),
        y=float(lm.y),
        z=float(getattr(lm, "z", 0.0)),
        visibility=float(vis) if vis is not None else float("nan"),
    )


def map_mediapipe_blazepose33_to_canonical(
    landmarks: Sequence[Any] | None,
) -> dict[CanonicalJointName, JointObservation] | None:
    """
    Map one frame's 33 landmarks to the internal canonical dict.

    Returns:
        ``None`` if ``landmarks`` is ``None`` or not length 33.
    """
    if landmarks is None:
        return None
    n = len(landmarks)
    if n != 33:
        return None
    out: dict[CanonicalJointName, JointObservation] = {}
    for joint, idx in MEDIAPIPE_BLAZEPOSE33_TO_CANONICAL.items():
        out[joint] = _obs_from_landmark(landmarks[idx])
    return out
