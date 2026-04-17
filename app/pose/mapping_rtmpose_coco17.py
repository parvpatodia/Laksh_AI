"""
RTMPose (MMPose COCO-17, ``to_openpose=False``) → canonical joint dict.

``rtmlib`` returns pixel-space x,y in the **input image**; we normalize by
``image_width`` / ``image_height`` to match MediaPipe-style normalized coordinates
used in ``gym_baseline_metrics``.
"""
from __future__ import annotations

import numpy as np

from app.pose.canonical import CANONICAL_JOINT_ORDER, CanonicalJointName, JointObservation


def coco17_pixels_to_canonical(
    keypoints_xy: np.ndarray,
    scores: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> dict[CanonicalJointName, JointObservation]:
    """
    Args:
        keypoints_xy: (17, 2) in pixel coordinates.
        scores: (17,) confidence in roughly [0, 1].
    """
    if keypoints_xy.shape != (17, 2) or scores.shape != (17,):
        raise ValueError(f"Expected (17,2) and (17,), got {keypoints_xy.shape}, {scores.shape}")
    w = max(1, int(image_width))
    h = max(1, int(image_height))
    out: dict[CanonicalJointName, JointObservation] = {}
    for i, jn in enumerate(CANONICAL_JOINT_ORDER):
        x = float(np.clip(keypoints_xy[i, 0] / w, 0.0, 1.0))
        y = float(np.clip(keypoints_xy[i, 1] / h, 0.0, 1.0))
        vis = float(np.clip(scores[i], 0.0, 1.0))
        out[jn] = JointObservation(x=x, y=y, z=0.0, visibility=vis)
    return out


def canonical_to_gym_raw_row(
    can: dict[CanonicalJointName, JointObservation] | None,
) -> dict[str, np.ndarray]:
    """One frame → 12 joint vectors [x, y, visibility] for aggregate_metrics."""
    nan = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    keys = [
        (CanonicalJointName.LEFT_WRIST, "left_wrist"),
        (CanonicalJointName.RIGHT_WRIST, "right_wrist"),
        (CanonicalJointName.LEFT_ELBOW, "left_elbow"),
        (CanonicalJointName.RIGHT_ELBOW, "right_elbow"),
        (CanonicalJointName.LEFT_SHOULDER, "left_shoulder"),
        (CanonicalJointName.RIGHT_SHOULDER, "right_shoulder"),
        (CanonicalJointName.LEFT_HIP, "left_hip"),
        (CanonicalJointName.RIGHT_HIP, "right_hip"),
        (CanonicalJointName.LEFT_KNEE, "left_knee"),
        (CanonicalJointName.RIGHT_KNEE, "right_knee"),
        (CanonicalJointName.LEFT_ANKLE, "left_ankle"),
        (CanonicalJointName.RIGHT_ANKLE, "right_ankle"),
    ]
    row: dict[str, np.ndarray] = {}
    if can is None:
        for _, sk in keys:
            row[sk] = nan
        return row
    for jn, sk in keys:
        o = can[jn]
        row[sk] = np.array([o.x, o.y, o.visibility], dtype=np.float64)
    return row
