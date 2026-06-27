"""
Internal canonical pose joint contract (P0 of POSE_UPGRADE_EXECUTION_PLAN.md).

Uses **COCO-17-style** joint *names* so future RTMPose / ViTPose heads can map into the
same vocabulary. Coordinates are **normalized image space** (same convention as
MediaPipe pose image landmarks: x,y in [0,1] origin top-left unless the decoder says otherwise).

Bump ``CANONICAL_JOINT_SCHEMA_VERSION`` when joint set or axis semantics change.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

CANONICAL_JOINT_SCHEMA_VERSION = "1.0.0"

# COCO 17 keypoint names (order matches common COCO skeleton indexing 0..16).
class CanonicalJointName(StrEnum):
    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


CANONICAL_JOINT_ORDER: tuple[CanonicalJointName, ...] = tuple(CanonicalJointName)


@dataclass(frozen=True)
class JointObservation:
    """One joint in normalized image coordinates + MediaPipe-style visibility in [0,1]."""

    x: float
    y: float
    z: float
    visibility: float


def canonical_joint_dict_from_mapping(
    m: Mapping[CanonicalJointName, JointObservation],
) -> dict[str, dict[str, float]]:
    """JSON-serializable nested dict keyed by joint name string."""
    return {
        j.value: {"x": o.x, "y": o.y, "z": o.z, "visibility": o.visibility}
        for j, o in m.items()
    }
