"""Pluggable 2D pose backends for Phase A benchmarks (MediaPipe + optional RTMPose)."""
from __future__ import annotations

from app.pose.backends.mediapipe_backend import MediaPipePoseBackend
from app.pose.backends.rtmpose_backend import RTMPosePoseBackend

__all__ = ["MediaPipePoseBackend", "RTMPosePoseBackend", "get_pose_backend"]


def get_pose_backend(name: str):
    key = (name or "").strip().lower()
    if key == "mediapipe":
        return MediaPipePoseBackend()
    if key == "rtmpose":
        return RTMPosePoseBackend()
    raise NotImplementedError(
        f"Unknown pose backend {name!r}. Supported: 'mediapipe', 'rtmpose' "
        "(rtmpose requires: pip install -r requirements-pose-optional.txt)."
    )
