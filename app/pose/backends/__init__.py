"""Pluggable 2D pose backends for Phase A benchmarks (MediaPipe first; RTMPose later)."""
from __future__ import annotations

from app.pose.backends.mediapipe_backend import MediaPipePoseBackend

__all__ = ["MediaPipePoseBackend", "get_pose_backend"]


def get_pose_backend(name: str):
    key = (name or "").strip().lower()
    if key == "mediapipe":
        return MediaPipePoseBackend()
    raise NotImplementedError(
        f"Unknown pose backend {name!r}. Only 'mediapipe' is implemented; "
        "add RTMPose/ViTPose behind the same PoseBaselineResult contract when ready."
    )
