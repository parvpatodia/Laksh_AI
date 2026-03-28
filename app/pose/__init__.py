"""
Pose package: types and backends for Phase A benchmarks.

Heavy deps (OpenCV, MediaPipe) load only when importing
``app.pose.mediapipe_baseline`` or ``app.pose.backends``.
"""

from app.pose.types import PoseBaselineResult

__all__ = ["PoseBaselineResult"]
