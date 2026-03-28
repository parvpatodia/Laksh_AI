from __future__ import annotations

from app.pose.mediapipe_baseline import run_mediapipe_pose_baseline
from app.pose.types import PoseBaselineResult


class MediaPipePoseBackend:
    """MediaPipe Pose Landmarker (heavy) — VIDEO mode, same options as KinematicAnalyzer."""

    name = "mediapipe"

    def run(
        self,
        video_path: str,
        *,
        start_sec: float | None = None,
        end_sec: float | None = None,
        multipass: bool = False,
    ) -> PoseBaselineResult:
        return run_mediapipe_pose_baseline(
            video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            multipass=multipass,
        )
