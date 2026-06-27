from __future__ import annotations

from app.pose.rtmpose_baseline import run_rtmpose_pose_baseline
from app.pose.types import PoseBaselineResult


class RTMPosePoseBackend:
    """RTMPose + YOLOX via ``rtmlib`` (optional dependency). See ``requirements-pose-optional.txt``."""

    name = "rtmpose"

    def run(
        self,
        video_path: str,
        *,
        start_sec: float | None = None,
        end_sec: float | None = None,
        multipass: bool = False,
        person_isolation: str | None = None,
    ) -> PoseBaselineResult:
        return run_rtmpose_pose_baseline(
            video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            multipass=multipass,
            person_isolation=person_isolation,
        )
