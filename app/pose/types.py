"""Structured outputs for pose-only evaluation (gym Phase A)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PoseBaselineResult:
    """
    Backend-agnostic summary for a single clip. Populated by MediaPipe today;
    same fields should be filled when adding RTMPose / other 2D backbones.
    """

    backend: str
    video_path: str
    ok: bool
    error: str | None = None
    n_frames: int = 0
    n_frames_with_pose: int = 0
    detection_rate: float = 0.0
    visibility_core_when_detected: float = 0.0
    visibility_core_all_frames: float = 0.0
    hip_mid_displacement_median_norm: float | None = None
    max_people_seen: int = 0
    selected_pass: str | None = None
    pose_usable_heuristic: bool = False
    reason_codes: list[str] = field(default_factory=list)
    fps: float | None = None
    # True only when FFmpeg H.264/30fps normalize ran; False if original file was used (see logs).
    ffmpeg_preprocess_applied: bool = False
    # Versions, model hash, landmarker options — see app.pose.provenance / docs/POSE_EVALUATION_PROTOCOL.md
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def merge_reason_codes(detection_rate: float, n_frames: int, vis: float) -> list[str]:
    """Human-readable taxonomy for manifests and gates (not user-facing product copy)."""
    codes: list[str] = []
    if n_frames < 3:
        codes.append("short_clip")
    if detection_rate < 0.05:
        codes.append("very_low_detection")
    elif detection_rate < 0.15:
        codes.append("low_detection")
    if vis < 0.25:
        codes.append("low_visibility_core")
    return codes
