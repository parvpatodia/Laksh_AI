"""
FFmpeg normalization for reliable OpenCV + MediaPipe decode (HEVC/VFR/rotation).
Single source of truth for KinematicAnalyzer and pose baseline evaluation.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_video_for_pose(video_path: str, *, timeout_sec: float = 120.0) -> tuple[str, bool, bool]:
    """
    Re-encode to H.264, constant 30 fps, max 720p, bake rotation.

    Returns:
        (path_to_use, is_temporary, ffmpeg_applied) — if is_temporary, caller must os.unlink(path) when done.
        ffmpeg_applied is True only when FFmpeg ran successfully and produced the returned path.
    """
    suffix = Path(video_path).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    out_path = tmp.name

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "20",
        "-vf",
        "scale=-2:min'(720,ih)',fps=30",
        "-an",
        "-movflags",
        "+faststart",
        out_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout_sec)
        if result.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
            logger.warning(
                "FFmpeg normalisation failed (rc=%s); using original video. stderr_tail=%s",
                result.returncode,
                result.stderr.decode(errors="replace")[-800:],
            )
            os.unlink(out_path)
            return video_path, False, False
        logger.info(
            "FFmpeg normalisation OK → %s (%s KB)",
            out_path,
            os.path.getsize(out_path) // 1024,
        )
        return out_path, True, True
    except Exception as exc:
        logger.warning("FFmpeg unavailable or timed out: %s; using original video.", exc)
        try:
            os.unlink(out_path)
        except OSError:
            pass
        return video_path, False, False
