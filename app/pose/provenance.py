"""
Reproducibility metadata for pose baseline runs.

Every JSONL row should carry enough information to reproduce *approximately* the same
numeric outcome on another machine (same asset bytes, same library versions, same
preprocess path). This does not guarantee bit-identical floats across CPUs/GPUs.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

from app.pose.mediapipe_common import (
    LANDMARKER_MIN_POSE_DETECTION_CONFIDENCE,
    LANDMARKER_MIN_POSE_PRESENCE_CONFIDENCE,
    LANDMARKER_MIN_TRACKING_CONFIDENCE,
    LANDMARKER_NUM_POSES,
    MODEL_URL,
    default_model_path,
)

# Bump when JSONL semantic fields or metric definitions change.
POSE_BASELINE_SCHEMA_VERSION = "1.1.0"

LANDMARKER_OPTIONS_RECORD = {
    "running_mode": "VIDEO",
    "num_poses": LANDMARKER_NUM_POSES,
    "min_pose_detection_confidence": LANDMARKER_MIN_POSE_DETECTION_CONFIDENCE,
    "min_pose_presence_confidence": LANDMARKER_MIN_POSE_PRESENCE_CONFIDENCE,
    "min_tracking_confidence": LANDMARKER_MIN_TRACKING_CONFIDENCE,
    "model_url": MODEL_URL,
}


def _sha256_file(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> tuple[str, int]:
    """Return (hexdigest, bytes_read). Reads at most max_bytes for very large assets."""
    size = path.stat().st_size
    to_read = min(size, max_bytes)
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = to_read
        while remaining > 0:
            chunk = f.read(min(65536, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    digest = h.hexdigest()
    if size > max_bytes:
        return f"{digest}:truncated_{max_bytes}of{size}_bytes", to_read
    return digest, to_read


def build_mediapipe_pose_provenance(
    *,
    ffmpeg_preprocess_applied: bool,
    multipass: bool,
    pose_usable_gate_applied: dict[str, float | int] | None = None,
    calibration_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Serializable dict merged into PoseBaselineResult.provenance and JSONL.

    Includes model fingerprint (SHA-256 of on-disk .task up to 64 MiB), package version,
    and explicit landmarker hyperparameters for audit trails.
    """
    out: dict[str, Any] = {
        "pose_baseline_schema_version": POSE_BASELINE_SCHEMA_VERSION,
        "backend_implementation_id": "mediapipe.tasks.vision.PoseLandmarker",
        "landmarker_options": LANDMARKER_OPTIONS_RECORD,
        "ffmpeg_preprocess_applied": ffmpeg_preprocess_applied,
        "frame_preprocess_multipass": multipass,
        "platform_sys": sys.platform,
    }
    try:
        import mediapipe as mp

        out["mediapipe_package_version"] = getattr(mp, "__version__", "unknown")
    except Exception:
        out["mediapipe_package_version"] = "import_failed"

    model_path = default_model_path()
    out["pose_model_path_relative"] = str(model_path.name)
    if not model_path.is_file():
        out["pose_model_sha256"] = None
        out["pose_model_size_bytes"] = 0
        out["pose_model_status"] = "missing"
    else:
        out["pose_model_size_bytes"] = model_path.stat().st_size
        try:
            digest, nread = _sha256_file(model_path)
            out["pose_model_sha256"] = digest
            out["pose_model_bytes_hashed"] = nread
            out["pose_model_status"] = "ok"
        except OSError:
            out["pose_model_sha256"] = None
            out["pose_model_status"] = "unreadable"

    if pose_usable_gate_applied is not None:
        out["pose_usable_gate_applied"] = pose_usable_gate_applied
    if calibration_record is not None:
        out["calibration"] = calibration_record

    return out
