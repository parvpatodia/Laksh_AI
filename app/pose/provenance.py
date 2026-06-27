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

from app.pose.canonical import CANONICAL_JOINT_SCHEMA_VERSION

from app.pose.mediapipe_common import (
    LANDMARKER_MIN_POSE_DETECTION_CONFIDENCE,
    LANDMARKER_MIN_POSE_PRESENCE_CONFIDENCE,
    LANDMARKER_MIN_TRACKING_CONFIDENCE,
    LANDMARKER_NUM_POSES,
    MODEL_URL,
    default_model_path,
)

# Bump when JSONL semantic fields or metric definitions change.
POSE_BASELINE_SCHEMA_VERSION = "1.2.0"

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
    **extra_fields: Any,
) -> dict[str, Any]:
    """Serializable provenance dict merged into PoseBaselineResult and JSONL.

    Includes model fingerprint (SHA-256 of on-disk .task up to 64 MiB),
    package version, and explicit landmarker hyperparameters for audit trails.

    Parameters
    ----------
    extra_fields:
        A6 extension: caller-supplied fields merged last (e.g. signals_used,
        n_shots_detected, n_shots_valid, git_commit_sha, analysis_mode).
        Extra keys must not overlap with built-in keys; if they do the built-in
        value wins (no silent override of the SHA-pinned provenance record).
    """
    out: dict[str, Any] = {
        "pose_baseline_schema_version": POSE_BASELINE_SCHEMA_VERSION,
        "canonical_joint_schema_version": CANONICAL_JOINT_SCHEMA_VERSION,
        "canonical_joint_set": "coco_17_names",
        "canonical_mapping_id": "mediapipe_blazepose33_v1",
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

    # A6: merge caller-supplied fields without overriding built-ins.
    for k, v in extra_fields.items():
        if k not in out:
            out[k] = v

    return out


def build_rtmlib_rtmpose_pose_provenance(
    *,
    ffmpeg_preprocess_applied: bool,
    multipass: bool,
    rtmlib_mode: str,
    device: str,
    to_openpose: bool,
    pose_usable_gate_applied: dict[str, float | int] | None = None,
    calibration_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Provenance for ``rtmlib.Body`` (YOLOX + RTMPose ONNX zoo). First run may download zips.
    """
    out: dict[str, Any] = {
        "pose_baseline_schema_version": POSE_BASELINE_SCHEMA_VERSION,
        "canonical_joint_schema_version": CANONICAL_JOINT_SCHEMA_VERSION,
        "canonical_joint_set": "coco_17_names",
        "canonical_mapping_id": "coco17_xy_pixels_normalized_v1",
        "backend_implementation_id": "rtmlib.Body(YOLOX+RTMPose)",
        "rtmlib_mode": rtmlib_mode,
        "rtmlib_device": device,
        "rtmlib_to_openpose_skeleton": to_openpose,
        "ffmpeg_preprocess_applied": ffmpeg_preprocess_applied,
        "frame_preprocess_multipass": multipass,
        "platform_sys": sys.platform,
    }
    try:
        import onnxruntime as ort

        out["onnxruntime_version"] = getattr(ort, "__version__", "unknown")
    except Exception:
        out["onnxruntime_version"] = "import_failed"
    try:
        import rtmlib

        out["rtmlib_version"] = getattr(rtmlib, "__version__", "unknown")
    except Exception:
        out["rtmlib_version"] = "import_failed"

    if pose_usable_gate_applied is not None:
        out["pose_usable_gate_applied"] = pose_usable_gate_applied
    if calibration_record is not None:
        out["calibration"] = calibration_record
    return out
