"""
Shared MediaPipe Pose Landmarker construction (heavy model, VIDEO mode).
Used by physics_engine.KinematicAnalyzer and pose baseline evaluation.
"""
from __future__ import annotations

import logging
import ssl
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)

# Exposed for evaluation provenance — keep in sync with create_pose_landmarker().
LANDMARKER_NUM_POSES = 2
LANDMARKER_MIN_POSE_DETECTION_CONFIDENCE = 0.3
LANDMARKER_MIN_POSE_PRESENCE_CONFIDENCE = 0.3
LANDMARKER_MIN_TRACKING_CONFIDENCE = 0.3


def default_model_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "pose_landmarker_heavy.task"


def ensure_pose_model_file(model_path: Path | None = None) -> Path:
    """Download heavy landmarker task if missing."""
    path = model_path or default_model_path()
    if path.exists():
        return path
    logger.info("Downloading MediaPipe pose landmarker (heavy)…")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(MODEL_URL, context=ctx) as response, open(path, "wb") as out_file:
            out_file.write(response.read())
    except ssl.SSLError:
        logger.warning(
            "SSL verify failed for model download; retrying without verify (dev only — fix CA bundle in prod)"
        )
        ctx_dev = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx_dev.check_hostname = False
        ctx_dev.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(MODEL_URL, context=ctx_dev) as response, open(path, "wb") as out_file:
            out_file.write(response.read())
    logger.info("Pose model download complete.")
    return path


def create_pose_landmarker():
    """
    Create and return vision.PoseLandmarker (VIDEO mode) or raise.
    Caller must close() when done.
    """
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options

    model_path = ensure_pose_model_file()
    opts = vision.PoseLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=LANDMARKER_NUM_POSES,
        min_pose_detection_confidence=LANDMARKER_MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=LANDMARKER_MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=LANDMARKER_MIN_TRACKING_CONFIDENCE,
    )
    return vision.PoseLandmarker.create_from_options(opts)
