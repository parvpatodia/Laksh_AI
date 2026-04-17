"""
Expected checksums for downloadable inference artifacts (reproducible CI / eval).

MediaPipe hosts `pose_landmarker_heavy.task` at a versioned URL; if Google replaces
the blob without a path bump, update this constant and document in CHANGELOG or PR.

See: scripts/download_pose_model.py (same DEFAULT_URL as here).
"""
from __future__ import annotations

# SHA-256 of float16/1 heavy task from DEFAULT_URL in download_pose_model.py
# (verify: curl -sL "$URL" | shasum -a 256)
POSE_LANDMARKER_HEAVY_TASK_SHA256 = (
    "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b"
)
