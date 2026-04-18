"""Build the :class:`~app.api.v1.schema.ProvenanceModel` block.

Provenance is cached at module import time because all four SHA /
version components only change across deploys -- not per request.
Computing them per request would add ~15ms and, worse, hide drift if
the calibration manifest is hot-swapped under a running process.
"""
from __future__ import annotations

import os
import subprocess
from functools import cache
from pathlib import Path

from app.api.v1.schema import ProvenanceModel
from app.gym.calibration_v0 import (
    CALIBRATION_V0_MANIFEST_VERSION,
    compute_manifest_sha256 as compute_calibration_manifest_sha256,
    load_calibration_v0,
)
from app.gym.exercises_v0 import compute_manifest_sha256 as compute_exercise_manifest_sha256
from app.gym.pipeline import DEFAULT_CALIBRATION_CONFIG
from app.pose.provenance import POSE_BASELINE_SCHEMA_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[3]


@cache
def _git_commit_sha() -> str | None:
    """Short git SHA if available -- None on read-only / non-git deploys.

    Prefer the ``LAKSH_GIT_SHA`` env var (set by the Dockerfile) so the
    deploy image does not need ``.git`` mounted.
    """
    env = os.environ.get("LAKSH_GIT_SHA")
    if env:
        return env.strip()[:40]
    try:
        out = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
        return out.strip()[:40]
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return None


@cache
def _calibration_manifest_sha() -> tuple[str, str]:
    """Return ``(sha256, manifest_version)`` for the default calibration JSON."""
    try:
        manifest = load_calibration_v0(DEFAULT_CALIBRATION_CONFIG)
    except (OSError, ValueError):
        return ("unavailable", CALIBRATION_V0_MANIFEST_VERSION)
    return compute_calibration_manifest_sha256(manifest), manifest.manifest_version


@cache
def _exercise_manifest_sha() -> str:
    """Return SHA-256 of the frozen exercise manifest."""
    return compute_exercise_manifest_sha256()


def build_provenance(model: str = "none_frames_json") -> ProvenanceModel:
    """Build the provenance block shipped with every v1 response.

    Parameters
    ----------
    model:
        The pose model used for this request. Pass
        ``"mediapipe_pose_landmarker_heavy"`` on video paths and
        ``"none_frames_json"`` on pre-extracted-frame paths.
    """
    cal_sha, cal_version = _calibration_manifest_sha()
    return ProvenanceModel(
        git_commit_sha=_git_commit_sha(),
        pose_baseline_version=POSE_BASELINE_SCHEMA_VERSION,
        exercise_manifest_sha=_exercise_manifest_sha(),
        calibration_manifest_sha=cal_sha,
        calibration_manifest_version=cal_version,
        model=model,
    )
