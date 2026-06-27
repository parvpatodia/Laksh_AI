"""Convert pose-backend output into the canonical-frame format consumed by
:mod:`app.gym.rep_segmenter` and :mod:`app.gym.rep_features`.

The mediapipe baseline stores per-frame joint data as a flat dict of
``{side}_{joint}: np.ndarray([x, y, visibility])``.  The rep-level modules
expect ``list[dict[str, JointObservation] | None]`` — one dict per frame, keyed
by canonical joint name string, or ``None`` when the whole frame has no pose.

This module contains two entrypoints:

``raw_2d_to_canonical_frames``
    Pure converter: takes the ``raw_2d`` dict already extracted from video and
    produces the frame list.  Used by both the script and tests (no MediaPipe
    needed for the converter itself).

``extract_canonical_frames``
    Full pipeline: opens a video, runs MediaPipe, then converts.  Requires
    ``mediapipe`` and ``cv2``; imported lazily so the rest of the module can be
    imported in CI environments that lack those packages.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from app.pose.canonical import JointObservation

# Mapping from raw_2d key (e.g. "left_shoulder") to canonical joint name string.
# Only the 12 joints extracted by _extract_2d_one_pass are present here.
_RAW_KEY_TO_CANONICAL: dict[str, str] = {
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
}


def raw_2d_to_canonical_frames(
    raw_2d: dict[str, np.ndarray],
) -> list[dict[str, JointObservation] | None]:
    """Convert ``raw_2d`` (from ``_extract_2d_one_pass``) to canonical frames.

    Each frame is either ``None`` (all joints NaN => no pose detected) or a
    ``dict[canonical_name, JointObservation]`` for all joints that are finite.
    A frame where some joints are NaN and others are finite keeps the finite
    ones; the joint is simply absent from the dict.  Callers (``_get_joint``)
    return ``None`` for absent joints, which feeds into missingness accounting.
    """
    if not raw_2d:
        return []
    # All arrays must have the same length; take the length from the first one.
    first_key = next(iter(raw_2d))
    n_frames = len(raw_2d[first_key])

    frames: list[dict[str, JointObservation] | None] = []
    for i in range(n_frames):
        frame_dict: dict[str, JointObservation] = {}
        for raw_key, arr in raw_2d.items():
            canonical = _RAW_KEY_TO_CANONICAL.get(raw_key)
            if canonical is None:
                continue  # unknown key; skip
            row = arr[i]  # shape (3,): [x, y, visibility] or [nan, nan, nan]
            if not np.all(np.isfinite(row)):
                continue  # frame has no pose for this joint; omit from dict
            frame_dict[canonical] = JointObservation(
                x=float(row[0]),
                y=float(row[1]),
                z=0.0,  # 2D pipeline; depth not available
                visibility=float(row[2]),
            )
        frames.append(frame_dict if frame_dict else None)
    return frames


def frames_json_to_canonical_frames(
    frames_raw: list[dict[str, Any] | None],
) -> list[dict[str, JointObservation] | None]:
    """Convert a pre-serialised frame list (from ``--frames-json`` input) to
    canonical frames.

    Expected element shape::

        {"left_shoulder": {"x": 0.4, "y": 0.5, "z": 0.0, "visibility": 0.9}, ...}

    or ``null`` / ``None`` for no-pose frames.
    """
    out: list[dict[str, JointObservation] | None] = []
    for raw in frames_raw:
        if raw is None:
            out.append(None)
            continue
        if not isinstance(raw, dict):
            out.append(None)
            continue
        frame_dict: dict[str, JointObservation] = {}
        for joint_name, obs in raw.items():
            if not isinstance(obs, dict):
                continue
            x = obs.get("x")
            y = obs.get("y")
            z = obs.get("z", 0.0)
            vis = obs.get("visibility", 1.0)
            if x is None or y is None:
                continue
            try:
                xf, yf, zf, vf = float(x), float(y), float(z), float(vis)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(xf) and math.isfinite(yf)):
                continue
            frame_dict[str(joint_name)] = JointObservation(x=xf, y=yf, z=zf, visibility=vf)
        out.append(frame_dict if frame_dict else None)
    return out


def extract_canonical_frames(
    video_path: str | Path,
    *,
    multipass: bool = False,
    person_isolation: str | None = None,
) -> tuple[float, list[dict[str, JointObservation] | None]]:
    """Run MediaPipe on ``video_path`` and return ``(fps, canonical_frames)``.

    Requires ``mediapipe`` and ``cv2`` at runtime (imported lazily).

    Parameters
    ----------
    video_path:
        Path to the video file.
    multipass:
        Mirror the ``multipass`` option of :func:`run_mediapipe_pose_baseline`:
        if ``True``, try baseline/gamma/denoise variants and pick the best.
    person_isolation:
        Optional person-isolation mode (e.g. ``"haar_mil_v1"``).
    """
    import sys
    from pathlib import Path as _Path

    # Resolve repo root so imports work when called from scripts/.
    _repo_root = _Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from app.pose.mediapipe_baseline import _extract_2d_one_pass  # type: ignore[attr-defined]
    from app.pose.mediapipe_common import create_pose_landmarker
    from app.pose.person_isolation import normalize_person_isolation_mode
    from app.pose.preprocess import normalize_video_for_pose
    import os

    iso_mode = normalize_person_isolation_mode(person_isolation)
    norm_path, is_temp, _ = normalize_video_for_pose(str(video_path))
    landmarker = create_pose_landmarker()
    variants = (
        ["baseline", "gamma_contrast", "denoise_sharpen"] if multipass else ["baseline"]
    )
    best_fps = 30.0
    best_raw: dict[str, np.ndarray] = {}
    best_u = -1.0

    try:
        from app.pose.gym_baseline_metrics import utility_score

        for variant in variants:
            fps, raw_2d, _max_p, n_fr, _iso = _extract_2d_one_pass(
                landmarker, norm_path, variant, None, None, person_isolation_mode=iso_mode
            )
            u = utility_score(raw_2d, n_fr)
            if u > best_u:
                best_u = u
                best_fps = fps
                best_raw = raw_2d
    finally:
        try:
            landmarker.close()
        except Exception:
            pass
        if is_temp:
            try:
                os.unlink(norm_path)
            except OSError:
                pass

    return best_fps, raw_2d_to_canonical_frames(best_raw)
