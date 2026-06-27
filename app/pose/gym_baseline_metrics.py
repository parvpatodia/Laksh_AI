"""
Shared 2D sequence metrics for gym Phase A baselines (MediaPipe, RTMPose, …).

Same joint layout as the original MediaPipe baseline: 12 limbs × [x, y, visibility]
in **normalized image coordinates** for x,y; visibility in [0, 1].
"""
from __future__ import annotations

import cv2
import numpy as np


def preprocess_frame_max720(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    max_dim = 720
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def enhance_frame_variant(frame: np.ndarray, variant: str) -> np.ndarray:
    if variant == "gamma_contrast":
        f = frame.astype(np.float32) / 255.0
        f = np.power(np.clip(f, 0.0, 1.0), 0.85)
        f = np.clip((f - 0.5) * 1.18 + 0.5, 0.0, 1.0)
        return (f * 255.0).astype(np.uint8)
    if variant == "denoise_sharpen":
        den = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
        gauss = cv2.GaussianBlur(den, (0, 0), 1.0)
        return cv2.addWeighted(den, 1.35, gauss, -0.35, 0)
    return frame


def frame_has_pose(plm_slice: dict[str, np.ndarray], frame_i: int) -> bool:
    lh = plm_slice["left_hip"][frame_i]
    rh = plm_slice["right_hip"][frame_i]
    return np.isfinite(lh[0]) and np.isfinite(rh[0])


def core_visibility_for_frame(plm_slice: dict[str, np.ndarray], frame_i: int) -> float | None:
    keys = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    row = []
    for k in keys:
        arr = plm_slice[k][frame_i]
        if np.isfinite(arr[0]) and np.isfinite(arr[2]):
            row.append(float(arr[2]))
    return float(np.mean(row)) if row else None


def aggregate_metrics(
    raw_2d: dict[str, np.ndarray], n_frames: int
) -> tuple[int, float, float, float, float | None]:
    if n_frames == 0:
        return 0, 0.0, 0.0, 0.0, None

    n_with = 0
    vis_when = []
    vis_all_num = []
    hip_mid_pts = []

    for i in range(n_frames):
        if frame_has_pose(raw_2d, i):
            n_with += 1
            v = core_visibility_for_frame(raw_2d, i)
            if v is not None:
                vis_when.append(v)
        for k in (
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ):
            arr = raw_2d[k][i]
            vis_all_num.append(float(arr[2]) if np.isfinite(arr[2]) else 0.0)

        lh, rh = raw_2d["left_hip"][i], raw_2d["right_hip"][i]
        if np.isfinite(lh[0]) and np.isfinite(rh[0]):
            hip_mid_pts.append(((lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5))

    det_rate = n_with / max(1, n_frames)
    vis_det = float(np.mean(vis_when)) if vis_when else 0.0
    vis_all = float(np.mean(vis_all_num)) if vis_all_num else 0.0

    disp = None
    if len(hip_mid_pts) >= 3:
        pts = np.array(hip_mid_pts, dtype=np.float64)
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        disp = float(np.median(d))

    return n_with, det_rate, vis_det, vis_all, disp


def utility_score(raw_2d: dict[str, np.ndarray], n_frames: int) -> float:
    n_with, det_rate, vis_det, _, _ = aggregate_metrics(raw_2d, n_frames)
    return n_with + vis_det * 20.0
