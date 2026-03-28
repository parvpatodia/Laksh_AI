"""
MediaPipe-only pose sequence metrics for Phase A (gym manifest / backbone comparison).

Does not run basketball shot heuristics — only detection rate, core-joint visibility,
and a simple temporal stability proxy on hip midpoint.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np

from app.pose.calibration import load_gym_pose_usable_gate
from app.pose.mediapipe_common import create_pose_landmarker
from app.pose.preprocess import normalize_video_for_pose
from app.pose.provenance import build_mediapipe_pose_provenance
from app.pose.types import PoseBaselineResult, merge_reason_codes

logger = logging.getLogger(__name__)


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    max_dim = 720
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _enhance_frame_variant(frame: np.ndarray, variant: str) -> np.ndarray:
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


def _extract_2d_one_pass(
    landmarker: Any,
    video_path: str,
    variant: str,
    start_sec: float | None,
    end_sec: float | None,
) -> tuple[float, dict[str, np.ndarray], int, int]:
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_frame = 0
    end_frame = total_frames if total_frames > 0 else 999999
    if start_sec is not None and start_sec >= 0:
        start_frame = min(int(start_sec * fps), total_frames - 1) if total_frames > 0 else int(start_sec * fps)
    if end_sec is not None and end_sec > (start_sec or 0):
        end_frame = min(int(end_sec * fps), total_frames) if total_frames > 0 else int(end_sec * fps)

    # 12 joints × [x, y, visibility] (reuse basketball joint names for shoulders/arms/legs)
    joints = ["wrist", "elbow", "shoulder", "hip", "knee", "ankle"]
    sides = ["left", "right"]
    from app.pose import constants as C

    idx_map = [
        (C.LEFT_WRIST, C.RIGHT_WRIST, "wrist"),
        (C.LEFT_ELBOW, C.RIGHT_ELBOW, "elbow"),
        (C.LEFT_SHOULDER, C.RIGHT_SHOULDER, "shoulder"),
        (C.LEFT_HIP, C.RIGHT_HIP, "hip"),
        (C.LEFT_KNEE, C.RIGHT_KNEE, "knee"),
        (C.LEFT_ANKLE, C.RIGHT_ANKLE, "ankle"),
    ]

    data_2d: dict[str, list] = {f"{s}_{j}": [] for s in sides for j in joints}
    frame_idx = 0
    last_t_ms = -1
    max_people = 0
    try:
        while True:
            t_ms_raw = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx < start_frame:
                frame_idx += 1
                continue
            if frame_idx >= end_frame:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = _preprocess_frame(rgb)
            rgb = _enhance_frame_variant(rgb, variant)

            t_ms = max(t_ms_raw, last_t_ms + 1)
            last_t_ms = t_ms

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, t_ms)

            plm = None
            if result:
                pwlm = getattr(result, "pose_world_landmarks", None)
                plms = getattr(result, "pose_landmarks", None)
                n_poses = max(len(pwlm) if pwlm else 0, len(plms) if plms else 0)
                max_people = max(max_people, n_poses)
                if plms and len(plms) > 0:
                    plm = plms[0]

            for li, ri, name in idx_map:
                data_2d[f"left_{name}"].append(
                    np.array([plm[li].x, plm[li].y, plm[li].visibility], dtype=np.float64)
                    if plm is not None and li < len(plm)
                    else np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                )
                data_2d[f"right_{name}"].append(
                    np.array([plm[ri].x, plm[ri].y, plm[ri].visibility], dtype=np.float64)
                    if plm is not None and ri < len(plm)
                    else np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                )

            frame_idx += 1
    finally:
        cap.release()

    raw_2d = {k: np.array(v, dtype=np.float64) for k, v in data_2d.items()}
    n_frames = len(next(iter(raw_2d.values()))) if raw_2d else 0
    return fps, raw_2d, max_people, n_frames


def _frame_has_pose(plm_slice: dict[str, np.ndarray], frame_i: int) -> bool:
    lh = plm_slice["left_hip"][frame_i]
    rh = plm_slice["right_hip"][frame_i]
    return np.isfinite(lh[0]) and np.isfinite(rh[0])


def _core_visibility_for_frame(plm_slice: dict[str, np.ndarray], frame_i: int) -> float | None:
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


def _aggregate_metrics(raw_2d: dict[str, np.ndarray], n_frames: int) -> tuple[int, float, float, float, float | None]:
    if n_frames == 0:
        return 0, 0.0, 0.0, 0.0, None

    n_with = 0
    vis_when = []
    vis_all_num = []
    hip_mid_pts = []

    for i in range(n_frames):
        if _frame_has_pose(raw_2d, i):
            n_with += 1
            v = _core_visibility_for_frame(raw_2d, i)
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


def _utility(raw_2d: dict[str, np.ndarray], n_frames: int) -> float:
    n_with, det_rate, vis_det, _, _ = _aggregate_metrics(raw_2d, n_frames)
    return n_with + vis_det * 20.0


def run_mediapipe_pose_baseline(
    video_path: str,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    multipass: bool = False,
) -> PoseBaselineResult:
    """
    Run MediaPipe pose landmarker on a clip; return detection / visibility / stability metrics.

    multipass=False: single ``baseline`` frame variant (fastest, strictest raw benchmark).
    multipass=True: same variant sweep as KinematicAnalyzer (baseline, gamma, denoise) picking best utility.
    """
    norm_path, is_temp, ffmpeg_applied = normalize_video_for_pose(video_path)
    gate, cal_record = load_gym_pose_usable_gate()
    prov = build_mediapipe_pose_provenance(
        ffmpeg_preprocess_applied=ffmpeg_applied,
        multipass=multipass,
        pose_usable_gate_applied=gate.as_dict(),
        calibration_record=cal_record,
    )
    try:
        landmarker = create_pose_landmarker()
    except Exception as e:
        if is_temp:
            try:
                os.unlink(norm_path)
            except OSError:
                pass
        return PoseBaselineResult(
            backend="mediapipe",
            video_path=video_path,
            ok=False,
            error=f"{type(e).__name__}: {e}",
            reason_codes=["pose_init_failed"],
            ffmpeg_preprocess_applied=ffmpeg_applied,
            provenance=prov,
        )

    variants = ["baseline", "gamma_contrast", "denoise_sharpen"] if multipass else ["baseline"]
    best_fps = 30.0
    best_raw: dict[str, np.ndarray] = {}
    best_n = 0
    best_people = 0
    best_name = "baseline"
    best_u = -1.0

    try:
        for variant in variants:
            try:
                fps, raw_2d, max_p, n_fr = _extract_2d_one_pass(
                    landmarker, norm_path, variant, start_sec, end_sec
                )
            except ValueError as e:
                return PoseBaselineResult(
                    backend="mediapipe",
                    video_path=video_path,
                    ok=False,
                    error=str(e),
                    reason_codes=["decode_error"],
                    ffmpeg_preprocess_applied=ffmpeg_applied,
                    provenance=prov,
                )
            u = _utility(raw_2d, n_fr)
            if u > best_u:
                best_u = u
                best_fps = fps
                best_raw = raw_2d
                best_n = n_fr
                best_people = max_p
                best_name = variant if multipass else "baseline"

        n_with, det_rate, vis_det, vis_all, disp = _aggregate_metrics(best_raw, best_n)
        selected = "multipass_best" if multipass else "baseline_only"
        if multipass:
            selected = f"multipass_best:{best_name}"

        usable = (
            det_rate >= gate.min_detection_rate
            and vis_det >= gate.min_visibility_core_when_detected
            and best_n >= gate.min_n_frames
        )

        reasons = merge_reason_codes(det_rate, best_n, vis_det)
        if not usable:
            reasons.append("pose_not_usable_heuristic")

        return PoseBaselineResult(
            backend="mediapipe",
            video_path=video_path,
            ok=True,
            n_frames=best_n,
            n_frames_with_pose=n_with,
            detection_rate=round(det_rate, 4),
            visibility_core_when_detected=round(vis_det, 4),
            visibility_core_all_frames=round(vis_all, 4),
            hip_mid_displacement_median_norm=round(disp, 6) if disp is not None else None,
            max_people_seen=best_people,
            selected_pass=selected,
            pose_usable_heuristic=usable,
            reason_codes=reasons,
            fps=round(best_fps, 2),
            ffmpeg_preprocess_applied=ffmpeg_applied,
            provenance=prov,
        )
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
