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
from app.pose.gym_baseline_metrics import (
    aggregate_metrics,
    enhance_frame_variant,
    preprocess_frame_max720,
    utility_score,
)
from app.pose.mediapipe_common import create_pose_landmarker
from app.pose.person_isolation import (
    create_person_isolation,
    normalize_person_isolation_mode,
    unmap_normalized_xy_from_crop,
)
from app.pose.preprocess import normalize_video_for_pose
from app.pose.provenance import POSE_BASELINE_SCHEMA_VERSION, build_mediapipe_pose_provenance
from app.pose.reason_codes import merge_reason_codes
from app.pose.types import PoseBaselineResult

logger = logging.getLogger(__name__)


def _extract_2d_one_pass(
    landmarker: Any,
    video_path: str,
    variant: str,
    start_sec: float | None,
    end_sec: float | None,
    *,
    person_isolation_mode: str | None = None,
) -> tuple[float, dict[str, np.ndarray], int, int, dict[str, Any] | None]:
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

    isolation = create_person_isolation(person_isolation_mode)
    if isolation is not None:
        isolation.start_clip()

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
            rgb = preprocess_frame_max720(rgb)
            full_h, full_w = rgb.shape[0], rgb.shape[1]
            if isolation is not None:
                bgr_work = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                x0, y0, x1, y1 = isolation.step(bgr_work)
                wc, hc = x1 - x0, y1 - y0
                crop_rgb = rgb[y0:y1, x0:x1] if wc > 0 and hc > 0 else rgb
                if wc <= 0 or hc <= 0:
                    x0, y0, wc, hc = 0, 0, full_w, full_h
            else:
                x0, y0, wc, hc = 0, 0, full_w, full_h
                crop_rgb = rgb

            crop_rgb = enhance_frame_variant(crop_rgb, variant)

            t_ms = max(t_ms_raw, last_t_ms + 1)
            last_t_ms = t_ms

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
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
                if plm is not None and li < len(plm) and ri < len(plm):
                    if isolation is not None:
                        lx, ly = unmap_normalized_xy_from_crop(
                            plm[li].x,
                            plm[li].y,
                            x0,
                            y0,
                            wc,
                            hc,
                            full_w,
                            full_h,
                        )
                        rx, ry = unmap_normalized_xy_from_crop(
                            plm[ri].x,
                            plm[ri].y,
                            x0,
                            y0,
                            wc,
                            hc,
                            full_w,
                            full_h,
                        )
                        lv, rv = plm[li].visibility, plm[ri].visibility
                    else:
                        lx, ly, lv = plm[li].x, plm[li].y, plm[li].visibility
                        rx, ry, rv = plm[ri].x, plm[ri].y, plm[ri].visibility
                    data_2d[f"left_{name}"].append(
                        np.array([lx, ly, lv], dtype=np.float64)
                    )
                    data_2d[f"right_{name}"].append(
                        np.array([rx, ry, rv], dtype=np.float64)
                    )
                else:
                    nan = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                    data_2d[f"left_{name}"].append(nan)
                    data_2d[f"right_{name}"].append(nan)

            frame_idx += 1
    finally:
        cap.release()

    raw_2d = {k: np.array(v, dtype=np.float64) for k, v in data_2d.items()}
    n_frames = len(next(iter(raw_2d.values()))) if raw_2d else 0
    iso_stats = (
        isolation.stats_dict(mode=person_isolation_mode)
        if isolation is not None and person_isolation_mode
        else None
    )
    if iso_stats is not None:
        iso_stats["frames_processed_pose"] = n_frames
    return fps, raw_2d, max_people, n_frames, iso_stats


def run_mediapipe_pose_baseline(
    video_path: str,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    multipass: bool = False,
    person_isolation: str | None = None,
) -> PoseBaselineResult:
    """
    Run MediaPipe pose landmarker on a clip; return detection / visibility / stability metrics.

    multipass=False: single ``baseline`` frame variant (fastest, strictest raw benchmark).
    multipass=True: same variant sweep as KinematicAnalyzer (baseline, gamma, denoise) picking best utility.

    person_isolation: optional P2 ROI mode (e.g. ``haar_mil_v1``) — see ``app.pose.person_isolation``.
    """
    try:
        iso_mode_norm = normalize_person_isolation_mode(person_isolation)
    except ValueError as e:
        return PoseBaselineResult(
            backend="mediapipe",
            video_path=video_path,
            ok=False,
            error=str(e),
            reason_codes=["pose_init_failed"],
            ffmpeg_preprocess_applied=False,
            provenance={
                "pose_baseline_schema_version": POSE_BASELINE_SCHEMA_VERSION,
                "person_isolation_config_error": str(e),
            },
        )

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
    best_iso_stats: dict[str, Any] | None = None

    try:
        for variant in variants:
            try:
                fps, raw_2d, max_p, n_fr, iso_stats = _extract_2d_one_pass(
                    landmarker,
                    norm_path,
                    variant,
                    start_sec,
                    end_sec,
                    person_isolation_mode=iso_mode_norm,
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
            u = utility_score(raw_2d, n_fr)
            if u > best_u:
                best_u = u
                best_fps = fps
                best_raw = raw_2d
                best_n = n_fr
                best_people = max_p
                best_name = variant if multipass else "baseline"
                best_iso_stats = iso_stats

        if best_iso_stats is not None:
            prov["person_isolation"] = best_iso_stats
            prov["person_isolation_note"] = (
                "max_people_seen counts poses on the (possibly cropped) tensor passed to the landmarker; "
                "with ROI enabled it often drops vs full-frame runs — compare JSONL provenance."
            )

        n_with, det_rate, vis_det, vis_all, disp = aggregate_metrics(best_raw, best_n)
        selected = "multipass_best" if multipass else "baseline_only"
        if multipass:
            selected = f"multipass_best:{best_name}"

        usable = (
            det_rate >= gate.min_detection_rate
            and vis_det >= gate.min_visibility_core_when_detected
            and best_n >= gate.min_n_frames
        )

        reasons = merge_reason_codes(det_rate, best_n, vis_det, max_people_seen=best_people)
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
