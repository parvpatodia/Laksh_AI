"""
RTMPose (via ``rtmlib.Body``: YOLOX + RTMPose ONNX) gym baseline metrics.

**Optional dependency:** ``pip install -r requirements-pose-optional.txt``

First inference may **download** OpenMMLab ONNX zips (network). Uses the same
aggregate metrics as MediaPipe after mapping COCO-17 pixels → normalized coords
(see ``mapping_rtmpose_coco17``).

Environment (optional):
  ``RTMPOSE_MODE`` — ``lightweight`` (default), ``balanced``, or ``performance``
  ``RTMPOSE_DEVICE`` — ``cpu`` (default), ``cuda``, or ``mps``
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
from app.pose.mapping_rtmpose_coco17 import canonical_to_gym_raw_row, coco17_pixels_to_canonical
from app.pose.person_isolation import (
    create_person_isolation,
    normalize_person_isolation_mode,
)
from app.pose.preprocess import normalize_video_for_pose
from app.pose.provenance import POSE_BASELINE_SCHEMA_VERSION, build_rtmlib_rtmpose_pose_provenance
from app.pose.reason_codes import merge_reason_codes
from app.pose.types import PoseBaselineResult

logger = logging.getLogger(__name__)

_GYM_KEYS = [
    "left_wrist",
    "right_wrist",
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _empty_data_2d() -> dict[str, list]:
    return {k: [] for k in _GYM_KEYS}


def _extract_rtmlib_one_pass(
    body: Any,
    video_path: str,
    variant: str,
    start_sec: float | None,
    end_sec: float | None,
    *,
    person_isolation_mode: str | None = None,
) -> tuple[float, dict[str, np.ndarray], int, int, dict[str, Any] | None]:
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

    isolation = create_person_isolation(person_isolation_mode)
    if isolation is not None:
        isolation.start_clip()

    data_2d = _empty_data_2d()
    frame_idx = 0
    max_people = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx < start_frame:
                frame_idx += 1
                continue
            if frame_idx >= end_frame:
                break

            bgr = preprocess_frame_max720(frame)
            full_h, full_w = bgr.shape[0], bgr.shape[1]
            if isolation is not None:
                x0, y0, x1, y1 = isolation.step(bgr)
                wc, hc = x1 - x0, y1 - y0
                crop_bgr = bgr[y0:y1, x0:x1] if wc > 0 and hc > 0 else bgr
                if wc <= 0 or hc <= 0:
                    x0, y0 = 0, 0
                    wc, hc = full_w, full_h
            else:
                x0, y0, wc, hc = 0, 0, full_w, full_h
                crop_bgr = bgr

            crop_bgr = enhance_frame_variant(crop_bgr, variant)

            keypoints, scores = body(crop_bgr)
            kp = np.asarray(keypoints, dtype=np.float64)
            sc = np.asarray(scores, dtype=np.float64).reshape(-1)

            canon = None
            n_det = 0
            if kp.size > 0 and kp.ndim == 2 and kp.shape[0] >= 17 and kp.shape[0] % 17 == 0:
                n_det = kp.shape[0] // 17
                k0 = kp[:17, :2].copy()
                nk = kp.shape[0]
                if sc.size < nk:
                    sc_full = np.zeros(nk, dtype=np.float64)
                    sc_full[: sc.size] = sc[: sc.size]
                    sc = sc_full
                s0 = sc[:17]
                if isolation is not None:
                    k0[:, 0] += x0
                    k0[:, 1] += y0
                try:
                    canon = coco17_pixels_to_canonical(
                        k0, s0, image_width=full_w, image_height=full_h
                    )
                except ValueError:
                    canon = None
                    n_det = 0

            max_people = max(max_people, n_det)
            row = canonical_to_gym_raw_row(canon)
            for sk in _GYM_KEYS:
                data_2d[sk].append(row[sk])

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


def run_rtmpose_pose_baseline(
    video_path: str,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    multipass: bool = False,
    person_isolation: str | None = None,
) -> PoseBaselineResult:
    try:
        iso_mode_norm = normalize_person_isolation_mode(person_isolation)
    except ValueError as e:
        return PoseBaselineResult(
            backend="rtmpose",
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
    mode = os.environ.get("RTMPOSE_MODE", "lightweight").strip().lower()
    if mode not in ("lightweight", "balanced", "performance"):
        mode = "lightweight"
    device = os.environ.get("RTMPOSE_DEVICE", "cpu").strip().lower()
    to_openpose = False

    prov = build_rtmlib_rtmpose_pose_provenance(
        ffmpeg_preprocess_applied=ffmpeg_applied,
        multipass=multipass,
        rtmlib_mode=mode,
        device=device,
        to_openpose=to_openpose,
        pose_usable_gate_applied=gate.as_dict(),
        calibration_record=cal_record,
    )

    try:
        from rtmlib import Body
    except ImportError as e:
        if is_temp:
            try:
                os.unlink(norm_path)
            except OSError:
                pass
        return PoseBaselineResult(
            backend="rtmpose",
            video_path=video_path,
            ok=False,
            error=(
                f"{type(e).__name__}: {e} — install optional deps: "
                "pip install -r requirements-pose-optional.txt"
            ),
            reason_codes=["pose_init_failed"],
            ffmpeg_preprocess_applied=ffmpeg_applied,
            provenance=prov,
        )

    try:
        body = Body(
            mode=mode,
            to_openpose=to_openpose,
            backend="onnxruntime",
            device=device,
        )
    except Exception as e:
        logger.exception("rtmlib Body failed to initialize")
        if is_temp:
            try:
                os.unlink(norm_path)
            except OSError:
                pass
        return PoseBaselineResult(
            backend="rtmpose",
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
                fps, raw_2d, max_p, n_fr, iso_stats = _extract_rtmlib_one_pass(
                    body,
                    norm_path,
                    variant,
                    start_sec,
                    end_sec,
                    person_isolation_mode=iso_mode_norm,
                )
            except ValueError as e:
                return PoseBaselineResult(
                    backend="rtmpose",
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
                "max_people_seen counts COCO-17 instances on the crop passed to rtmlib.Body; "
                "with ROI enabled it may differ from full-frame YOLOX behaviour."
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
            backend="rtmpose",
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
        if is_temp:
            try:
                os.unlink(norm_path)
            except OSError:
                pass
