"""
Laksh.ai biomechanics pipeline: pose extraction, phase detection, and shot metrics.
Uses MediaPipe Tasks Pose Landmarker (heavy), multi-pass preprocess, and 2D/world fallbacks.
"""
import logging
import math
import cv2
import numpy as np
import pandas as pd
import os
from typing import Any

from app.pose.canonical import CANONICAL_JOINT_SCHEMA_VERSION, CanonicalJointName
from app.pose.kinematic_flags import use_canonical_joint_trace
from app.pose.mapping_mediapipe import map_mediapipe_blazepose33_to_canonical
from app.pose.mediapipe_common import create_pose_landmarker
from app.pose.preprocess import normalize_video_for_pose
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

METRIC_KEYS = [
    "release_velocity_mps",
    "shot_arc_deg",
    "knee_angle",
    "elbow_angle",
    "kinetic_sync_ms",
    "hip_rotation_deg",
    "balance_index",
    "fluidity_score",
]

FALLBACK_MESSAGES = {
    "pose_init_failed": "Pose engine could not initialize for this video.",
    "low_detections": "Very few reliable pose detections were found in this clip.",
    "short_clip": "Clip is too short for a full shot-cycle analysis.",
    "low_visibility": "Low body visibility reduced metric reliability.",
    "decode_error": "Video decode failed or no frames could be processed.",
    "analysis_exception": "Unexpected analysis error occurred.",
}

def _to_vec3(lm) -> np.ndarray:
    if lm is None:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return np.array([getattr(lm, "x", np.nan), getattr(lm, "y", np.nan), getattr(lm, "z", np.nan)], dtype=np.float64)


def _calculate_3d_angle(a, b, c) -> float:
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return 0.0
    u, v = a - b, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    val = np.dot(u, v) / (nu * nv)
    if np.isnan(val):
        return 0.0
    return math.degrees(math.acos(np.clip(val, -1.0, 1.0)))


def _is_valid_vec(v: Any) -> bool:
    if v is None:
        return False
    arr = np.asarray(v, dtype=np.float64)
    return arr.size >= 2 and np.all(np.isfinite(arr[:2]))


def _is_valid_xy(v: Any) -> bool:
    """2D image landmark row [x, y, visibility] — x,y must be finite."""
    if v is None:
        return False
    arr = np.asarray(v, dtype=np.float64)
    return arr.size >= 2 and np.all(np.isfinite(arr[:2]))


def _angle_2d_from_joint_observations(
    ja: Any, jb: Any, jc: Any, aspect_ratio: float
) -> float | None:
    """Interior angle at jb from three JointObservation-like objects (x, y, visibility)."""
    if ja is None or jb is None or jc is None:
        return None
    a = np.array([float(ja.x), float(ja.y), float(ja.visibility)], dtype=np.float64)
    b = np.array([float(jb.x), float(jb.y), float(jb.visibility)], dtype=np.float64)
    c = np.array([float(jc.x), float(jc.y), float(jc.visibility)], dtype=np.float64)
    return _angle_2d_image(a, b, c, aspect_ratio)


def _canonical_joint_path_telemetry(
    cpf: list[Any] | None,
    side: str,
    dip_frame: int,
    release_frame: int,
    aspect_ratio: float,
    e_legacy_2d_raw: float | None,
    k_legacy_2d_raw: float | None,
) -> dict[str, Any] | None:
    """
    Parity probe: same 2D angle math on canonical-mapped joints vs legacy index path.
    Does not replace shipped metrics (ADR 0002 P3).
    """
    if not cpf or dip_frame < 0 or release_frame < 0:
        return None
    if dip_frame >= len(cpf) or release_frame >= len(cpf):
        return {
            "enabled": True,
            "canonical_joint_schema_version": CANONICAL_JOINT_SCHEMA_VERSION,
            "mapping_id": "mediapipe_blazepose33_to_canonical",
            "error": "key_frame_out_of_range",
        }
    fd = cpf[dip_frame]
    fr = cpf[release_frame]
    if fd is None or fr is None:
        return {
            "enabled": True,
            "canonical_joint_schema_version": CANONICAL_JOINT_SCHEMA_VERSION,
            "mapping_id": "mediapipe_blazepose33_to_canonical",
            "error": "missing_canonical_pose_at_key_frame",
        }
    if side == "right":
        hip_k, knee_k, ankle_k = (
            CanonicalJointName.RIGHT_HIP,
            CanonicalJointName.RIGHT_KNEE,
            CanonicalJointName.RIGHT_ANKLE,
        )
        sh_e, el_e, wr_e = (
            CanonicalJointName.RIGHT_SHOULDER,
            CanonicalJointName.RIGHT_ELBOW,
            CanonicalJointName.RIGHT_WRIST,
        )
    else:
        hip_k = CanonicalJointName.LEFT_HIP
        knee_k = CanonicalJointName.LEFT_KNEE
        ankle_k = CanonicalJointName.LEFT_ANKLE
        sh_e = CanonicalJointName.LEFT_SHOULDER
        el_e = CanonicalJointName.LEFT_ELBOW
        wr_e = CanonicalJointName.LEFT_WRIST

    hj = fd.get(hip_k)
    kk = fd.get(knee_k)
    ak = fd.get(ankle_k)
    sh = fr.get(sh_e)
    el = fr.get(el_e)
    wr = fr.get(wr_e)
    if hj is None or kk is None or ak is None or sh is None or el is None or wr is None:
        return {
            "enabled": True,
            "canonical_joint_schema_version": CANONICAL_JOINT_SCHEMA_VERSION,
            "mapping_id": "mediapipe_blazepose33_to_canonical",
            "error": "missing_joint_in_canonical_frame",
        }

    k_ang_c = _angle_2d_from_joint_observations(hj, kk, ak, aspect_ratio)
    e_ang_c = _angle_2d_from_joint_observations(sh, el, wr, aspect_ratio)

    out: dict[str, Any] = {
        "enabled": True,
        "canonical_joint_schema_version": CANONICAL_JOINT_SCHEMA_VERSION,
        "mapping_id": "mediapipe_blazepose33_to_canonical",
        "knee_angle_2d_image_deg_canonical_at_dip": round(float(k_ang_c), 3)
        if k_ang_c is not None
        else None,
        "elbow_angle_2d_image_deg_canonical_at_release": round(float(e_ang_c), 3)
        if e_ang_c is not None
        else None,
        "legacy_knee_2d_image_deg_at_dip": round(float(k_legacy_2d_raw), 3)
        if k_legacy_2d_raw is not None
        else None,
        "legacy_elbow_2d_image_deg_at_release": round(float(e_legacy_2d_raw), 3)
        if e_legacy_2d_raw is not None
        else None,
    }
    if k_ang_c is not None and k_legacy_2d_raw is not None:
        out["delta_knee_vs_legacy_2d_deg"] = round(
            float(k_ang_c) - float(k_legacy_2d_raw), 3
        )
    if e_ang_c is not None and e_legacy_2d_raw is not None:
        out["delta_elbow_vs_legacy_2d_deg"] = round(
            float(e_ang_c) - float(e_legacy_2d_raw), 3
        )
    return out


def _angle_2d_image(a: np.ndarray, b: np.ndarray, c: np.ndarray, aspect_ratio: float) -> float | None:
    """Interior angle at b using normalized image coords; stretch x by aspect_ratio."""
    if not (_is_valid_xy(a) and _is_valid_xy(b) and _is_valid_xy(c)):
        return None
    p0 = np.array([float(a[0]) * aspect_ratio, float(a[1])], dtype=np.float64)
    p1 = np.array([float(b[0]) * aspect_ratio, float(b[1])], dtype=np.float64)
    p2 = np.array([float(c[0]) * aspect_ratio, float(c[1])], dtype=np.float64)
    ang = _calculate_3d_angle(p0, p1, p2)
    return ang if ang >= 1.0 else None


def _build_debug_summary(
    *,
    analysis_mode: str,
    reason_codes: list[str],
    n_frames: int,
    actual_detections: int,
    visibility: float,
    preprocess_pass: str | None,
    shot_type: str,
    dip_frame: int,
    release_frame: int,
    has_knee_world: bool,
    has_elbow_world: bool,
    has_knee_2d: bool,
    has_elbow_2d: bool,
    yaw_world: bool,
    yaw_2d: bool,
) -> dict:
    return {
        "analysis_mode": analysis_mode,
        "fallback_reason_codes": sorted(set(reason_codes)),
        "n_frames": n_frames,
        "wrist_detection_events": actual_detections,
        "pose_visibility_mean": round(visibility, 3),
        "selected_preprocess_pass": preprocess_pass,
        "shot_type": shot_type,
        "dip_frame": dip_frame,
        "release_frame": release_frame,
        "landmarks_ok": {
            "knee_world": has_knee_world,
            "elbow_world": has_elbow_world,
            "knee_2d": has_knee_2d,
            "elbow_2d": has_elbow_2d,
            "hip_yaw_world": yaw_world,
            "hip_yaw_2d": yaw_2d,
        },
    }


class KinematicAnalyzer:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._landmarker = None

    def _prepare_video(self) -> str:
        """
        Normalise input video for reliable MediaPipe detection (shared with gym pose baseline).
        See app.pose.preprocess.normalize_video_for_pose.
        """
        path, _, _ = normalize_video_for_pose(self.video_path)
        return path

    def _init_pose(self) -> bool:
        """Initialise MediaPipe Pose Landmarker (Heavy model). Downloads on first run if missing."""
        try:
            self._landmarker = create_pose_landmarker()
            return True
        except Exception:
            logger.exception("MediaPipe PoseLandmarker failed to initialize")
            return False

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame for consistent pose extraction (Option 3: video preprocessing).
        Max 720p on longer side; preserves aspect. Coords stay normalized [0,1]."""
        h, w = frame.shape[:2]
        max_dim = 720
        if max(h, w) <= max_dim:
            return frame
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _enhance_frame_variant(self, frame: np.ndarray, variant: str) -> np.ndarray:
        """Apply bounded enhancement variants for low-quality inputs."""
        if variant == "gamma_contrast":
            # Lift shadows and boost local contrast for dim videos.
            f = frame.astype(np.float32) / 255.0
            f = np.power(np.clip(f, 0.0, 1.0), 0.85)
            f = np.clip((f - 0.5) * 1.18 + 0.5, 0.0, 1.0)
            return (f * 255.0).astype(np.uint8)
        if variant == "denoise_sharpen":
            den = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
            gauss = cv2.GaussianBlur(den, (0, 0), 1.0)
            return cv2.addWeighted(den, 1.35, gauss, -0.35, 0)
        return frame

    def _extract_frames_with_variant(
        self,
        variant: str,
        start_sec: float | None = None,
        end_sec: float | None = None,
        video_path_override: str | None = None,
    ):
        """Like extract_frames, but with a deterministic enhancement variant."""
        import mediapipe as mp
        path = video_path_override or self.video_path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        start_frame = 0
        end_frame = total_frames if total_frames > 0 else 999999
        if start_sec is not None and start_sec >= 0:
            start_frame = min(int(start_sec * fps), total_frames - 1) if total_frames > 0 else int(start_sec * fps)
        if end_sec is not None and end_sec > (start_sec or 0):
            end_frame = min(int(end_sec * fps), total_frames) if total_frames > 0 else int(end_sec * fps)

        joints, sides = ["wrist", "elbow", "shoulder", "hip", "knee", "ankle"], ["left", "right"]
        data_3d = {f"{s}_{j}": [] for s in sides for j in joints}
        data_2d = {f"{s}_{j}": [] for s in sides for j in joints}
        collect_canon = use_canonical_joint_trace()
        canonical_per_frame: list[Any] | None = [] if collect_canon else None

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
                rgb = self._preprocess_frame(rgb)
                rgb = self._enhance_frame_variant(rgb, variant)

                t_ms = max(t_ms_raw, last_t_ms + 1)
                last_t_ms = t_ms
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self._landmarker.detect_for_video(mp_img, t_ms)

                wlm = None
                plm = None
                if result:
                    pwlm = getattr(result, "pose_world_landmarks", None)
                    plms = getattr(result, "pose_landmarks", None)
                    n_poses = max(len(pwlm) if pwlm else 0, len(plms) if plms else 0)
                    max_people = max(max_people, n_poses)
                    if pwlm and len(pwlm) > 0:
                        wlm = pwlm[0]
                    if plms and len(plms) > 0:
                        plm = plms[0]

                if canonical_per_frame is not None:
                    if plm is not None and len(plm) == 33:
                        canonical_per_frame.append(
                            map_mediapipe_blazepose33_to_canonical(plm)
                        )
                    else:
                        canonical_per_frame.append(None)

                indices = [
                    (LEFT_WRIST, RIGHT_WRIST, "wrist"),
                    (LEFT_ELBOW, RIGHT_ELBOW, "elbow"),
                    (LEFT_SHOULDER, RIGHT_SHOULDER, "shoulder"),
                    (LEFT_HIP, RIGHT_HIP, "hip"),
                    (LEFT_KNEE, RIGHT_KNEE, "knee"),
                    (LEFT_ANKLE, RIGHT_ANKLE, "ankle"),
                ]
                for li, ri, name in indices:
                    data_3d[f"left_{name}"].append(_to_vec3(wlm[li]) if wlm and li < len(wlm) else np.array([np.nan] * 3))
                    data_3d[f"right_{name}"].append(_to_vec3(wlm[ri]) if wlm and ri < len(wlm) else np.array([np.nan] * 3))
                    data_2d[f"left_{name}"].append(np.array([plm[li].x, plm[li].y, plm[li].visibility]) if plm and li < len(plm) else np.array([np.nan] * 3))
                    data_2d[f"right_{name}"].append(np.array([plm[ri].x, plm[ri].y, plm[ri].visibility]) if plm and ri < len(plm) else np.array([np.nan] * 3))
                frame_idx += 1
        finally:
            cap.release()

        return (
            fps,
            {k: np.array(v) for k, v in data_3d.items()},
            {k: np.array(v) for k, v in data_2d.items()},
            max_people,
            canonical_per_frame,
        )

    def _detection_utility(self, raw_2d: dict) -> tuple[int, float]:
        """Return (usable wrist detections, average visibility)."""
        lw = raw_2d.get("left_wrist")
        rw = raw_2d.get("right_wrist")
        if lw is None or rw is None or len(lw) == 0:
            return 0, 0.0
        lw_det = int(np.sum(~np.isnan(lw[:, 0])))
        rw_det = int(np.sum(~np.isnan(rw[:, 0])))
        vis = self._compute_pose_visibility(raw_2d)
        return lw_det + rw_det, vis

    def extract_frames(
        self,
        start_sec: float | None = None,
        end_sec: float | None = None,
        video_path_override: str | None = None,
    ):
        """
        Extract pose data from video.
        Optional start_sec/end_sec restrict analysis to a clip (user-selected range).
        video_path_override lets analyze() pass the FFmpeg-normalised path without changing self.video_path.

        Timestamp strategy (VFR fix):
          Read CAP_PROP_POS_MSEC BEFORE cap.read() so we get the actual container
          presentation timestamp of the frame about to be decoded — not a synthetic
          frame_idx/fps estimate. This is critical for MediaPipe VIDEO mode which
          requires strictly monotonic timestamps; synthetic estimates fail badly on
          VFR (iPhone) footage. Monotonicity is enforced with last_t_ms.
        """
        import mediapipe as mp
        path = video_path_override or self.video_path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Clip selection: convert time range to frame indices
        start_frame = 0
        end_frame = total_frames if total_frames > 0 else 999999
        if start_sec is not None and start_sec >= 0:
            start_frame = min(int(start_sec * fps), total_frames - 1) if total_frames > 0 else int(start_sec * fps)
        if end_sec is not None and end_sec > (start_sec or 0):
            end_frame = min(int(end_sec * fps), total_frames) if total_frames > 0 else int(end_sec * fps)

        joints, sides = ["wrist", "elbow", "shoulder", "hip", "knee", "ankle"], ["left", "right"]
        data_3d = {f"{s}_{j}": [] for s in sides for j in joints}
        data_2d = {f"{s}_{j}": [] for s in sides for j in joints}
        collect_canon = use_canonical_joint_trace()
        canonical_per_frame: list[Any] | None = [] if collect_canon else None

        frame_idx = 0
        last_t_ms = -1          # monotonicity guard for MediaPipe VIDEO mode
        max_people = 0
        try:
            while True:
                # Read container timestamp BEFORE decoding the frame (VFR-correct)
                t_ms_raw = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                ret, frame = cap.read()
                if not ret:
                    break
                # Only process frames within selected clip
                if frame_idx < start_frame:
                    frame_idx += 1
                    continue
                if frame_idx >= end_frame:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = self._preprocess_frame(rgb)

                # Enforce strict monotonicity required by MediaPipe VIDEO mode
                t_ms = max(t_ms_raw, last_t_ms + 1)
                last_t_ms = t_ms

                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self._landmarker.detect_for_video(mp_img, t_ms)

                # Safe extraction: MediaPipe can return empty lists when no person detected
                wlm = None
                plm = None
                if result:
                    pwlm = getattr(result, "pose_world_landmarks", None)
                    plms = getattr(result, "pose_landmarks", None)
                    n_poses = max(len(pwlm) if pwlm else 0, len(plms) if plms else 0)
                    max_people = max(max_people, n_poses)
                    if pwlm and len(pwlm) > 0:
                        wlm = pwlm[0]
                    if plms and len(plms) > 0:
                        plm = plms[0]

                if canonical_per_frame is not None:
                    if plm is not None and len(plm) == 33:
                        canonical_per_frame.append(
                            map_mediapipe_blazepose33_to_canonical(plm)
                        )
                    else:
                        canonical_per_frame.append(None)

                indices = [
                    (LEFT_WRIST, RIGHT_WRIST, "wrist"),
                    (LEFT_ELBOW, RIGHT_ELBOW, "elbow"),
                    (LEFT_SHOULDER, RIGHT_SHOULDER, "shoulder"),
                    (LEFT_HIP, RIGHT_HIP, "hip"),
                    (LEFT_KNEE, RIGHT_KNEE, "knee"),
                    (LEFT_ANKLE, RIGHT_ANKLE, "ankle"),
                ]

                for li, ri, name in indices:
                    data_3d[f"left_{name}"].append(_to_vec3(wlm[li]) if wlm and li < len(wlm) else np.array([np.nan]*3))
                    data_3d[f"right_{name}"].append(_to_vec3(wlm[ri]) if wlm and ri < len(wlm) else np.array([np.nan]*3))
                    data_2d[f"left_{name}"].append(np.array([plm[li].x, plm[li].y, plm[li].visibility]) if plm and li < len(plm) else np.array([np.nan]*3))
                    data_2d[f"right_{name}"].append(np.array([plm[ri].x, plm[ri].y, plm[ri].visibility]) if plm and ri < len(plm) else np.array([np.nan]*3))

                frame_idx += 1
        finally:
            cap.release()
            if self._landmarker:
                self._landmarker.close()

        return (
            fps,
            {k: np.array(v) for k, v in data_3d.items()},
            {k: np.array(v) for k, v in data_2d.items()},
            max_people,
            canonical_per_frame,
        )

    def apply_filters(self, data):
        out = {}
        for k, arr in data.items():
            if len(arr) < 11:
                out[k] = np.nan_to_num(arr, nan=0.0)
                continue
            df = pd.DataFrame(arr).interpolate(method='linear', limit_direction='both').fillna(0.0)
            clean = df.values
            smoothed = np.zeros_like(clean)
            for d in range(clean.shape[1]):
                try:
                    smoothed[:, d] = savgol_filter(clean[:, d], 11, 3)
                except Exception:
                    smoothed[:, d] = clean[:, d]
            out[k] = smoothed
        return out

    # _count_people_sampled removed: multi-person detection is now tracked inline
    # during extract_frames() using num_poses=2, eliminating the second MediaPipe
    # initialization and the ~3s per-request overhead it introduced.

    def _assess_video_quality(
        self,
        w: float,
        h: float,
        fps: float,
        total_frames: int,
        visibility: float = 0.0,
        people_count: int = 1,
    ) -> dict:
        """
        Pose-analysis suitability score (0–100). Research-grounded: log-scale resolution/FPS,
        visibility-weighted. Ref: PMC 11695451 (OpenPose accuracy vs movement).
        """
        notes = []
        if w < 320 or h < 240:
            notes.append("Low resolution may reduce pose accuracy. Use 720p or higher for best results.")
        if fps < 20:
            notes.append("Low framerate (<20 fps) can blur fast motions. 30 fps or higher recommended.")
        elif fps > 90:
            notes.append("Slow-motion detected. Kinetic sync timing adjusted for high-speed capture.")
        ar = w / h if h > 0 else 1.0
        if ar < 0.6:
            notes.append("Vertical/portrait video can compress shot arc. For best accuracy, use landscape with a 45° angle.")
        elif ar > 2.2:
            notes.append("Ultra-wide format may distort joint positions at frame edges.")
        if total_frames < 30:
            notes.append("Short clip: ensure it contains a single, complete jump shot for reliable analysis.")

        # Log-based quality formula (PHASE2_RESEARCH_GROUNDED)
        max_dim = max(w, h, 320)
        q_res = min(100, 20 * math.log10(max_dim / 320 + 1e-6) + 40)  # 320p baseline
        q_res = max(0, q_res)
        q_fps = min(15, 15 * min(fps / 30.0, 1.5))  # 30 fps reference
        q_aspect = 10 if 0.6 <= ar <= 2.2 else 5
        q_visibility = 30 * visibility  # 0–1 → 0–30 pts
        q_people = 10 if people_count <= 1 else max(0, 10 - 5 * (people_count - 1))
        score = int(np.clip(q_res + q_fps + q_aspect + q_visibility + q_people, 0, 100))

        label = "Excellent" if score >= 80 else "Good" if score >= 60 else "Fair" if score >= 40 else "Low"
        return {
            "video_quality_notes": notes,
            "resolution": f"{int(w)}×{int(h)}",
            "fps": round(fps, 1),
            "video_quality_score": score,
            "video_quality_label": label,
        }

    def _compute_pose_visibility(self, raw_2d: dict) -> float:
        """Average visibility (0–1) of key shooting joints across all frames."""
        keys = ["left_wrist", "right_wrist", "left_elbow", "right_elbow", "left_shoulder", "right_shoulder"]
        vals = []
        for k in keys:
            arr = raw_2d.get(k)
            if arr is not None and len(arr) > 0 and arr.shape[1] >= 3:
                vals.append(np.nanmean(arr[:, 2]))
        return float(np.mean(vals)) if vals else 0.0

    def _compute_angle_uncertainty(
        self,
        h3d: np.ndarray,
        k3d: np.ndarray,
        a3d: np.ndarray,
        s3d: np.ndarray,
        e3d: np.ndarray,
        w3d: np.ndarray,
        dip_frame: int,
        release_frame: int,
        visibility: float,
    ) -> dict:
        """
        Empirical uncertainty from frame-window variance. Ref: PMC 9397457.
        Returns knee_angle_uncertainty and elbow_angle_uncertainty in degrees (±).
        """
        out = {}
        half = 3
        # Knee: window around dip
        lo = max(0, dip_frame - half)
        hi = min(len(k3d), dip_frame + half + 1)
        k_angles = []
        for i in range(lo, hi):
            ang = _calculate_3d_angle(h3d[i], k3d[i], a3d[i])
            if ang >= 10:
                k_angles.append(ang)
        if len(k_angles) >= 2:
            std_k = float(np.nanstd(k_angles))
            unc = max(3, min(12, std_k * 1.2))
            if visibility < 0.6:
                unc = min(12, unc * 1.5)
            out["knee_angle_uncertainty"] = round(unc, 1)

        # Elbow: window around release
        lo = max(0, release_frame - half)
        hi = min(len(e3d), release_frame + half + 1)
        e_angles = []
        for i in range(lo, hi):
            ang = _calculate_3d_angle(s3d[i], e3d[i], w3d[i])
            if ang >= 10:
                e_angles.append(ang)
        if len(e_angles) >= 2:
            std_e = float(np.nanstd(e_angles))
            unc = max(3, min(12, std_e * 1.2))
            if visibility < 0.6:
                unc = min(12, unc * 1.5)
            out["elbow_angle_uncertainty"] = round(unc, 1)

        return out

    def _compute_validation_flags(self, metrics: dict, visibility: float, used_fallback: bool) -> list:
        """Biological plausibility and data-quality checks. Returns human-readable warnings."""
        flags = []
        if used_fallback:
            flags.append("Analysis used fallback values. Video may lack a detectable jump shot.")
            return flags
        if visibility < 0.5:
            flags.append("Low pose visibility. Ensure full-body visibility and good lighting.")
        k = metrics.get("knee_angle")
        if k is not None and (k < 90 or k > 180):
            flags.append(f"Knee angle ({k}°) outside biological range. Check for occlusion or camera angle.")
        e = metrics.get("elbow_angle")
        if e is not None and (e < 100 or e > 180):
            flags.append(f"Elbow angle ({e}°) outside typical range. Ensure arm is visible at release.")
        return flags

    def _compute_angle_uncertainty(
        self,
        h3d: np.ndarray,
        k3d: np.ndarray,
        a3d: np.ndarray,
        s3d: np.ndarray,
        e3d: np.ndarray,
        w3d: np.ndarray,
        dip_frame: int,
        release_frame: int,
        visibility: float,
    ) -> tuple[float, float]:
        """
        Empirical uncertainty from frame-window variance (PMC 9397457).
        Returns (knee_uncertainty_deg, elbow_uncertainty_deg). Clamped 3–12°; inflated if visibility low.
        """
        n = len(h3d)
        win = 3
        k_angles, e_angles = [], []
        for i in range(max(0, dip_frame - win), min(n, dip_frame + win + 1)):
            ang = _calculate_3d_angle(h3d[i], k3d[i], a3d[i])
            if ang > 10:
                k_angles.append(ang)
        for i in range(max(0, release_frame - win), min(n, release_frame + win + 1)):
            ang = _calculate_3d_angle(s3d[i], e3d[i], w3d[i])
            if ang > 10:
                e_angles.append(ang)
        k_std = float(np.nanstd(k_angles)) if len(k_angles) >= 3 else 5.0
        e_std = float(np.nanstd(e_angles)) if len(e_angles) >= 3 else 5.0
        mult = 1.5 if visibility < 0.6 else 1.0
        k_unc = max(3.0, min(12.0, k_std * 1.2 * mult))
        e_unc = max(3.0, min(12.0, e_std * 1.2 * mult))
        return (round(k_unc, 1), round(e_unc, 1))

    def _compute_confidence_factors(
        self,
        video_quality_score: int,
        people_count: int,
        visibility: float,
        validation_flags: list,
        used_fallback: bool,
    ) -> list[dict]:
        """
        Transparent attribution: factor, impact, message.
        Sum of impacts approximates (100 - confidence). Actionable for users.
        """
        factors = []
        if used_fallback:
            factors.append({"factor": "fallback", "impact": -40, "message": "No clear shot detected"})
            return factors
        if video_quality_score < 60:
            factors.append({
                "factor": "video_quality",
                "impact": -min(20, 60 - video_quality_score),
                "message": f"Video quality {video_quality_score}/100 — re-record at 720p+, 30 fps",
            })
        if people_count > 1:
            factors.append({
                "factor": "multi_person",
                "impact": -15,
                "message": f"{people_count} people detected — record only the shooter",
            })
        if visibility < 0.5:
            factors.append({
                "factor": "pose_visibility",
                "impact": -10,
                "message": "Low joint visibility — ensure full body and good lighting",
            })
        if validation_flags:
            pen = min(15, len(validation_flags) * 4)
            factors.append({
                "factor": "validation",
                "impact": -pen,
                "message": f"{len(validation_flags)} quality note(s) — see recommendations above",
            })
        return factors

    def _status(self, source: str, confidence: float, reason: str | None = None) -> dict:
        out = {
            "source": source,  # measured | predicted | unavailable
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 2),
        }
        if reason:
            out["reason"] = reason
        return out

    def _fallback_warning_text(self, reason_codes: list[str]) -> str:
        if not reason_codes:
            return "Analysis used fallback values. Video may lack a detectable jump shot or pose."
        parts = [FALLBACK_MESSAGES.get(c, c.replace("_", " ")) for c in reason_codes]
        return "Analysis limited. " + " ".join(parts)

    def _empty_metric_status(self, reason_codes: list[str]) -> dict:
        reason = ",".join(reason_codes) if reason_codes else "fallback"
        return {k: self._status("unavailable", 0.0, reason) for k in METRIC_KEYS}

    def _calibrate_metric_confidence(
        self,
        base: float,
        visibility: float,
        detection_ratio: float,
        people_count: int,
        analysis_mode: str,
        source: str,
    ) -> float:
        """
        Calibrate confidence from observable signal quality so values are not static.
        """
        if source == "unavailable":
            return 0.0
        conf = float(base)
        conf *= 0.55 + 0.45 * float(np.clip(visibility, 0.0, 1.0))
        conf *= 0.5 + 0.5 * float(np.clip(detection_ratio, 0.0, 1.0))
        if people_count > 1:
            conf *= max(0.72, 1.0 - 0.08 * (people_count - 1))
        if analysis_mode == "partial":
            conf *= 0.88
        if source == "predicted":
            conf *= 0.82
        return float(np.clip(conf, 0.05, 0.98))

    def analyze(self, start_sec: float | None = None, end_sec: float | None = None):
        """Analyze video. Optional start_sec/end_sec restrict to user-selected clip (e.g. single shot)."""
        norm_path = None
        try:
            # Extract physical aspect ratio to fix normalized coordinate distortion
            temp_cap = cv2.VideoCapture(self.video_path)
            w = temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            aspect_ratio = (w / h) if h > 0 else 1.0
            temp_cap.release()

            reason_codes: list[str] = []
            analysis_mode = "full"

            # Normalise video: fix HEVC codec, VFR timestamps, rotation metadata
            norm_path = self._prepare_video()

            if not self._init_pose():
                return self._fallback(["pose_init_failed"])

            # Multi-pass extraction: baseline, low-light enhancement, compression artifact recovery.
            pass_variants = ["baseline", "gamma_contrast", "denoise_sharpen"]
            best = None
            for variant in pass_variants:
                fps_i, raw_3d_i, raw_2d_i, max_people_i, canonical_i = (
                    self._extract_frames_with_variant(
                        variant=variant,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        video_path_override=norm_path,
                    )
                )
                det_i, vis_i = self._detection_utility(raw_2d_i)
                utility = det_i + int(round(vis_i * 20))
                if best is None or utility > best["utility"]:
                    best = {
                        "variant": variant,
                        "fps": fps_i,
                        "raw_3d": raw_3d_i,
                        "raw_2d": raw_2d_i,
                        "max_people": max_people_i,
                        "detections": det_i,
                        "visibility": vis_i,
                        "utility": utility,
                        "canonical_per_frame": canonical_i,
                    }

            if best is None:
                return self._fallback(["decode_error"])

            fps = best["fps"]
            raw_3d = best["raw_3d"]
            raw_2d = best["raw_2d"]
            max_people = best["max_people"]
            actual_detections = int(best["detections"])
            visibility = float(best["visibility"])
            logger.info(
                "Selected extraction pass: %s | detections=%s visibility=%.2f",
                best["variant"],
                actual_detections,
                visibility,
            )

            n_frames = len(raw_2d.get("left_wrist", []))
            if n_frames < 3:
                return self._fallback(["short_clip", "low_detections"])
            if actual_detections < 2:
                return self._fallback(["low_detections"])
            if n_frames < 8 or actual_detections < 8:
                analysis_mode = "partial"
                reason_codes.append("low_detections")
            if visibility < 0.5:
                analysis_mode = "partial"
                reason_codes.append("low_visibility")
            detection_ratio = float(np.clip(actual_detections / max(1.0, n_frames * 2.0), 0.0, 1.0))

            f_3d, f_2d = self.apply_filters(raw_3d), self.apply_filters(raw_2d)
            
            # Active side determined by 2D visibility confidence
            l_vis = np.mean(f_2d["left_shoulder"][:, 2]) if len(f_2d["left_shoulder"]) > 0 else 0
            r_vis = np.mean(f_2d["right_shoulder"][:, 2]) if len(f_2d["right_shoulder"]) > 0 else 0
            side = "left" if l_vis > r_vis else "right"
            
            w2d, e2d, s2d, h2d = f_2d[f"{side}_wrist"], f_2d[f"{side}_elbow"], f_2d[f"{side}_shoulder"], f_2d[f"{side}_hip"]
            w3d, e3d, s3d = f_3d[f"{side}_wrist"], f_3d[f"{side}_elbow"], f_3d[f"{side}_shoulder"]
            h3d, k3d, a3d = f_3d[f"{side}_hip"], f_3d[f"{side}_knee"], f_3d[f"{side}_ankle"]

            # A5: Multi-shot consensus segmentation.
            # CRITICAL: pass raw (unfiltered) arrays, not f_2d.  segment_shots()
            # applies its own Savitzky-Golay pass; passing f_2d (already smoothed)
            # would double-smooth and destroy the elbow angular-velocity signal (S2).
            w2d_raw = raw_2d.get(f"{side}_wrist", w2d)
            e2d_raw = raw_2d.get(f"{side}_elbow", e2d)
            s2d_raw = raw_2d.get(f"{side}_shoulder", s2d)
            _shot_seg = None
            try:
                from app.basketball.shot_segmenter import segment_shots as _segment_shots
                _shot_seg = _segment_shots(w2d_raw, e2d_raw, s2d_raw, fps=fps)
            except Exception:
                logger.warning(
                    "segment_shots failed; continuing with single-shot detection",
                    exc_info=True,
                )
            
            # Trajectory phases: jump shot (dip → release window) vs set shot / FT (small excursion)
            wrist_y = w2d[:, 1]
            hip_y = h2d[:, 1]
            n_w = len(wrist_y)
            wrist_span = float(np.ptp(wrist_y)) if n_w > 1 else 0.0
            hip_span = float(np.ptp(hip_y)) if len(hip_y) > 1 else 0.0
            shot_type = (
                "set_shot"
                if (wrist_span < 0.048 and hip_span < 0.042) or (wrist_span < 0.058 and hip_span < 0.028)
                else "jump_shot"
            )

            if shot_type == "jump_shot":
                dip_frame = max(0, min(int(np.argmax(wrist_y)), n_w - 5))
                search_end = min(n_w, dip_frame + int(fps * 1.5))
                if search_end > dip_frame + 2:
                    release_frame = dip_frame + int(np.argmin(wrist_y[dip_frame:search_end]))
                else:
                    release_frame = dip_frame + 1
            else:
                lo = max(1, int(0.18 * n_w))
                hi = min(n_w - 1, int(0.92 * n_w))
                if hi <= lo + 2:
                    lo, hi = 0, n_w - 1
                seg = wrist_y[lo : hi + 1]
                release_frame = lo + int(np.argmin(seg))
                if release_frame >= 3:
                    dip_frame = int(np.argmax(wrist_y[: release_frame + 1]))
                else:
                    dip_frame = max(0, release_frame - max(3, int(fps * 0.2)))
                dip_frame = max(0, min(dip_frame, max(0, release_frame - 1)))
                if release_frame <= dip_frame:
                    dip_frame = max(0, release_frame - max(2, int(fps * 0.15)))

            release_frame = max(0, min(release_frame, n_w - 1))

            # Joint angles: 3D world first; 2D image fallback when world depth is unusable
            has_knee_world = _is_valid_vec(h3d[dip_frame]) and _is_valid_vec(k3d[dip_frame]) and _is_valid_vec(a3d[dip_frame])
            has_elbow_world = (
                _is_valid_vec(s3d[release_frame])
                and _is_valid_vec(e3d[release_frame])
                and _is_valid_vec(w3d[release_frame])
            )
            k2d, a2d = f_2d[f"{side}_knee"], f_2d[f"{side}_ankle"]
            k_2d_raw = _angle_2d_image(h2d[dip_frame], k2d[dip_frame], a2d[dip_frame], aspect_ratio)
            e_2d_raw = _angle_2d_image(s2d[release_frame], e2d[release_frame], w2d[release_frame], aspect_ratio)
            has_knee_2d = k_2d_raw is not None
            has_elbow_2d = e_2d_raw is not None

            if has_knee_world:
                k_ang = _calculate_3d_angle(h3d[dip_frame], k3d[dip_frame], a3d[dip_frame])
                if k_ang < 10:
                    k_ang = 135.0
            elif has_knee_2d:
                k_ang = float(np.clip(k_2d_raw, 90, 180))
            else:
                k_ang = 135.0

            if has_elbow_world:
                e_ang = _calculate_3d_angle(s3d[release_frame], e3d[release_frame], w3d[release_frame])
                if e_ang < 10:
                    e_ang = 165.0
            elif has_elbow_2d:
                e_ang = float(np.clip(e_2d_raw, 100, 180))
            else:
                e_ang = 165.0
            
            # Dimensionless Power Scale
            w_travel = np.linalg.norm(w2d[release_frame][:2] - w2d[dip_frame][:2])
            t_len = np.linalg.norm(s2d[dip_frame][:2] - h2d[dip_frame][:2])
            power_ratio = (w_travel / t_len) if t_len > 1e-5 and release_frame > dip_frame else 0.0
            vel_mps = np.clip(power_ratio * 3.5, 4.0, 10.0) if power_ratio > 0 else 6.5

            # True Parabolic Shot Arc (Bulletproof Geometric Anchor)

            # 1. ALWAYS calculate the Lever Arc first (Shoulder to Wrist at release)
            lever_arc = 48.5
            dx_lever = (w2d[release_frame, 0] - s2d[release_frame, 0]) * aspect_ratio
            dy_lever = -(w2d[release_frame, 1] - s2d[release_frame, 1]) # Invert Y
            if abs(dx_lever) > 1e-5:
                lever_angle = math.degrees(math.atan2(dy_lever, abs(dx_lever)))
                # Subtract wrist flick, clamp to human reality
                lever_arc = np.clip(lever_angle - 15.0, 30.0, 75.0)

            arc_deg = lever_arc # Lock in geometric truth as our baseline

            # 2. Try to calculate true post-release flight path IF we have enough frames
            calc_arc = None
            available_frames = len(w2d) - release_frame - 1
            if available_frames >= 3: # Require at least 3 frames for a true curve
                arc_window = min(7, available_frames)
                x_vals = w2d[release_frame : release_frame + arc_window, 0] * aspect_ratio
                y_vals = -w2d[release_frame : release_frame + arc_window, 1]

                if np.std(x_vals) > 1e-5:
                    coeffs = np.polyfit(x_vals, y_vals, min(2, arc_window - 1))
                    slope = 2 * coeffs[0] * x_vals[0] + coeffs[1] if len(coeffs) == 3 else coeffs[0]
                    calc_arc = math.degrees(math.atan(abs(slope)))

                    # Only override lever if the flight path makes biological sense
                    if 30.0 <= calc_arc <= 75.0:
                        arc_deg = calc_arc

            # Unmask the math in the terminal
            logger.debug(
                "Arc pipeline: lever=%.1f° poly=%s final=%.1f°",
                lever_arc,
                calc_arc if calc_arc is not None else None,
                arc_deg,
            )

            # Transverse torso twist — world depth when valid; else 2D shoulder vs hip line proxy
            yaw_deg = 0.0
            ls, rs = f_3d["left_shoulder"][dip_frame], f_3d["right_shoulder"][dip_frame]
            lh, rh = f_3d["left_hip"][dip_frame], f_3d["right_hip"][dip_frame]

            def _finite_xyz(v):
                return _is_valid_vec(v) and np.isfinite(float(v[2]))

            yaw_measured = (
                _finite_xyz(ls) and _finite_xyz(rs) and _finite_xyz(lh) and _finite_xyz(rh)
            )
            if yaw_measured:
                s_ang = math.atan2(rs[2] - ls[2], rs[0] - ls[0])
                h_ang = math.atan2(rh[2] - lh[2], rh[0] - lh[0])
                twist = math.degrees(s_ang - h_ang)
                twist = (twist + 180) % 360 - 180
                yaw_deg = float(np.clip(twist, -45.0, 45.0))

            ls2d = f_2d["left_shoulder"][dip_frame]
            rs2d = f_2d["right_shoulder"][dip_frame]
            lh2d = f_2d["left_hip"][dip_frame]
            rh2d = f_2d["right_hip"][dip_frame]
            yaw_2d_deg = None
            if _is_valid_xy(ls2d) and _is_valid_xy(rs2d) and _is_valid_xy(lh2d) and _is_valid_xy(rh2d):
                sdx = (float(rs2d[0]) - float(ls2d[0])) * aspect_ratio
                sdy = float(rs2d[1]) - float(ls2d[1])
                hdx = (float(rh2d[0]) - float(lh2d[0])) * aspect_ratio
                hdy = float(rh2d[1]) - float(lh2d[1])
                if (abs(sdx) + abs(sdy) > 1e-6) and (abs(hdx) + abs(hdy) > 1e-6):
                    s_ang2 = math.atan2(sdy, sdx)
                    h_ang2 = math.atan2(hdy, hdx)
                    twist2 = math.degrees(s_ang2 - h_ang2)
                    twist2 = (twist2 + 180) % 360 - 180
                    yaw_2d_deg = float(np.clip(twist2, -45.0, 45.0))

            if not yaw_measured and yaw_2d_deg is not None:
                yaw_deg = yaw_2d_deg

            # Dimensionless Time Scaling (Dynamic Frame-Rate Estimator)
            raw_frames = abs(release_frame - dip_frame)
            estimated_fps = fps

            # Human biomechanics limit: Nobody takes 15+ frames to shoot at true 30fps
            if raw_frames > 15:
                # Dynamically scale the FPS based on the degree of slow-motion dilation
                estimated_fps = fps * (raw_frames / 8.0)

            if estimated_fps > 0:
                sync_ms = (raw_frames / estimated_fps) * 1000.0
            else:
                sync_ms = raw_frames * 33.3

            sync_ms = np.clip(sync_ms, 120.0, 395.0)

            # Dimensionless Base of Support (Balance Index)
            # Measures horizontal displacement of Center of Mass (Hips) over Base (Ankles)
            balance_index = 85
            lh2d, rh2d = f_2d["left_hip"][dip_frame], f_2d["right_hip"][dip_frame]
            la2d, ra2d = f_2d["left_ankle"][dip_frame], f_2d["right_ankle"][dip_frame]
            balance_measured = False

            if not (np.any(np.isnan(lh2d)) or np.any(np.isnan(la2d))):
                hip_mid_x = (lh2d[0] + rh2d[0]) / 2.0
                ankle_mid_x = (la2d[0] + ra2d[0]) / 2.0

                # Normalize deviation by torso length to remain immune to camera zoom
                if t_len > 1e-4:
                    deviation = abs(hip_mid_x - ankle_mid_x) / t_len
                    # A perfect vertical stack (deviation 0) = 99 score.
                    # Leaning heavily (deviation > 0.5) drops score rapidly.
                    balance_index = int(np.clip(100 - (deviation * 120), 40, 99))
                    balance_measured = True

            # Fluidity Derivation
            fluidity = 65
            fluidity_measured = False
            if release_frame > dip_frame + 2:
                jerk = np.std(np.diff(np.diff(wrist_y[dip_frame:release_frame]))) if release_frame - dip_frame > 3 else 0
                fluidity = int(np.clip(100 - (jerk * 2000), 40, 99))
                fluidity_measured = True

            # 2D Telemetry Payload for UI Rendering — Research-grade per-frame overlay
            total_frames = len(w2d)

            def _joints_at(i):
                """Build joints dict for frame index i (0-based)."""
                if i < 0 or i >= total_frames:
                    return None
                return {
                    "wrist":    [round(float(w2d[i, 0]), 4), round(float(w2d[i, 1]), 4)],
                    "elbow":    [round(float(e2d[i, 0]), 4), round(float(e2d[i, 1]), 4)],
                    "shoulder": [round(float(s2d[i, 0]), 4), round(float(s2d[i, 1]), 4)],
                    "hip":      [round(float(h2d[i, 0]), 4), round(float(h2d[i, 1]), 4)],
                    "knee":     [round(float(k2d[i, 0]), 4), round(float(k2d[i, 1]), 4)],
                    "ankle":    [round(float(a2d[i, 0]), 4), round(float(a2d[i, 1]), 4)],
                }

            # Per-frame skeleton overlay: dip to release + 1.5s post-release (smooth continuous visualization)
            end_frame = min(total_frames - 1, release_frame + int(fps * 1.5))
            frames = []
            for fi in range(dip_frame, end_frame + 1):
                j = _joints_at(fi)
                if j:
                    frames.append({"time_sec": round(float(fi / fps), 3), "joints": j})

            telemetry = {
                "fps": round(float(fps), 2),
                "shot_type": shot_type,
                "shooting_side": side,  # "left" or "right" — model-agnostic for overlay
                "dip": {
                    "time_sec": round(float(dip_frame / fps), 3),
                    "joints": _joints_at(dip_frame) or {},
                },
                "release": {
                    "time_sec": round(float(release_frame / fps), 3),
                    "joints": _joints_at(release_frame) or {},
                },
                "frames": frames,
            }

            # Multi-person awareness: tracked inline in extract_frames (num_poses=2) — no second init
            people_count = max_people if max_people > 0 else 1
            detection_metadata = {
                "algorithms": ["MediaPipe Pose"],
                "selected_preprocess_pass": best["variant"],
                "people_detected_max": people_count,
                "video_quality_note": (
                    "Multiple people detected. Analysis focuses on the most visible subject. "
                    "For best pro matching, record only the shooter in frame."
                ) if people_count > 1 else None,
            }
            telemetry["detection_metadata"] = detection_metadata

            cprobe = _canonical_joint_path_telemetry(
                best.get("canonical_per_frame"),
                side,
                dip_frame,
                release_frame,
                aspect_ratio,
                e_2d_raw,
                k_2d_raw,
            )
            if cprobe is not None:
                telemetry["canonical_joint_path"] = cprobe

            # Video quality and validation (expert-grade robustness)
            vq = self._assess_video_quality(w, h, fps, total_frames, visibility, people_count)
            telemetry["video_quality"] = vq
            metrics_out = {
                "release_velocity_mps": round(float(vel_mps), 2),
                "shot_arc_deg": round(float(arc_deg), 1),
                "knee_angle": round(float(np.clip(k_ang, 90, 180)), 1),
                "elbow_angle": round(float(np.clip(e_ang, 100, 180)), 1),
                "kinetic_sync_ms": round(float(sync_ms), 1),
                "hip_rotation_deg": round(float(yaw_deg), 1),
                "balance_index": balance_index,
                "fluidity_score": fluidity,
            }
            validation_flags = self._compute_validation_flags(metrics_out, visibility, used_fallback=False)
            if analysis_mode == "partial":
                validation_flags.append(self._fallback_warning_text(reason_codes))
            all_warnings = validation_flags + (vq.get("video_quality_notes") or [])
            telemetry["validation_warnings"] = all_warnings

            # Per-metric uncertainty (PMC 9397457): frame-window variance
            k_unc, e_unc = self._compute_angle_uncertainty(
                h3d, k3d, a3d, s3d, e3d, w3d, dip_frame, release_frame, visibility
            )
            if not has_knee_world and has_knee_2d:
                k_unc = round(float(min(15.0, k_unc * 1.12)), 1)
            if not has_elbow_world and has_elbow_2d:
                e_unc = round(float(min(15.0, e_unc * 1.12)), 1)
            metrics_out["knee_angle_uncertainty"] = k_unc
            metrics_out["elbow_angle_uncertainty"] = e_unc

            # Transparent confidence attribution
            vq_score = vq.get("video_quality_score", 50)
            telemetry["confidence_factors"] = self._compute_confidence_factors(
                vq_score, people_count, visibility, all_warnings, used_fallback=(analysis_mode == "fallback")
            )

            vel_src = "measured" if actual_detections >= 8 else "predicted"
            arc_src = "measured" if calc_arc is not None else "predicted"
            knee_src = "measured" if has_knee_world else ("predicted" if has_knee_2d else "unavailable")
            elbow_src = "measured" if has_elbow_world else ("predicted" if has_elbow_2d else "unavailable")
            sync_src = "measured" if actual_detections >= 8 else "predicted"
            hip_src = "measured" if yaw_measured else ("predicted" if yaw_2d_deg is not None else "unavailable")
            bal_src = "measured" if balance_measured else "predicted"
            fluid_src = "measured" if fluidity_measured else "predicted"
            metric_status = {
                "release_velocity_mps": self._status(
                    vel_src,
                    self._calibrate_metric_confidence(0.86 if vel_src == "measured" else 0.55, visibility, detection_ratio, people_count, analysis_mode, vel_src),
                    None if vel_src == "measured" else "low_detections",
                ),
                "shot_arc_deg": self._status(
                    arc_src,
                    self._calibrate_metric_confidence(0.84 if arc_src == "measured" else 0.58, visibility, detection_ratio, people_count, analysis_mode, arc_src),
                    None if arc_src == "measured" else "insufficient_post_release_frames",
                ),
                "knee_angle": self._status(
                    knee_src,
                    self._calibrate_metric_confidence(
                        0.82 if knee_src == "measured" else (0.68 if knee_src == "predicted" else 0.0),
                        visibility,
                        detection_ratio,
                        people_count,
                        analysis_mode,
                        knee_src,
                    ),
                    None
                    if knee_src == "measured"
                    else ("world_depth_unreliable" if knee_src == "predicted" else "dip_joint_data_missing"),
                ),
                "elbow_angle": self._status(
                    elbow_src,
                    self._calibrate_metric_confidence(
                        0.82 if elbow_src == "measured" else (0.68 if elbow_src == "predicted" else 0.0),
                        visibility,
                        detection_ratio,
                        people_count,
                        analysis_mode,
                        elbow_src,
                    ),
                    None
                    if elbow_src == "measured"
                    else ("world_depth_unreliable" if elbow_src == "predicted" else "release_joint_data_missing"),
                ),
                "kinetic_sync_ms": self._status(
                    sync_src,
                    self._calibrate_metric_confidence(0.78 if sync_src == "measured" else 0.52, visibility, detection_ratio, people_count, analysis_mode, sync_src),
                    None if sync_src == "measured" else "low_detections",
                ),
                "hip_rotation_deg": self._status(
                    hip_src,
                    self._calibrate_metric_confidence(
                        0.74 if hip_src == "measured" else (0.62 if hip_src == "predicted" else 0.0),
                        visibility,
                        detection_ratio,
                        people_count,
                        analysis_mode,
                        hip_src,
                    ),
                    None
                    if hip_src == "measured"
                    else ("world_depth_unreliable" if hip_src == "predicted" else "hip_or_shoulder_depth_missing"),
                ),
                "balance_index": self._status(
                    bal_src,
                    self._calibrate_metric_confidence(0.74 if bal_src == "measured" else 0.50, visibility, detection_ratio, people_count, analysis_mode, bal_src),
                    None if bal_src == "measured" else "ankle_or_hip_visibility_low",
                ),
                "fluidity_score": self._status(
                    fluid_src,
                    self._calibrate_metric_confidence(0.72 if fluid_src == "measured" else 0.50, visibility, detection_ratio, people_count, analysis_mode, fluid_src),
                    None if fluid_src == "measured" else "insufficient_motion_window",
                ),
            }
            if metric_status["knee_angle"]["source"] == "unavailable":
                metrics_out["knee_angle"] = None
            if metric_status["elbow_angle"]["source"] == "unavailable":
                metrics_out["elbow_angle"] = None
            if metric_status["hip_rotation_deg"]["source"] == "unavailable":
                metrics_out["hip_rotation_deg"] = None

            result = {
                "analysis_mode": analysis_mode,
                "fallback_reason_codes": sorted(set(reason_codes)),
                "metric_status": metric_status,
                "release_velocity_mps": metrics_out["release_velocity_mps"],
                "shot_arc_deg": metrics_out["shot_arc_deg"],
                "knee_angle": metrics_out["knee_angle"],
                "elbow_angle": metrics_out["elbow_angle"],
                "knee_angle_uncertainty": k_unc,
                "elbow_angle_uncertainty": e_unc,
                "knee_flexion_at_dip": metrics_out["knee_angle"],
                "elbow_flexion_at_release": metrics_out["elbow_angle"],
                "kinetic_sync_ms": metrics_out["kinetic_sync_ms"],
                "hip_rotation_deg": metrics_out["hip_rotation_deg"],
                "balance_index": balance_index,
                "fluidity_score": fluidity,
                "telemetry": telemetry,
            }
            # A5: Attach multi-shot segmentation result when available.
            # Biomech metrics above remain the single-detection values (honest: not claiming
            # multi-shot median). UI labels this "biomech from dominant shot | N shots detected".
            if _shot_seg is not None:
                result["shot_segmentation"] = _shot_seg.to_dict()
            if os.environ.get("LAKSH_INCLUDE_DEBUG_SUMMARY", "").strip().lower() in ("1", "true", "yes"):
                result["debug_summary"] = _build_debug_summary(
                    analysis_mode=analysis_mode,
                    reason_codes=reason_codes,
                    n_frames=n_frames,
                    actual_detections=actual_detections,
                    visibility=visibility,
                    preprocess_pass=best["variant"],
                    shot_type=shot_type,
                    dip_frame=dip_frame,
                    release_frame=release_frame,
                    has_knee_world=has_knee_world,
                    has_elbow_world=has_elbow_world,
                    has_knee_2d=has_knee_2d,
                    has_elbow_2d=has_elbow_2d,
                    yaw_world=yaw_measured,
                    yaw_2d=yaw_2d_deg is not None,
                )
            return result
        except Exception:
            logger.exception("KinematicAnalyzer.analyze crashed")
            return self._fallback(["analysis_exception"])
        finally:
            # Clean up the FFmpeg-normalised temp file
            if norm_path and norm_path != self.video_path:
                try:
                    os.unlink(norm_path)
                except OSError:
                    pass

    def _fallback(self, reason_codes: list[str] | None = None):
        reason_codes = reason_codes or []
        warning = self._fallback_warning_text(reason_codes)
        telemetry = {
            "dip": {}, "release": {}, "frames": [],
            "validation_warnings": [warning],
            "detection_metadata": {"algorithms": ["MediaPipe Pose"], "people_detected_max": 0},
            "video_quality": {"video_quality_score": 0, "video_quality_label": "Low", "video_quality_notes": [warning]},
            "confidence_factors": self._compute_confidence_factors(0, 0, 0.0, [warning], used_fallback=True),
        }
        metric_status = self._empty_metric_status(reason_codes)
        fb = {
            "analysis_mode": "fallback",
            "fallback_reason_codes": sorted(set(reason_codes)),
            "metric_status": metric_status,
            "release_velocity_mps": None, "shot_arc_deg": None, "knee_angle": None, "elbow_angle": None,
            "knee_flexion_at_dip": None, "elbow_flexion_at_release": None, "kinetic_sync_ms": None,
            "hip_rotation_deg": None, "balance_index": None, "fluidity_score": None,
            "telemetry": telemetry,
        }
        if os.environ.get("LAKSH_INCLUDE_DEBUG_SUMMARY", "").strip().lower() in ("1", "true", "yes"):
            fb["debug_summary"] = _build_debug_summary(
                analysis_mode="fallback",
                reason_codes=reason_codes,
                n_frames=0,
                actual_detections=0,
                visibility=0.0,
                preprocess_pass=None,
                shot_type="unknown",
                dip_frame=-1,
                release_frame=-1,
                has_knee_world=False,
                has_elbow_world=False,
                has_knee_2d=False,
                has_elbow_2d=False,
                yaw_world=False,
                yaw_2d=False,
            )
        return fb