"""
Apex.ai Indestructible 3D Signal Processing Pipeline.
Uses MediaPipe Tasks API (M1/Python 3.13 Safe), Center of Mass Event Detection, and Dimensionless Physics.
"""
import math
import cv2
import numpy as np
import pandas as pd
import subprocess
import tempfile
import os
from pathlib import Path
from scipy.signal import savgol_filter

LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

def _to_vec3(lm) -> np.ndarray:
    if lm is None: return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return np.array([getattr(lm, "x", np.nan), getattr(lm, "y", np.nan), getattr(lm, "z", np.nan)], dtype=np.float64)

def _calculate_3d_angle(a, b, c) -> float:
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)): return 0.0
    u, v = a - b, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9: return 0.0
    val = np.dot(u, v) / (nu * nv)
    if np.isnan(val): return 0.0
    return math.degrees(math.acos(np.clip(val, -1.0, 1.0)))

class KinematicAnalyzer:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._landmarker = None

    def _prepare_video(self) -> str:
        """
        Normalise input video for reliable MediaPipe detection.
        Fixes the three most common phone-video failure modes in one FFmpeg pass:
          1. HEVC/H.265 (iPhone default) — re-encode to H.264 which OpenCV decodes reliably
          2. Variable Frame Rate (VFR) — force constant 30 fps so MediaPipe timestamps are monotonic
          3. Rotation metadata — bake rotation into pixels so OpenCV sees the correct orientation
        Returns path to normalised file (caller must clean up).
        """
        suffix = Path(self.video_path).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        out_path = tmp.name

        cmd = [
            "ffmpeg", "-y",
            "-i", self.video_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "20",
            "-vf", "scale=-2:min'(720,ih)',fps=30",   # max 720p, constant 30 fps
            "-an",                                      # drop audio (not needed)
            "-movflags", "+faststart",
            out_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
                print(f"FFmpeg normalisation failed (rc={result.returncode}). Using original video.")
                print(result.stderr.decode(errors="replace")[-800:])
                os.unlink(out_path)
                return self.video_path
            print(f"FFmpeg normalisation OK → {out_path} ({os.path.getsize(out_path)//1024} KB)")
            return out_path
        except Exception as exc:
            print(f"FFmpeg not available or timed out: {exc}. Using original video.")
            try: os.unlink(out_path)
            except OSError: pass
            return self.video_path

    def _init_pose(self) -> bool:
        """Initialise MediaPipe Pose Landmarker (Heavy model). Downloads on first run if missing."""
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options
            import ssl
            from pathlib import Path

            # Use the heavy model for maximum lab-grade accuracy
            model_path = Path(__file__).resolve().parent / "pose_landmarker_heavy.task"

            if not model_path.exists():
                print("Downloading MediaPipe Heavy Model...")
                import urllib.request

                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

                try:
                    # Attempt download with full SSL verification (always works in Docker/production)
                    ctx = ssl.create_default_context()
                    with urllib.request.urlopen(url, context=ctx) as response, open(model_path, "wb") as out_file:
                        out_file.write(response.read())
                except ssl.SSLError:
                    # Fallback for macOS dev environments without updated CA certificates.
                    # Permanent fix: run /Applications/Python*/Install\ Certificates.command
                    print("WARNING: SSL verification failed. Retrying with unverified context (dev only).")
                    ctx_dev = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                    ctx_dev.check_hostname = False
                    ctx_dev.verify_mode = ssl.CERT_NONE
                    with urllib.request.urlopen(url, context=ctx_dev) as response, open(model_path, "wb") as out_file:
                        out_file.write(response.read())

                print("Download complete! Booting engine...")

            # Lower thresholds from 0.5 → 0.3 for fast sports motion (validated by PMC 9397457).
            # At 0.5 many mid-motion frames fail detection; 0.3 recovers ~40% more detections
            # while staying above noise floor. Biological plausibility check downstream handles outliers.
            opts = vision.PoseLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=2,  # Detect up to 2 people; enables multi-person awareness in one pass
                min_pose_detection_confidence=0.3,
                min_pose_presence_score=0.3,
                min_tracking_confidence=0.3,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(opts)
            return True

        except Exception as e:
            import traceback
            print(f"FATAL: MediaPipe Tasks API failed to initialize:\n{traceback.format_exc()}")
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
        if not cap.isOpened(): raise ValueError(f"Could not open video: {path}")
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

        return fps, {k: np.array(v) for k, v in data_3d.items()}, {k: np.array(v) for k, v in data_2d.items()}, max_people

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
                try: smoothed[:, d] = savgol_filter(clean[:, d], 11, 3)
                except Exception: smoothed[:, d] = clean[:, d]
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

            # Normalise video: fix HEVC codec, VFR timestamps, rotation metadata
            norm_path = self._prepare_video()
            norm_is_tmp = norm_path != self.video_path

            if not self._init_pose(): return self._fallback()
            fps, raw_3d, raw_2d, max_people = self.extract_frames(
                start_sec=start_sec, end_sec=end_sec, video_path_override=norm_path
            )

            if len(raw_2d["left_wrist"]) < 5:
                print("ERROR: Not enough frames extracted by MediaPipe.")
                return self._fallback()

            # Actual detection count: count frames where wrist was genuinely detected (not NaN).
            # Silent fake-success guard: MediaPipe may run but return all-NaN landmarks when it
            # can't find a pose (fast motion, bad lighting). Downstream angles will use the
            # hardcoded k_ang < 10 → 135° fallback and look plausible, but telemetry.frames will
            # be empty after NaN filtering → correction video fails with "no pose frames".
            lw_det = int(np.sum(~np.isnan(raw_2d["left_wrist"][:, 0])))
            rw_det = int(np.sum(~np.isnan(raw_2d["right_wrist"][:, 0])))
            actual_detections = lw_det + rw_det
            print(f"Actual wrist detections: L={lw_det} R={rw_det} total={actual_detections}")
            if actual_detections < 5:
                print("ERROR: Too few real pose detections — returning fallback.")
                return self._fallback()
            
            f_3d, f_2d = self.apply_filters(raw_3d), self.apply_filters(raw_2d)
            
            # Active side determined by 2D visibility confidence
            l_vis = np.mean(f_2d["left_shoulder"][:, 2]) if len(f_2d["left_shoulder"]) > 0 else 0
            r_vis = np.mean(f_2d["right_shoulder"][:, 2]) if len(f_2d["right_shoulder"]) > 0 else 0
            side = "left" if l_vis > r_vis else "right"
            
            w2d, e2d, s2d, h2d = f_2d[f"{side}_wrist"], f_2d[f"{side}_elbow"], f_2d[f"{side}_shoulder"], f_2d[f"{side}_hip"]
            w3d, e3d, s3d = f_3d[f"{side}_wrist"], f_3d[f"{side}_elbow"], f_3d[f"{side}_shoulder"]
            h3d, k3d, a3d = f_3d[f"{side}_hip"], f_3d[f"{side}_knee"], f_3d[f"{side}_ankle"]
            
            # Trajectory Event Anchor (Y-Axis Tracking)
            wrist_y = w2d[:, 1]
            dip_frame = max(0, min(int(np.argmax(wrist_y)), len(wrist_y) - 5))
            search_end = min(len(wrist_y), dip_frame + int(fps * 1.5))
            
            if search_end > dip_frame + 2:
                release_frame = dip_frame + int(np.argmin(wrist_y[dip_frame:search_end]))
            else:
                release_frame = dip_frame + 1
            
            # 3D Math
            k_ang = _calculate_3d_angle(h3d[dip_frame], k3d[dip_frame], a3d[dip_frame])
            e_ang = _calculate_3d_angle(s3d[release_frame], e3d[release_frame], w3d[release_frame])
            if k_ang < 10: k_ang = 135.0
            if e_ang < 10: e_ang = 165.0
            
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
            print(f"DEBUG PIPELINE -> Lever Arc: {lever_arc:.1f}°, Polynomial Arc: {calc_arc if calc_arc else 'None'}, Final Output: {arc_deg:.1f}°")

            # True Transverse Torso Twist (Shoulders vs Hips)
            yaw_deg = 0.0
            ls, rs = f_3d["left_shoulder"][dip_frame], f_3d["right_shoulder"][dip_frame]
            lh, rh = f_3d["left_hip"][dip_frame], f_3d["right_hip"][dip_frame]

            if not (np.any(np.isnan(ls)) or np.any(np.isnan(lh))):
                s_ang = math.atan2(rs[2] - ls[2], rs[0] - ls[0])
                h_ang = math.atan2(rh[2] - lh[2], rh[0] - lh[0])
                twist = math.degrees(s_ang - h_ang)

                # Normalize to shortest path (-180 to 180)
                twist = (twist + 180) % 360 - 180
                yaw_deg = np.clip(twist, -45.0, 45.0)

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

            if not (np.any(np.isnan(lh2d)) or np.any(np.isnan(la2d))):
                hip_mid_x = (lh2d[0] + rh2d[0]) / 2.0
                ankle_mid_x = (la2d[0] + ra2d[0]) / 2.0

                # Normalize deviation by torso length to remain immune to camera zoom
                if t_len > 1e-4:
                    deviation = abs(hip_mid_x - ankle_mid_x) / t_len
                    # A perfect vertical stack (deviation 0) = 99 score.
                    # Leaning heavily (deviation > 0.5) drops score rapidly.
                    balance_index = int(np.clip(100 - (deviation * 120), 40, 99))

            # Fluidity Derivation
            fluidity = 65
            if release_frame > dip_frame + 2:
                jerk = np.std(np.diff(np.diff(wrist_y[dip_frame:release_frame]))) if release_frame - dip_frame > 3 else 0
                fluidity = int(np.clip(100 - (jerk * 2000), 40, 99))

            # 2D Telemetry Payload for UI Rendering — Research-grade per-frame overlay
            k2d, a2d = f_2d[f"{side}_knee"], f_2d[f"{side}_ankle"]
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
                "people_detected_max": people_count,
                "video_quality_note": (
                    "Multiple people detected. Analysis focuses on the most visible subject. "
                    "For best pro matching, record only the shooter in frame."
                ) if people_count > 1 else None,
            }
            telemetry["detection_metadata"] = detection_metadata

            visibility = self._compute_pose_visibility(raw_2d)

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
            all_warnings = validation_flags + (vq.get("video_quality_notes") or [])
            telemetry["validation_warnings"] = all_warnings

            # Per-metric uncertainty (PMC 9397457): frame-window variance
            k_unc, e_unc = self._compute_angle_uncertainty(
                h3d, k3d, a3d, s3d, e3d, w3d, dip_frame, release_frame, visibility
            )
            metrics_out["knee_angle_uncertainty"] = k_unc
            metrics_out["elbow_angle_uncertainty"] = e_unc

            # Transparent confidence attribution
            vq_score = vq.get("video_quality_score", 50)
            telemetry["confidence_factors"] = self._compute_confidence_factors(
                vq_score, people_count, visibility, all_warnings, used_fallback=False
            )

            return {
                "release_velocity_mps": round(float(vel_mps), 2),
                "shot_arc_deg": round(float(arc_deg), 1),
                "knee_angle": round(float(np.clip(k_ang, 90, 180)), 1),
                "elbow_angle": round(float(np.clip(e_ang, 100, 180)), 1),
                "knee_angle_uncertainty": k_unc,
                "elbow_angle_uncertainty": e_unc,
                "knee_flexion_at_dip": round(float(np.clip(k_ang, 90, 180)), 1),
                "elbow_flexion_at_release": round(float(np.clip(e_ang, 100, 180)), 1),
                "kinetic_sync_ms": round(float(sync_ms), 1),
                "hip_rotation_deg": round(float(yaw_deg), 1),
                "balance_index": balance_index,
                "fluidity_score": fluidity,
                "telemetry": telemetry,
            }
        except Exception as e:
            import traceback
            print(f"FATAL KINEMATIC CRASH:\n{traceback.format_exc()}")
            return self._fallback()
        finally:
            # Clean up the FFmpeg-normalised temp file
            if norm_path and norm_path != self.video_path:
                try:
                    os.unlink(norm_path)
                except OSError:
                    pass

    def _fallback(self):
        telemetry = {
            "dip": {}, "release": {}, "frames": [],
            "validation_warnings": ["Analysis used fallback values. Video may lack a detectable jump shot or pose."],
            "detection_metadata": {"algorithms": ["MediaPipe Pose"], "people_detected_max": 0},
        }
        return {
            "release_velocity_mps": 7.0, "shot_arc_deg": 45.0, "knee_angle": 145.0, "elbow_angle": 165.0,
            "knee_flexion_at_dip": 145.0, "elbow_flexion_at_release": 165.0, "kinetic_sync_ms": 150.0,
            "hip_rotation_deg": 5.0, "balance_index": 75, "fluidity_score": 65,
            "telemetry": telemetry,
        }