"""
Laksh.ai Phase 1 Correction Video Engine.

Generates a side-by-side before/after visualization.

Left panel  → USER FORM:     actual video frame + skeleton overlay, joints
                              color-coded GREEN/YELLOW/RED vs personalized target.
Right panel → CORRECTED FORM: same video frame (desaturated) + cyan corrected
                              skeleton + arrows showing required joint movement.

Correction targets (priority order):
  1. Matched pro's actual values   — derived from kinematic_deltas (personalized)
  2. Sport config ideal midpoints  — population average for the sport
  3. Hardcoded fallback constants  — absolute last resort

When no video source path is provided, falls back to black-background skeleton
rendering (useful for testing, not production UX).
"""
import math
import os
import tempfile
import cv2
import numpy as np

# ── Aesthetic constants (match dashboard.html palette) ─────────────────────
BG_COLOR   = (11, 8, 7)          # #07080B in BGR — used in fallback mode only
DIVIDER    = (46, 34, 31)         # #1F222E in BGR
GREEN      = (20, 255, 57)        # #39FF14 neon green  — joint "good"
RED        = (60, 60, 255)        # #FF3C3C neon red    — joint "needs work"
YELLOW     = (0, 220, 255)        # #FFDC00 amber       — joint "acceptable"
CYAN       = (255, 229, 0)        # #00E5FF neon cyan   — corrected ideal
GREY       = (120, 120, 120)
WHITE      = (240, 240, 240)

PANEL_W, PANEL_H = 640, 540
DIVIDER_W        = 4
FRAME_W          = PANEL_W * 2 + DIVIDER_W   # 1284 px

# Skeleton opacity overlay alpha (0.0 = invisible, 1.0 = opaque)
SKEL_ALPHA       = 0.85    # left panel  — skeleton opacity over video
CORR_SKEL_ALPHA  = 0.90    # right panel — corrected skeleton opacity
VIDEO_DIM_FACTOR = 0.45    # right panel — how much to darken/desaturate the video

# ── Skeleton connectivity ───────────────────────────────────────────────────
BONES = [
    ("wrist",    "elbow"),
    ("elbow",    "shoulder"),
    ("shoulder", "hip"),
    ("hip",      "knee"),
    ("knee",     "ankle"),
]

# ── Sport config ideal range midpoints (fallback when no pro match) ─────────
_SPORT_IDEALS: dict[str, dict[str, float]] = {
    "basketball": {
        "knee_angle":      152.5,   # midpoint of 140–165
        "elbow_angle":     171.5,   # midpoint of 165–178
        "shot_arc_deg":     50.0,   # midpoint of 45–55
        "balance_index":    92.0,   # midpoint of 85–99
        "hip_rotation_deg": 10.0,   # midpoint of 5–15
    },
    "tennis": {
        "knee_angle":      145.0,
        "elbow_angle":     170.0,
        "shot_arc_deg":     45.0,
        "balance_index":    90.0,
        "hip_rotation_deg": 20.0,
    },
    "golf": {
        "knee_angle":      155.0,
        "elbow_angle":     175.0,
        "shot_arc_deg":     50.0,
        "balance_index":    91.0,
        "hip_rotation_deg": 30.0,
    },
}

# Quality acceptance windows per metric
_QUALITY_WINDOWS: dict[str, tuple[float, float]] = {
    "knee_angle":       (135.0, 170.0),
    "elbow_angle":      (155.0, 180.0),
    "shot_arc_deg":     (42.0,  58.0),
    "balance_index":    (75.0,  100.0),
    "hip_rotation_deg": (3.0,   20.0),
}


# ── Target builder — data-driven, personalized ──────────────────────────────

def _build_targets(stats: dict, kinematic_deltas: dict | None,
                   sport: str) -> dict:
    """
    Build personalized correction targets for this athlete.

    Priority:
      1. Matched pro's actual values  (user_metric + gap from kinematic_deltas)
      2. Sport config ideal range midpoints
    """
    base = _SPORT_IDEALS.get(sport, _SPORT_IDEALS["basketball"]).copy()

    if kinematic_deltas and isinstance(kinematic_deltas, dict) and "error" not in kinematic_deltas:
        mapping = {
            "knee_angle":      ("knee_angle",      "knee_gap"),
            "elbow_angle":     ("elbow_angle",      "elbow_gap"),
            "shot_arc_deg":    ("shot_arc_deg",     "arc_gap"),
            "balance_index":   ("balance_index",    "bal_gap"),
            "hip_rotation_deg":("hip_rotation_deg", "hip_gap"),
        }
        for target_key, (stat_key, delta_key) in mapping.items():
            user_val = stats.get(stat_key)
            gap_val  = kinematic_deltas.get(delta_key)
            if user_val is not None and gap_val is not None:
                base[target_key] = float(user_val) + float(gap_val)

    return base


# ── Frame-time index builder ─────────────────────────────────────────────────

def _build_frame_index(frames: list) -> list[tuple[float, dict]]:
    """
    Return sorted list of (time_sec, joints_dict) for fast nearest-time lookup.
    """
    index = []
    for f in frames:
        t = f.get("time_sec")
        j = f.get("joints")
        if t is not None and j:
            index.append((float(t), j))
    return sorted(index, key=lambda x: x[0])


def _nearest_joints(index: list, t: float) -> dict | None:
    """Binary-search for the closest telemetry frame to time t."""
    if not index:
        return None
    lo, hi = 0, len(index) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if index[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    # Check lo and lo-1
    best = lo
    if lo > 0 and abs(index[lo-1][0] - t) < abs(index[lo][0] - t):
        best = lo - 1
    return index[best][1]


# ── Video frame resize with letterbox ────────────────────────────────────────

def _resize_to_panel(frame: np.ndarray, W: int = PANEL_W,
                     H: int = PANEL_H) -> np.ndarray:
    """
    Resize a video frame to fit WxH while preserving aspect ratio.
    Letterboxed regions are filled with near-black (matches BG palette).
    """
    h, w = frame.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W, 3), (15, 12, 11), dtype=np.uint8)
    y0 = (H - nh) // 2
    x0 = (W - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas, (x0, y0, nw, nh)   # type: ignore[return-value]


# ── Joint → pixel mapping (accounts for letterbox offsets) ──────────────────

def _joints_to_px(joints_norm: dict, x0: int, y0: int,
                  nw: int, nh: int) -> dict:
    """
    Map normalised [0,1] joint coords to pixel coords within the letterboxed
    video region. Joints are placed relative to the actual video content area,
    not the full panel — so skeleton aligns precisely with the athlete's body.
    """
    result = {}
    for k, v in joints_norm.items():
        px = int(max(0, min(PANEL_W - 1,  x0 + v[0] * nw)))
        py = int(max(0, min(PANEL_H - 1,  y0 + v[1] * nh)))
        result[k] = (px, py)
    return result


# ── Correction geometry ──────────────────────────────────────────────────────

def _correct_joints_px(joints_px: dict, joints_norm: dict, stats: dict,
                        targets: dict, frame_t: float,
                        dip_t: float, release_t: float,
                        x0: int, y0: int, nw: int, nh: int) -> dict:
    """
    Compute corrected pixel positions by applying biomechanical corrections
    in normalised space, then mapping back to pixel coords.
    Corrections are proportional to the delta vs. personalized target,
    so a user already at their target sees zero shift.
    """
    c = {k: list(v) for k, v in joints_norm.items()}
    knee_angle  = stats.get("knee_angle",  targets["knee_angle"])
    elbow_angle = stats.get("elbow_angle", targets["elbow_angle"])
    balance_idx = stats.get("balance_index", targets["balance_index"])

    # Dip phase: knee correction
    if abs(frame_t - dip_t) < 0.18 and "knee" in c and "hip" in c and "ankle" in c:
        deficit = targets["knee_angle"] - knee_angle
        if abs(deficit) > 5:
            torso_h = abs(c["hip"][1] - c["ankle"][1]) or 0.18
            c["knee"][1] = max(0.02, min(0.98, c["knee"][1] - (deficit / 90.0) * torso_h * 0.35))
            c["knee"][0] = max(0.02, min(0.98, c["knee"][0] + (deficit / 90.0) * torso_h * 0.15))

    # Release phase: elbow extension
    if abs(frame_t - release_t) < 0.18 and "wrist" in c and "elbow" in c:
        deficit = targets["elbow_angle"] - elbow_angle
        if deficit > 5:
            dx = c["wrist"][0] - c["elbow"][0]
            dy = c["wrist"][1] - c["elbow"][1]
            dist = math.sqrt(dx**2 + dy**2) or 1e-4
            scale = 1.0 + (deficit / 50.0) * 0.28
            c["wrist"][0] = max(0.02, min(0.98, c["elbow"][0] + dx * scale))
            c["wrist"][1] = max(0.02, min(0.98, c["elbow"][1] + dy * scale))

    # Balance: hip over ankle
    if balance_idx < targets["balance_index"] - 5 and "hip" in c and "ankle" in c:
        deficit = targets["balance_index"] - balance_idx
        shift_x = (c["ankle"][0] - c["hip"][0]) * (deficit / 100.0) * 0.4
        c["hip"][0] = max(0.05, min(0.95, c["hip"][0] + shift_x))

    return _joints_to_px({k: tuple(v) for k, v in c.items()}, x0, y0, nw, nh)


# ── Joint quality color ───────────────────────────────────────────────────────

def _joint_quality_color(name: str, stats: dict, targets: dict,
                          frame_t: float, dip_t: float,
                          release_t: float) -> tuple:
    at_dip     = abs(frame_t - dip_t)     < 0.18
    at_release = abs(frame_t - release_t) < 0.18

    if name == "knee" and at_dip:
        val = stats.get("knee_angle")
        if val is None:
            return GREEN
        lo, hi = _QUALITY_WINDOWS["knee_angle"]
        return GREEN if lo <= val <= hi else (YELLOW if abs(val - targets["knee_angle"]) < 20 else RED)

    if name in ("elbow", "wrist") and at_release:
        val = stats.get("elbow_angle")
        if val is None:
            return GREEN
        lo, hi = _QUALITY_WINDOWS["elbow_angle"]
        return GREEN if lo <= val <= hi else (YELLOW if abs(val - targets["elbow_angle"]) < 15 else RED)

    return GREEN


# ── Drawing helpers ──────────────────────────────────────────────────────────

def _draw_skeleton(img: np.ndarray, joints_px: dict, default_color: tuple,
                   highlight: dict | None = None, thickness: int = 4,
                   radius: int = 8, alpha: float = 1.0) -> None:
    if alpha < 1.0:
        overlay = img.copy()
        _draw_skeleton(overlay, joints_px, default_color, highlight, thickness, radius, 1.0)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
        return
    for a_name, b_name in BONES:
        a = joints_px.get(a_name)
        b = joints_px.get(b_name)
        if a and b:
            cv2.line(img, a, b, default_color, thickness, cv2.LINE_AA)
    for name, pt in joints_px.items():
        color = (highlight or {}).get(name, default_color)
        cv2.circle(img, pt, radius,     color,    -1, cv2.LINE_AA)
        cv2.circle(img, pt, radius + 2, (0,0,0),   2, cv2.LINE_AA)


def _label(img: np.ndarray, text: str, pos: tuple, color: tuple = WHITE,
           scale: float = 0.55, thickness: int = 1) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    # Drop shadow for readability over video
    cv2.putText(img, text, (pos[0]+1, pos[1]+1), font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def _phase_label(frame_t: float, dip_t: float, release_t: float) -> str:
    if frame_t < dip_t - 0.05:         return "SETUP"
    if frame_t < release_t - 0.05:     return "DRIVE"
    if abs(frame_t - release_t) < 0.1: return "RELEASE"
    return "FOLLOW-THROUGH"


def _draw_hud_bar(img: np.ndarray, title: str, subtitle: str,
                  title_color: tuple) -> None:
    """Semi-transparent bar at the top of a panel with title + subtitle."""
    bar_h = 62
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (PANEL_W, bar_h), (8, 6, 5), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    _label(img, title,    (14, 26), title_color, scale=0.65, thickness=2)
    _label(img, subtitle, (14, 50), GREY,        scale=0.38, thickness=1)


def _draw_metric_badge(img: np.ndarray, text: str, color: tuple,
                       y_offset: int = 90) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
    pad = 8
    x0, y0 = 12, y_offset - th - pad
    x1, y1 = 12 + tw + pad * 2, y_offset + pad
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    cv2.addWeighted(overlay, 0.20, img, 0.80, 0, img)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
    _label(img, text, (x0 + pad, y0 + th + pad - 2), color, scale=0.48, thickness=1)


def _draw_correction_arrows(img: np.ndarray, actual_px: dict,
                             corrected_px: dict) -> None:
    for name in ["knee", "wrist", "elbow"]:
        a = actual_px.get(name)
        c = corrected_px.get(name)
        if a and c:
            dist = math.sqrt((a[0]-c[0])**2 + (a[1]-c[1])**2)
            if dist > 8:
                cv2.arrowedLine(img, a, c, CYAN, 2, cv2.LINE_AA, tipLength=0.28)


def _make_right_bg(frame: np.ndarray) -> np.ndarray:
    """
    Desaturate + dim the video frame for the right panel so the cyan
    corrected skeleton stands out clearly against the real body.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey_bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(grey_bgr, VIDEO_DIM_FACTOR,
                               np.zeros_like(grey_bgr), 1 - VIDEO_DIM_FACTOR, 0)
    return blended


# ── Main generation function ─────────────────────────────────────────────────

def generate_correction_video(
    telemetry: dict,
    stats: dict,
    athlete_name: str = "Athlete",
    kinematic_deltas: dict | None = None,
    sport: str = "basketball",
    pro_match: str | None = None,
    video_path: str | None = None,
) -> bytes | None:
    """
    Build a side-by-side before/after correction MP4.

    When `video_path` is provided (production path):
      Each panel shows the actual user video frame with skeleton overlaid.
      Left  → real video + color-coded skeleton on the user's actual body
      Right → desaturated real video + cyan corrected skeleton + arrows

    When `video_path` is None (fallback / testing):
      Renders skeleton-only on black background.

    Parameters
    ----------
    telemetry        : dict   — `telemetry` block from /analyze-video response
    stats            : dict   — 8D metrics block
    athlete_name     : str    — shown in left-panel HUD
    kinematic_deltas : dict   — gaps vs. matched pro (enables personalized targets)
    sport            : str    — sport ID for ideal range lookup
    pro_match        : str    — matched pro name for right-panel subtitle
    video_path       : str    — path to the uploaded video file on disk
    """
    frames    = telemetry.get("frames") or []
    dip       = telemetry.get("dip") or {}
    release   = telemetry.get("release") or {}
    fps_tel   = float(telemetry.get("fps") or 30.0)
    dip_t     = float(dip.get("time_sec") or 0.0)
    release_t = float(release.get("time_sec") or (dip_t + 0.4))

    if len(frames) < 4:
        return None

    targets  = _build_targets(stats, kinematic_deltas, sport)
    tel_idx  = _build_frame_index(frames)

    knee_angle  = stats.get("knee_angle",  targets["knee_angle"])
    elbow_angle = stats.get("elbow_angle", targets["elbow_angle"])
    arc_deg     = stats.get("shot_arc_deg", targets["shot_arc_deg"])

    if kinematic_deltas and "error" not in (kinematic_deltas or {}):
        source_label = f"Target: {pro_match or 'Matched Pro'} mechanics"
    else:
        source_label = f"Target: {sport.capitalize()} ideal range"

    corrections = []
    if abs(knee_angle  - targets["knee_angle"])  > 5:
        corrections.append(f"Knee {knee_angle:.0f}\xb0\u2192{targets['knee_angle']:.0f}\xb0")
    if abs(elbow_angle - targets["elbow_angle"]) > 5:
        corrections.append(f"Elbow {elbow_angle:.0f}\xb0\u2192{targets['elbow_angle']:.0f}\xb0")
    if abs(arc_deg     - targets["shot_arc_deg"]) > 3:
        corrections.append(f"Arc {arc_deg:.0f}\xb0\u2192{targets['shot_arc_deg']:.0f}\xb0")
    subtitle_r = (
        ("  \u2022  ".join(corrections) + f"  |  {source_label}")
        if corrections else f"Mechanics within optimal range  |  {source_label}"
    )

    # ── Open source video if provided ───────────────────────────────────────
    cap = None
    video_fps = fps_tel
    total_video_frames = None
    use_video = False

    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            video_fps        = cap.get(cv2.CAP_PROP_FPS) or fps_tel
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            use_video = True
        else:
            cap = None

    # Output fps matches source video (or telemetry if no video)
    out_fps = video_fps

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=tempfile.gettempdir())
    os.close(tmp_fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, out_fps, (FRAME_W, PANEL_H))

    if use_video:
        # ── Video-backed rendering ───────────────────────────────────────────
        frame_idx = 0
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            t = frame_idx / video_fps
            frame_idx += 1

            joints_norm = _nearest_joints(tel_idx, t)
            if joints_norm is None:
                continue

            # Resize video frame to panel with letterboxing
            left_bg, (x0, y0, nw, nh) = _resize_to_panel(bgr_frame)
            right_bg                   = _make_right_bg(left_bg.copy())

            joints_px = _joints_to_px(joints_norm, x0, y0, nw, nh)
            corr_px   = _correct_joints_px(
                joints_px, joints_norm, stats, targets,
                t, dip_t, release_t, x0, y0, nw, nh
            )

            highlight = {
                name: _joint_quality_color(name, stats, targets, t, dip_t, release_t)
                for name in joints_px
            }

            # Left: actual video + color-coded skeleton
            _draw_skeleton(left_bg, joints_px, GREEN, highlight=highlight,
                           alpha=SKEL_ALPHA)

            # Right: grey ghost (actual pos) + cyan corrected skeleton + arrows
            _draw_skeleton(right_bg, joints_px, (70, 70, 70), thickness=2,
                           radius=5, alpha=0.6)
            _draw_skeleton(right_bg, corr_px, CYAN,
                           alpha=CORR_SKEL_ALPHA)
            _draw_correction_arrows(right_bg, joints_px, corr_px)

        # Phase + HUD overlays
            phase = _phase_label(t, dip_t, release_t)
            _draw_hud_bar(left_bg,
                          f"{athlete_name.upper()} — YOUR FORM",
                          "Green=good  Yellow=borderline  Red=needs work",
                          GREEN)
            _label(left_bg, phase, (PANEL_W - 155, 30), CYAN, scale=0.52)

            _draw_hud_bar(right_bg, "CORRECTED FORM", subtitle_r, CYAN)
            _label(right_bg, phase, (PANEL_W - 155, 30), GREEN, scale=0.52)

            # Metric badges at key frames
            if abs(t - dip_t) < 0.18:
                lo, hi = _QUALITY_WINDOWS["knee_angle"]
                kc = GREEN if lo <= knee_angle <= hi else RED
                _draw_metric_badge(left_bg,  f"KNEE  {knee_angle:.0f}\xb0", kc, PANEL_H - 55)
                _draw_metric_badge(right_bg, f"KNEE  {targets['knee_angle']:.0f}\xb0  (target)",
                                   CYAN, PANEL_H - 55)
            if abs(t - release_t) < 0.18:
                lo, hi = _QUALITY_WINDOWS["elbow_angle"]
                ec = GREEN if lo <= elbow_angle <= hi else RED
                _draw_metric_badge(left_bg,  f"ELBOW  {elbow_angle:.0f}\xb0", ec, PANEL_H - 55)
                _draw_metric_badge(right_bg, f"ELBOW  {targets['elbow_angle']:.0f}\xb0  (target)",
                                   CYAN, PANEL_H - 55)

            div = np.full((PANEL_H, DIVIDER_W, 3), DIVIDER, dtype=np.uint8)
            writer.write(np.hstack([left_bg, div, right_bg]))

        cap.release()

    else:
        # ── Skeleton-only fallback (no video source) ─────────────────────────
        for frame_data in frames:
            frame_t     = float(frame_data.get("time_sec") or 0.0)
            joints_norm = frame_data.get("joints") or {}
            if not joints_norm:
                continue

            left  = np.full((PANEL_H, PANEL_W, 3), BG_COLOR, dtype=np.uint8)
            right = np.full((PANEL_H, PANEL_W, 3), BG_COLOR, dtype=np.uint8)

            # In fallback mode joints are full-panel (no letterbox offset)
            joints_px = {
                k: (int(max(0, min(PANEL_W-1, v[0]*PANEL_W))),
                    int(max(0, min(PANEL_H-1, v[1]*PANEL_H))))
                for k, v in joints_norm.items()
            }
            corr_norm = {}
            c = {k: list(v) for k, v in joints_norm.items()}
            ke = stats.get("knee_angle",  targets["knee_angle"])
            el = stats.get("elbow_angle", targets["elbow_angle"])
            bi = stats.get("balance_index", targets["balance_index"])
            if abs(frame_t - dip_t) < 0.18 and "knee" in c and "hip" in c and "ankle" in c:
                deficit = targets["knee_angle"] - ke
                if abs(deficit) > 5:
                    th_ = abs(c["hip"][1] - c["ankle"][1]) or 0.18
                    c["knee"][1] = max(0.02, min(0.98, c["knee"][1] - (deficit/90)*th_*0.35))
                    c["knee"][0] = max(0.02, min(0.98, c["knee"][0] + (deficit/90)*th_*0.15))
            if abs(frame_t - release_t) < 0.18 and "wrist" in c and "elbow" in c:
                deficit = targets["elbow_angle"] - el
                if deficit > 5:
                    dx_ = c["wrist"][0]-c["elbow"][0]; dy_ = c["wrist"][1]-c["elbow"][1]
                    dist_ = math.sqrt(dx_**2+dy_**2) or 1e-4
                    sc_ = 1.0+(deficit/50)*0.28
                    c["wrist"][0] = max(0.02, min(0.98, c["elbow"][0]+dx_*sc_))
                    c["wrist"][1] = max(0.02, min(0.98, c["elbow"][1]+dy_*sc_))
            if bi < targets["balance_index"]-5 and "hip" in c and "ankle" in c:
                deficit = targets["balance_index"] - bi
                shift_x = (c["ankle"][0]-c["hip"][0])*(deficit/100)*0.4
                c["hip"][0] = max(0.05, min(0.95, c["hip"][0]+shift_x))
            corr_px = {
                k: (int(max(0, min(PANEL_W-1, v[0]*PANEL_W))),
                    int(max(0, min(PANEL_H-1, v[1]*PANEL_H))))
                for k, v in {kk: tuple(vv) for kk, vv in c.items()}.items()
            }

            highlight = {
                name: _joint_quality_color(name, stats, targets, frame_t, dip_t, release_t)
                for name in joints_px
            }

            phase = _phase_label(frame_t, dip_t, release_t)
            _draw_hud_bar(left,
                          f"{athlete_name.upper()} — YOUR FORM",
                          "Skeleton preview (upload video for full overlay)",
                          GREEN)
            _draw_hud_bar(right, "CORRECTED FORM", subtitle_r, CYAN)
            _label(left,  phase, (PANEL_W-155, 30), CYAN,  scale=0.52)
            _label(right, phase, (PANEL_W-155, 30), GREEN, scale=0.52)

            _draw_skeleton(left, joints_px, GREEN, highlight=highlight)
            _draw_skeleton(right, joints_px, (55,55,55), thickness=2, radius=5)
            _draw_skeleton(right, corr_px, CYAN)
            _draw_correction_arrows(right, joints_px, corr_px)

            if abs(frame_t - dip_t) < 0.18:
                lo, hi = _QUALITY_WINDOWS["knee_angle"]
                ke_val = stats.get("knee_angle", targets["knee_angle"])
                kc = GREEN if lo <= ke_val <= hi else RED
                _draw_metric_badge(left,  f"KNEE  {ke_val:.0f}\xb0", kc, PANEL_H-55)
                _draw_metric_badge(right, f"KNEE  {targets['knee_angle']:.0f}\xb0  (target)", CYAN, PANEL_H-55)
            if abs(frame_t - release_t) < 0.18:
                lo, hi = _QUALITY_WINDOWS["elbow_angle"]
                el_val = stats.get("elbow_angle", targets["elbow_angle"])
                ec = GREEN if lo <= el_val <= hi else RED
                _draw_metric_badge(left,  f"ELBOW  {el_val:.0f}\xb0", ec, PANEL_H-55)
                _draw_metric_badge(right, f"ELBOW  {targets['elbow_angle']:.0f}\xb0  (target)", CYAN, PANEL_H-55)

            div = np.full((PANEL_H, DIVIDER_W, 3), DIVIDER, dtype=np.uint8)
            writer.write(np.hstack([left, div, right]))

    writer.release()

    with open(tmp_path, "rb") as fh:
        video_bytes = fh.read()
    os.unlink(tmp_path)

    return video_bytes if len(video_bytes) > 1024 else None
