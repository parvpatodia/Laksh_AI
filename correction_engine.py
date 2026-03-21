"""
Laksh.ai Phase 1 Correction Video Engine.

Generates a focused, side-by-side before/after correction video covering
exactly the shot window (dip → release + 1.5s) overlaid on the actual
user video. When the video file is unavailable, falls back to skeleton-only
rendering on a dark background.

Left panel  → YOUR FORM:      actual video frame + shooting-side skeleton,
                               joints color-coded GREEN/YELLOW/RED vs target.
Right panel → CORRECTED FORM: same frame desaturated + grey ghost skeleton +
                               cyan corrected skeleton + directional arrows.

Correction targets (priority order):
  1. Matched pro's actual values  — from kinematic_deltas (personalized)
  2. Sport config ideal midpoints — population average for the sport
"""
import math
import os
import tempfile
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Aesthetic constants (match dashboard.html palette) ─────────────────────
BG_COLOR  = (11, 8, 7)           # #07080B — used in skeleton-only fallback
DIVIDER   = (46, 34, 31)
GREEN     = (20, 255, 57)         # #39FF14 neon green — good joint
RED       = (60, 60, 255)         # #FF3C3C neon red   — needs work
YELLOW    = (0, 220, 255)         # #FFDC00 amber      — borderline
CYAN      = (255, 229, 0)         # #00E5FF neon cyan  — corrected target
GREY      = (120, 120, 120)
WHITE     = (240, 240, 240)

PANEL_W, PANEL_H  = 640, 540
DIVIDER_W         = 4
FRAME_W           = PANEL_W * 2 + DIVIDER_W   # 1284 px

SKEL_ALPHA        = 0.88   # skeleton opacity over video (left panel)
CORR_SKEL_ALPHA   = 0.92   # corrected skeleton opacity (right panel)
VIDEO_DIM         = 0.40   # desaturation level for right panel bg

# ── Skeleton connectivity (shooting-side chain) ─────────────────────────────
BONES = [
    ("wrist",    "elbow"),
    ("elbow",    "shoulder"),
    ("shoulder", "hip"),
    ("hip",      "knee"),
    ("knee",     "ankle"),
]

# ── Sport-specific ideal targets (fallback when no pro match) ────────────────
_SPORT_IDEALS: dict[str, dict[str, float]] = {
    "basketball": {
        "knee_angle":       152.5,   # midpoint 140–165
        "elbow_angle":      171.5,   # midpoint 165–178
        "shot_arc_deg":      50.0,   # midpoint 45–55
        "balance_index":     92.0,   # midpoint 85–99
        "hip_rotation_deg":  10.0,   # midpoint 5–15
    },
    "tennis": {
        "knee_angle":       145.0,
        "elbow_angle":      170.0,
        "shot_arc_deg":      45.0,
        "balance_index":     90.0,
        "hip_rotation_deg":  20.0,
    },
    "golf": {
        "knee_angle":       155.0,
        "elbow_angle":      175.0,
        "shot_arc_deg":      50.0,
        "balance_index":     91.0,
        "hip_rotation_deg":  30.0,
    },
}

_QUALITY_WINDOWS: dict[str, tuple[float, float]] = {
    "knee_angle":       (135.0, 170.0),
    "elbow_angle":      (155.0, 180.0),
    "shot_arc_deg":     (42.0,  58.0),
    "balance_index":    (75.0,  100.0),
    "hip_rotation_deg": (3.0,   20.0),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _has_nan(joints: dict) -> bool:
    for coords in joints.values():
        if any(isinstance(c, float) and math.isnan(c) for c in (coords or [])):
            return True
    return False


def _build_targets(stats: dict, kinematic_deltas: dict | None,
                   sport: str) -> dict:
    """
    Personalized correction targets (priority: pro match → sport ideal).
    kinematic_deltas format: {knee_gap, elbow_gap, arc_gap, bal_gap, hip_gap}
    where gap = pro_value − user_value.
    """
    base = _SPORT_IDEALS.get(sport, _SPORT_IDEALS["basketball"]).copy()
    if kinematic_deltas and isinstance(kinematic_deltas, dict) and "error" not in kinematic_deltas:
        mapping = [
            ("knee_angle",      "knee_angle",       "knee_gap"),
            ("elbow_angle",     "elbow_angle",      "elbow_gap"),
            ("shot_arc_deg",    "shot_arc_deg",     "arc_gap"),
            ("balance_index",   "balance_index",    "bal_gap"),
            ("hip_rotation_deg","hip_rotation_deg", "hip_gap"),
        ]
        for tgt_key, stat_key, delta_key in mapping:
            user_val = stats.get(stat_key)
            gap      = kinematic_deltas.get(delta_key)
            if user_val is not None and gap is not None:
                try:
                    base[tgt_key] = float(user_val) + float(gap)
                except (TypeError, ValueError):
                    pass
    return base


def _build_frame_index(frames: list) -> list[tuple[float, dict]]:
    """Sorted (time_sec, joints) pairs with NaN-containing frames removed."""
    index = []
    for f in frames:
        t = f.get("time_sec")
        j = f.get("joints") or {}
        if t is None or not j or _has_nan(j):
            continue
        index.append((float(t), j))
    return sorted(index, key=lambda x: x[0])


def _build_anchor_frames(dip: dict, release: dict,
                          fps: float) -> list[dict]:
    """
    Construct synthetic telemetry frames by interpolating between the dip
    and release key-frame joints when the main frames array is sparse/empty.
    Returns an empty list if anchor data is missing or NaN-contaminated.
    """
    dip_j   = dip.get("joints") or {}
    rel_j   = release.get("joints") or {}
    dip_t   = float(dip.get("time_sec") or 0.0)
    rel_t   = float(release.get("time_sec") or (dip_t + 0.4))

    if not dip_j or not rel_j:
        return []
    if _has_nan(dip_j) or _has_nan(rel_j):
        return []

    keys    = set(dip_j) & set(rel_j)
    if not keys:
        return []

    dt      = max(rel_t - dip_t, 1.0 / fps)
    # Timeline: setup → dip → lerp → release → follow-through
    timeline = [
        dip_t - 0.40,
        dip_t - 0.20,
        dip_t,
        dip_t + dt * 0.33,
        dip_t + dt * 0.67,
        rel_t,
        rel_t + 0.25,
        rel_t + 0.55,
        rel_t + 0.90,
        rel_t + 1.30,
        rel_t + 1.50,
    ]

    out = []
    for t in timeline:
        alpha = max(0.0, min(1.0, (t - dip_t) / dt))
        joints = {}
        for k in keys:
            d = dip_j[k]
            r = rel_j[k]
            joints[k] = [d[i] + (r[i] - d[i]) * alpha for i in range(min(len(d), len(r)))]
        out.append({"time_sec": round(t, 3), "joints": joints, "_synthetic": True})
    return out


def _nearest_joints(index: list, t: float) -> dict | None:
    """Binary-search the frame index for the closest time to t."""
    if not index:
        return None
    lo, hi = 0, len(index) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if index[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    best = lo
    if lo > 0 and abs(index[lo - 1][0] - t) < abs(index[lo][0] - t):
        best = lo - 1
    return index[best][1]


# ── Video-frame helpers ───────────────────────────────────────────────────────

def _resize_to_panel(frame: np.ndarray,
                     W: int = PANEL_W,
                     H: int = PANEL_H) -> tuple[np.ndarray, tuple]:
    """
    Letterbox `frame` to fit W×H while preserving aspect ratio.
    Returns (canvas, (x0, y0, content_w, content_h)).
    """
    h, w = frame.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas  = np.full((H, W, 3), (15, 12, 11), dtype=np.uint8)
    y0 = (H - nh) // 2
    x0 = (W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, (x0, y0, nw, nh)


def _make_right_bg(frame: np.ndarray) -> np.ndarray:
    """Desaturate + dim the video frame for the right panel."""
    grey     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey_bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(grey_bgr, VIDEO_DIM,
                           np.zeros_like(grey_bgr), 1.0 - VIDEO_DIM, 0)


def _joints_to_px(joints_norm: dict,
                  x0: int, y0: int, nw: int, nh: int) -> dict:
    """
    Map normalised [0,1] joint coords into the letterboxed video content area.
    Clamps to valid pixel range. Skips joints with invalid (non-finite) coords.
    """
    result = {}
    for k, v in joints_norm.items():
        try:
            px = int(np.clip(x0 + float(v[0]) * nw, 0, PANEL_W - 1))
            py = int(np.clip(y0 + float(v[1]) * nh, 0, PANEL_H - 1))
            if math.isfinite(px) and math.isfinite(py):
                result[k] = (px, py)
        except (TypeError, ValueError, IndexError):
            continue
    return result


def _norm_to_px_fallback(joints_norm: dict) -> dict:
    """Map normalised coords to full-panel pixels (skeleton-only fallback mode)."""
    result = {}
    for k, v in joints_norm.items():
        try:
            px = int(np.clip(float(v[0]) * PANEL_W, 0, PANEL_W - 1))
            py = int(np.clip(float(v[1]) * PANEL_H, 0, PANEL_H - 1))
            if math.isfinite(px) and math.isfinite(py):
                result[k] = (px, py)
        except (TypeError, ValueError, IndexError):
            continue
    return result


# ── Correction geometry ───────────────────────────────────────────────────────

def _correct_joints(joints_norm: dict, stats: dict, targets: dict,
                    frame_t: float, dip_t: float, release_t: float) -> dict:
    """
    Apply biomechanical corrections in normalised space.
    Correction magnitude is proportional to the delta vs. personalized target —
    a joint already at target sees zero shift.
    """
    c            = {k: list(v) for k, v in joints_norm.items()}
    knee_angle   = stats.get("knee_angle",   targets["knee_angle"])
    elbow_angle  = stats.get("elbow_angle",  targets["elbow_angle"])
    balance_idx  = stats.get("balance_index",targets["balance_index"])

    # Dip: knee flexion correction
    if abs(frame_t - dip_t) < 0.22 and all(k in c for k in ("knee", "hip", "ankle")):
        deficit = targets["knee_angle"] - knee_angle
        if abs(deficit) > 5:
            torso_h = abs(c["hip"][1] - c["ankle"][1]) or 0.18
            c["knee"][1] = float(np.clip(c["knee"][1] - (deficit / 90.0) * torso_h * 0.35, 0.02, 0.98))
            c["knee"][0] = float(np.clip(c["knee"][0] + (deficit / 90.0) * torso_h * 0.15, 0.02, 0.98))

    # Release: elbow extension
    if abs(frame_t - release_t) < 0.22 and all(k in c for k in ("wrist", "elbow")):
        deficit = targets["elbow_angle"] - elbow_angle
        if deficit > 5:
            dx = c["wrist"][0] - c["elbow"][0]
            dy = c["wrist"][1] - c["elbow"][1]
            dist = math.sqrt(dx**2 + dy**2) or 1e-4
            scale = 1.0 + (deficit / 50.0) * 0.28
            c["wrist"][0] = float(np.clip(c["elbow"][0] + dx * scale, 0.02, 0.98))
            c["wrist"][1] = float(np.clip(c["elbow"][1] + dy * scale, 0.02, 0.98))

    # Balance: hip centering over ankle
    if balance_idx < targets["balance_index"] - 5 and all(k in c for k in ("hip", "ankle")):
        deficit  = targets["balance_index"] - balance_idx
        shift_x  = (c["ankle"][0] - c["hip"][0]) * (deficit / 100.0) * 0.4
        c["hip"][0] = float(np.clip(c["hip"][0] + shift_x, 0.05, 0.95))

    return {k: tuple(v) for k, v in c.items()}


def _joint_quality_color(name: str, stats: dict, targets: dict,
                          frame_t: float, dip_t: float,
                          release_t: float) -> tuple:
    at_dip     = abs(frame_t - dip_t)     < 0.22
    at_release = abs(frame_t - release_t) < 0.22

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


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_skeleton(img: np.ndarray, joints_px: dict, default_color: tuple,
                   highlight: dict | None = None,
                   thickness: int = 4, radius: int = 9,
                   alpha: float = 1.0) -> None:
    """
    Draw MediaPipe-style skeleton: bones first, then joints on top.
    Joints get a subtle black outline ring for contrast over video.
    """
    if alpha < 1.0:
        overlay = img.copy()
        _draw_skeleton(overlay, joints_px, default_color, highlight,
                       thickness, radius, 1.0)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
        return

    # Bones
    for a_name, b_name in BONES:
        a = joints_px.get(a_name)
        b = joints_px.get(b_name)
        if a and b:
            cv2.line(img, a, b, default_color, thickness, cv2.LINE_AA)

    # Joints — outer black ring for contrast, inner coloured fill
    for name, pt in joints_px.items():
        color = (highlight or {}).get(name, default_color)
        cv2.circle(img, pt, radius + 3, (0, 0, 0), -1, cv2.LINE_AA)  # shadow
        cv2.circle(img, pt, radius,     color,      -1, cv2.LINE_AA)  # fill
        cv2.circle(img, pt, radius,     (255,255,255), 1, cv2.LINE_AA)  # rim


def _label(img: np.ndarray, text: str, pos: tuple,
           color: tuple = WHITE, scale: float = 0.55,
           thickness: int = 1) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    # Drop shadow for video readability
    cv2.putText(img, text, (pos[0]+1, pos[1]+1), font, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def _phase_label(frame_t: float, dip_t: float, release_t: float) -> str:
    if frame_t < dip_t - 0.05:          return "SETUP"
    if frame_t < release_t - 0.05:      return "DRIVE"
    if abs(frame_t - release_t) < 0.12: return "RELEASE"
    return "FOLLOW-THROUGH"


def _draw_hud_bar(img: np.ndarray, title: str, subtitle: str,
                  title_color: tuple) -> None:
    """Semi-transparent top HUD bar with title + subtitle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (PANEL_W, 66), (8, 6, 5), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    _label(img, title,    (14, 26), title_color, scale=0.62, thickness=2)
    _label(img, subtitle, (14, 52), GREY,        scale=0.36, thickness=1)


def _draw_metric_badge(img: np.ndarray, text: str,
                       color: tuple, y_offset: int) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.46, 1)
    pad = 8
    x0, y0 = 12, y_offset - th - pad
    x1, y1 = 12 + tw + pad * 2, y_offset + pad
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
    _label(img, text, (x0 + pad, y0 + th + pad - 2),
           color, scale=0.46, thickness=1)


def _draw_correction_arrows(img: np.ndarray,
                             actual_px: dict, corrected_px: dict) -> None:
    """Draw directional arrows from actual → corrected joint position."""
    for name in ("knee", "wrist", "elbow"):
        a = actual_px.get(name)
        c = corrected_px.get(name)
        if not a or not c:
            continue
        dist = math.sqrt((a[0]-c[0])**2 + (a[1]-c[1])**2)
        if dist > 8:
            # Thicker arrow with glow
            cv2.arrowedLine(img, a, c, (0,0,0), 4, cv2.LINE_AA, tipLength=0.30)
            cv2.arrowedLine(img, a, c, CYAN,    2, cv2.LINE_AA, tipLength=0.30)


def _render_frame_pair(
    left_bg:     np.ndarray,
    right_bg:    np.ndarray,
    joints_norm: dict,
    stats:       dict,
    targets:     dict,
    frame_t:     float,
    dip_t:       float,
    release_t:   float,
    athlete_name:str,
    subtitle_r:  str,
    knee_angle:  float,
    elbow_angle: float,
    is_synthetic:bool,
    letterbox:   tuple,        # (x0, y0, nw, nh)  or None for skeleton-only
) -> np.ndarray:
    """
    Build one output frame (left | divider | right) from a pair of panel images.
    Modifies left_bg and right_bg in place.
    """
    x0, y0, nw, nh = letterbox if letterbox else (0, 0, PANEL_W, PANEL_H)

    # Joint pixels
    if letterbox:
        joints_px = _joints_to_px(joints_norm, x0, y0, nw, nh)
        corr_norm = _correct_joints(joints_norm, stats, targets,
                                    frame_t, dip_t, release_t)
        corr_px   = _joints_to_px(corr_norm, x0, y0, nw, nh)
    else:
        joints_px = _norm_to_px_fallback(joints_norm)
        corr_norm = _correct_joints(joints_norm, stats, targets,
                                    frame_t, dip_t, release_t)
        corr_px   = _norm_to_px_fallback(corr_norm)

    highlight = {
        name: _joint_quality_color(name, stats, targets, frame_t, dip_t, release_t)
        for name in joints_px
    }

    phase = _phase_label(frame_t, dip_t, release_t)

    # ── LEFT PANEL ──────────────────────────────────────────────────────────
    synth_note = " (est.)" if is_synthetic else ""
    _draw_skeleton(left_bg, joints_px, GREEN, highlight=highlight,
                   alpha=SKEL_ALPHA)
    _draw_hud_bar(left_bg,
                  f"{athlete_name.upper()} — YOUR FORM{synth_note}",
                  "Green=good  Yellow=borderline  Red=needs work",
                  GREEN)
    _label(left_bg, phase, (PANEL_W - 160, 30), CYAN, scale=0.50)

    if abs(frame_t - dip_t) < 0.22:
        lo, hi = _QUALITY_WINDOWS["knee_angle"]
        kc = GREEN if lo <= knee_angle <= hi else RED
        _draw_metric_badge(left_bg, f"KNEE  {knee_angle:.0f}\xb0", kc, PANEL_H - 55)

    if abs(frame_t - release_t) < 0.22:
        lo, hi = _QUALITY_WINDOWS["elbow_angle"]
        ec = GREEN if lo <= elbow_angle <= hi else RED
        _draw_metric_badge(left_bg, f"ELBOW  {elbow_angle:.0f}\xb0", ec, PANEL_H - 55)

    # ── RIGHT PANEL ─────────────────────────────────────────────────────────
    _draw_skeleton(right_bg, joints_px, (65, 65, 65), thickness=2,
                   radius=5, alpha=0.65)               # grey ghost
    _draw_skeleton(right_bg, corr_px, CYAN,
                   alpha=CORR_SKEL_ALPHA)               # corrected in cyan
    _draw_correction_arrows(right_bg, joints_px, corr_px)
    _draw_hud_bar(right_bg, "CORRECTED FORM", subtitle_r, CYAN)
    _label(right_bg, phase, (PANEL_W - 160, 30), GREEN, scale=0.50)

    if abs(frame_t - dip_t) < 0.22:
        _draw_metric_badge(right_bg,
                           f"KNEE  {targets['knee_angle']:.0f}\xb0  (target)",
                           CYAN, PANEL_H - 55)
    if abs(frame_t - release_t) < 0.22:
        _draw_metric_badge(right_bg,
                           f"ELBOW  {targets['elbow_angle']:.0f}\xb0  (target)",
                           CYAN, PANEL_H - 55)

    div = np.full((PANEL_H, DIVIDER_W, 3), DIVIDER, dtype=np.uint8)
    return np.hstack([left_bg, div, right_bg])


# ── Main generation function ──────────────────────────────────────────────────

def generate_correction_video(
    telemetry:          dict,
    stats:              dict,
    athlete_name:       str          = "Athlete",
    kinematic_deltas:   dict | None  = None,
    sport:              str          = "basketball",
    pro_match:          str | None   = None,
    video_path:         str | None   = None,
    clip_start_sec:     float        = 0.0,
) -> bytes | None:
    """
    Build a side-by-side before/after correction MP4.

    Parameters
    ----------
    telemetry        : dict  — `telemetry` block from /analyze-video response
    stats            : dict  — 8D metrics block
    athlete_name     : str   — shown in left-panel HUD
    kinematic_deltas : dict  — gaps vs matched pro (enables personalized targets)
    sport            : str   — sport ID for ideal range lookup
    pro_match        : str   — matched pro name (shown in right-panel subtitle)
    video_path       : str   — path to the uploaded video file for overlay
    clip_start_sec   : float — when user selected a clip, this is the clip's
                               start time in the full video (seconds). Used to
                               correctly seek the video to the shot window.

    Returns
    -------
    bytes — raw MP4 bytes, or None if rendering failed.
    """
    frames    = telemetry.get("frames") or []
    dip       = telemetry.get("dip")     or {}
    release   = telemetry.get("release") or {}
    fps_tel   = float(telemetry.get("fps") or 30.0)
    dip_t     = float(dip.get("time_sec")     or 0.0)
    release_t = float(release.get("time_sec") or (dip_t + 0.5))

    # ── 1. Ensure we have at least a minimal frame set ───────────────────────
    is_synthetic = False
    frames = _build_frame_index(frames)       # already filters NaN & sorts

    if len(frames) < 4:
        logger.warning(
            "telemetry.frames has only %d valid entries — building anchor frames from dip/release",
            len(frames),
        )
        anchor = _build_anchor_frames(dip, release, fps_tel)
        if anchor:
            frames = _build_frame_index(anchor)
            is_synthetic = True
        else:
            logger.error("No usable pose data (frames and anchors both empty). "
                         "Analysis likely hit fallback — video may lack a clear shot.")
            return None

    if not frames:
        return None

    # ── 2. Build correction targets & subtitle ───────────────────────────────
    targets     = _build_targets(stats, kinematic_deltas, sport)
    knee_angle  = stats.get("knee_angle",  targets["knee_angle"])
    elbow_angle = stats.get("elbow_angle", targets["elbow_angle"])
    arc_deg     = stats.get("shot_arc_deg",targets["shot_arc_deg"])

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
        if corrections
        else f"Mechanics within optimal range  |  {source_label}"
    )

    # ── 3. Open video file ────────────────────────────────────────────────────
    cap       = None
    video_fps = fps_tel
    use_video = False

    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            vfps = cap.get(cv2.CAP_PROP_FPS)
            video_fps = vfps if vfps and vfps > 0 else fps_tel
            use_video = True
            # Seek to 0.5s before dip (absolute video time)
            # clip_start_sec converts clip-relative dip_t → absolute video time
            seek_abs = max(0.0, clip_start_sec + dip_t - 0.5)
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_abs * 1000.0)
        else:
            cap.release()
            cap = None

    out_fps   = video_fps
    # Render window: from seek_abs to clip_start_sec + release_t + 1.5s
    t_stop    = clip_start_sec + release_t + 1.5
    seek_abs  = max(0.0, clip_start_sec + dip_t - 0.5) if use_video else 0.0

    # ── 4. Write output video ─────────────────────────────────────────────────
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=tempfile.gettempdir())
    os.close(tmp_fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, out_fps, (FRAME_W, PANEL_H))

    try:
        if use_video:
            render_idx = 0
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break

                t_abs = seek_abs + render_idx / video_fps
                render_idx += 1

                if t_abs > t_stop:
                    break

                # Clip-relative time for telemetry lookup
                t_rel = t_abs - clip_start_sec

                joints_norm = _nearest_joints(frames, t_rel)
                if joints_norm is None:
                    continue

                left_bg, box = _resize_to_panel(bgr)
                right_bg      = _make_right_bg(left_bg.copy())

                # Determine if this frame is within the active telemetry window
                t_nearest = frames[0][0] if frames else 0
                for ft, fj in frames:
                    if abs(ft - t_rel) < abs(t_nearest - t_rel):
                        t_nearest = ft

                out_frame = _render_frame_pair(
                    left_bg, right_bg, joints_norm,
                    stats, targets,
                    t_rel, dip_t, release_t,
                    athlete_name, subtitle_r,
                    knee_angle, elbow_angle,
                    is_synthetic, box,
                )
                writer.write(out_frame)

            cap.release()

        else:
            # ── Skeleton-only fallback ────────────────────────────────────────
            for t_val, joints_norm in frames:
                left  = np.full((PANEL_H, PANEL_W, 3), BG_COLOR, dtype=np.uint8)
                right = np.full((PANEL_H, PANEL_W, 3), BG_COLOR, dtype=np.uint8)
                out_frame = _render_frame_pair(
                    left, right, joints_norm,
                    stats, targets,
                    t_val, dip_t, release_t,
                    athlete_name, subtitle_r,
                    knee_angle, elbow_angle,
                    is_synthetic, None,
                )
                writer.write(out_frame)

    except Exception:
        logger.exception("Error during correction video rendering")
    finally:
        writer.release()

    with open(tmp_path, "rb") as fh:
        video_bytes = fh.read()
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    if len(video_bytes) < 1024:
        logger.error("Output video too small (%d bytes) — no frames were written", len(video_bytes))
        return None

    return video_bytes
