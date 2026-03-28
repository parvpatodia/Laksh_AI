"""
All numeric thresholds used by the biomechanics pipeline in one place.

Previously these were scattered as magic numbers across physics_engine.py and
correction_engine.py with no names or explanation. Any change here propagates
to every consumer automatically — no more hunting for 0.048 in five files.

Sections mirror the analysis pipeline phases in KinematicAnalyzer.analyze().
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Frame pre-processing
# ---------------------------------------------------------------------------

FRAME_RESIZE_MAX_DIM: int = 720        # Max long-side pixels fed to MediaPipe
GAMMA_POWER: float = 0.85              # gamma_contrast: shadow lift exponent
CONTRAST_BOOST: float = 1.18           # gamma_contrast: local contrast multiplier
SHARPEN_SIGNAL_WEIGHT: float = 1.35    # denoise_sharpen: addWeighted signal weight
SHARPEN_BLUR_WEIGHT: float = -0.35     # denoise_sharpen: addWeighted blur weight (negative = subtract)
DENOISE_FILTER_STRENGTH: int = 3       # fastNlMeansDenoisingColored h / hColor
DENOISE_TEMPLATE_WINDOW: int = 7       # fastNlMeansDenoisingColored templateWindowSize
DENOISE_SEARCH_WINDOW: int = 21        # fastNlMeansDenoisingColored searchWindowSize


# ---------------------------------------------------------------------------
# Video quality scoring
# ---------------------------------------------------------------------------

VQ_MIN_WIDTH: int = 320                # Below this → low-res warning
VQ_MIN_HEIGHT: int = 240
VQ_MIN_FPS: float = 20.0               # Below this → framerate warning
VQ_SLOWMO_FPS: float = 90.0            # Above this → slow-motion note
VQ_MIN_ASPECT: float = 0.6             # Below this → portrait-crop warning
VQ_MAX_ASPECT: float = 2.2             # Above this → ultra-wide warning
VQ_MIN_FRAMES: int = 30                # Below this → short-clip note
VQ_FPS_REFERENCE: float = 30.0         # Denominator used in FPS quality score
VQ_RES_BASELINE: int = 320             # Log-scale resolution baseline (pixels)


# ---------------------------------------------------------------------------
# Shot-type classification (wrist / hip vertical span thresholds)
# Jump-shot: large vertical excursion. Set-shot / free-throw: small excursion.
# ---------------------------------------------------------------------------

JUMP_SHOT_WRIST_SPAN: float = 0.048    # Normalised image coords
JUMP_SHOT_HIP_SPAN: float = 0.042
SET_SHOT_WRIST_SPAN: float = 0.058     # Alternate gate (wrist wider than jump but hip very still)
SET_SHOT_HIP_SPAN: float = 0.028


# ---------------------------------------------------------------------------
# Phase detection: dip / release frame windows
# ---------------------------------------------------------------------------

JUMP_SHOT_SEARCH_DURATION_SEC: float = 1.5   # Post-dip window to search for release
SET_SHOT_DIP_WINDOW_LO: float = 0.18         # Fraction of total frames for set-shot dip search start
SET_SHOT_DIP_WINDOW_HI: float = 0.92         # Fraction of total frames for set-shot dip search end
SET_SHOT_FPS_DIP_FALLBACK_SEC: float = 0.20  # fps*N fallback dip offset when release is early
SET_SHOT_MIN_DIP_OFFSET_SEC: float = 0.15    # Minimum release-dip gap for set-shot

POST_RELEASE_DURATION_SEC: float = 1.5       # Telemetry frames captured after release


# ---------------------------------------------------------------------------
# Joint angle fallbacks (used when world-depth is unreliable AND 2D fails)
# These are intentionally imported from app.constants.METRIC_DEFAULTS, but
# kept here as biomechanical names for readability inside physics_engine.py.
# ---------------------------------------------------------------------------

KNEE_ANGLE_FALLBACK_DEG: float = 135.0    # League-average dip angle
ELBOW_ANGLE_FALLBACK_DEG: float = 165.0   # League-average release angle
KNEE_ANGLE_VALIDITY_MIN: float = 10.0     # Discard 3D angles below this (degenerate frame)
ELBOW_ANGLE_VALIDITY_MIN: float = 10.0


# ---------------------------------------------------------------------------
# Shot arc
# ---------------------------------------------------------------------------

WRIST_FLICK_OFFSET_DEG: float = 15.0       # Empirical correction: lever angle → true arc
ARC_BIOLOGICAL_MIN_DEG: float = 30.0       # Hard clamp — no realistic shot below this
ARC_BIOLOGICAL_MAX_DEG: float = 75.0       # Hard clamp — no realistic shot above this
ARC_DEFAULT_DEG: float = 48.5              # Used when lever angle cannot be computed
ARC_POST_RELEASE_FRAMES: int = 7           # Max frames used for parabolic arc fit


# ---------------------------------------------------------------------------
# Kinetic sync (dip → release timing)
# ---------------------------------------------------------------------------

KINETIC_SYNC_BASELINE_FRAMES: float = 8.0   # Reference frame count at 30 fps (empirical)
KINETIC_SYNC_MIN_MS: float = 120.0           # Hard clamp lower bound
KINETIC_SYNC_MAX_MS: float = 395.0           # Hard clamp upper bound
KINETIC_SYNC_FPS_DILATION_THRESHOLD: int = 15  # Raw frames > this triggers FPS correction


# ---------------------------------------------------------------------------
# Velocity (dimensionless pixel-space proxy scaled to m/s)
# ---------------------------------------------------------------------------

VELOCITY_SCALE_FACTOR: float = 3.5       # Empirical: power_ratio → approximate m/s
VELOCITY_MIN_MPS: float = 4.0            # Hard clamp lower
VELOCITY_MAX_MPS: float = 10.0           # Hard clamp upper
VELOCITY_DEFAULT_MPS: float = 6.5        # Used when power_ratio == 0


# ---------------------------------------------------------------------------
# Hip rotation (yaw)
# ---------------------------------------------------------------------------

YAW_CLAMP_DEG: float = 45.0              # ±45° is the biomechanical ceiling


# ---------------------------------------------------------------------------
# Balance index
# ---------------------------------------------------------------------------

BALANCE_DEFAULT: int = 85
BALANCE_DEVIATION_SCALE: float = 120.0   # Maps normalised hip-ankle offset → 0-100 score
BALANCE_SCORE_MIN: int = 40
BALANCE_SCORE_MAX: int = 99
BALANCE_TORSO_EPSILON: float = 1e-4      # Min torso length to avoid divide-by-zero


# ---------------------------------------------------------------------------
# Fluidity score
# ---------------------------------------------------------------------------

FLUIDITY_DEFAULT: int = 65
FLUIDITY_JERK_SCALE: float = 2000.0      # Maps wrist acceleration std-dev → 0-100 score
FLUIDITY_SCORE_MIN: int = 40
FLUIDITY_SCORE_MAX: int = 99
FLUIDITY_MIN_FRAMES: int = 3             # Minimum frames between dip and release to compute


# ---------------------------------------------------------------------------
# Angle uncertainty (PMC 9397457)
# ---------------------------------------------------------------------------

UNCERTAINTY_WINDOW_HALF: int = 3         # Half-window of frames around dip/release
UNCERTAINTY_CLAMP_MIN_DEG: float = 3.0
UNCERTAINTY_CLAMP_MAX_DEG: float = 12.0
UNCERTAINTY_VARIANCE_MULTIPLIER: float = 1.2
UNCERTAINTY_LOW_VISIBILITY_INFLATE: float = 1.5   # Applied when visibility < threshold
UNCERTAINTY_VISIBILITY_THRESHOLD: float = 0.6
UNCERTAINTY_MIN_SAMPLES: int = 3         # Min angle samples required to trust std-dev
UNCERTAINTY_FALLBACK_DEG: float = 5.0    # Used when fewer than min samples
UNCERTAINTY_2D_FALLBACK_INFLATE: float = 1.12  # Extra inflate when using 2D fallback angles
UNCERTAINTY_2D_FALLBACK_CAP_DEG: float = 15.0


# ---------------------------------------------------------------------------
# Per-metric confidence base values
# (measured, predicted) — calibrated so partial-mode analysis is not over-trusted
# ---------------------------------------------------------------------------

CONFIDENCE_VELOCITY = (0.86, 0.55)
CONFIDENCE_ARC = (0.84, 0.58)
CONFIDENCE_KNEE = (0.82, 0.68)
CONFIDENCE_ELBOW = (0.82, 0.68)
CONFIDENCE_KINETIC_SYNC = (0.78, 0.52)
CONFIDENCE_HIP_ROTATION = (0.74, 0.52)
CONFIDENCE_BALANCE = (0.72, 0.55)
CONFIDENCE_FLUIDITY = (0.70, 0.55)

# Map metric name → (measured_conf, predicted_conf)
METRIC_CONFIDENCE_MAP: dict[str, tuple[float, float]] = {
    "release_velocity_mps": CONFIDENCE_VELOCITY,
    "shot_arc_deg":         CONFIDENCE_ARC,
    "knee_angle":           CONFIDENCE_KNEE,
    "elbow_angle":          CONFIDENCE_ELBOW,
    "kinetic_sync_ms":      CONFIDENCE_KINETIC_SYNC,
    "hip_rotation_deg":     CONFIDENCE_HIP_ROTATION,
    "balance_index":        CONFIDENCE_BALANCE,
    "fluidity_score":       CONFIDENCE_FLUIDITY,
}
