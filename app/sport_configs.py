"""
Sports-agnostic configuration for Laksh.ai video analysis.
Each sport defines: event phases, key joints, metrics schema, and UI labels.
Adding a new sport = add a config entry + implement analyzer in physics_engine.

Note on labels/units: these describe single-camera, monocular-pose estimates.
They are directional practice-feedback cues, NOT lab-grade biomechanics. Metrics
that cannot be honestly measured from one phone camera (e.g. release power, hip
rotation) are labelled as proxies / low-confidence rather than given false units.
"""
from typing import TypedDict, List


class SportMetricConfig(TypedDict):
    key: str
    label: str
    unit: str
    ideal_range: str
    limitation: str


class SportConfig(TypedDict):
    id: str
    name: str
    description: str
    event_phases: List[str]  # e.g. ["Setup", "Dip", "Drive", "Release", "Follow-through"]
    metrics: List[SportMetricConfig]
    min_clip_sec: float
    recommended_aspect: str  # e.g. "16:9 landscape, 45° offset"
    pro_db_collection: str | None  # ChromaDB collection for pro matching; None = generic feedback only


# Basketball (jump shot) — fully implemented
BASKETBALL_CONFIG: SportConfig = {
    "id": "basketball",
    "name": "Basketball",
    "description": "Jump-shot form features from monocular video: arc, knee/elbow flexion, release timing, plus proxy power/balance scores. Directional practice cues, not lab-grade biomechanics.",
    "event_phases": ["Setup", "Dip", "Drive", "Release", "Follow-through"],
    "min_clip_sec": 2.0,
    "recommended_aspect": "16:9 landscape, 45° front-offset",
    "pro_db_collection": "apex_oracle_v7",
    "metrics": [
        {"key": "release_velocity_mps", "label": "Release Power", "unit": "index", "ideal_range": "7–9 (proxy)", "limitation": "2D pixel-ratio PROXY scaled into a plausible range — NOT a true velocity. No ball tracking or camera calibration; the 'm/s'-like number is not physically measured."},
        {"key": "shot_arc_deg", "label": "Shot Arc (est.)", "unit": "°", "ideal_range": "45–55", "limitation": "Wrist-trajectory proxy, not ball trajectory; side-view compresses arc."},
        {"key": "knee_angle", "label": "Knee Flexion", "unit": "°", "ideal_range": "140–165", "limitation": "MediaPipe 3D; coarse (~±5–15°) on a clean side view."},
        {"key": "elbow_angle", "label": "Elbow Flexion", "unit": "°", "ideal_range": "165–178", "limitation": "Single-camera; coarse (~±5–15°)."},
        {"key": "hip_rotation_deg", "label": "Hip Rotation (low-confidence)", "unit": "°", "ideal_range": "5–15", "limitation": "Depth-axis (yaw) from a single camera — low confidence; closer to noise than a reliable measurement."},
        {"key": "kinetic_sync_ms", "label": "Kinetic Sync", "unit": "ms", "ideal_range": "120–250", "limitation": "Frame-rate dependent."},
        {"key": "balance_index", "label": "Balance Index", "unit": "/100", "ideal_range": "85–99", "limitation": "2D projection proxy."},
        {"key": "fluidity_score", "label": "Fluidity Score", "unit": "/100", "ideal_range": "75–99", "limitation": "Pose-noise sensitive proxy."},
    ],
}

# Tennis serve — placeholder for future
TENNIS_CONFIG: SportConfig = {
    "id": "tennis",
    "name": "Tennis",
    "description": "Serve mechanics: toss, racquet drop, contact, follow-through. (Coming soon)",
    "event_phases": ["Toss", "Racquet Drop", "Contact", "Follow-through"],
    "min_clip_sec": 3.0,
    "recommended_aspect": "16:9 side view",
    "pro_db_collection": None,
    "metrics": [],
}

# Golf swing — placeholder
GOLF_CONFIG: SportConfig = {
    "id": "golf",
    "name": "Golf",
    "description": "Swing mechanics: backswing, downswing, impact, follow-through. (Coming soon)",
    "event_phases": ["Address", "Backswing", "Downswing", "Impact", "Follow-through"],
    "min_clip_sec": 3.0,
    "recommended_aspect": "16:9 DTL or face-on",
    "pro_db_collection": None,
    "metrics": [],
}


# Gym — 12 compound movements via the frozen v0 exercise taxonomy. Unlike
# basketball, gym has no single canonical movement; the exercise_id field is
# chosen per-request from app.gym.exercises_v0.EXERCISES_V0. The phases and
# metrics below describe the measurement spine shape, not an individual lift.
GYM_CONFIG: SportConfig = {
    "id": "gym",
    "name": "Gym",
    "description": "Compound-lift biomechanics: rep segmentation, per-rep feature vector, honest calibration.",
    "event_phases": ["Setup", "Eccentric", "Bottom", "Concentric", "Lockout"],
    "min_clip_sec": 3.0,
    "recommended_aspect": "16:9 side view, 2m offset, hip-height lens",
    "pro_db_collection": None,  # cited reference ranges will live in calibration_v0 config
    "metrics": [
        {"key": "rep_duration_s",           "label": "Rep Duration",        "unit": "s",              "ideal_range": "uncalibrated_v0", "limitation": "Awaiting labeled reference subset."},
        {"key": "eccentric_duration_s",     "label": "Eccentric Phase",     "unit": "s",              "ideal_range": "uncalibrated_v0", "limitation": "Phase split depends on peak detection."},
        {"key": "concentric_duration_s",    "label": "Concentric Phase",    "unit": "s",              "ideal_range": "uncalibrated_v0", "limitation": "Phase split depends on peak detection."},
        {"key": "tempo_ratio_ecc_over_con", "label": "Tempo Ratio",         "unit": "ratio",          "ideal_range": "uncalibrated_v0", "limitation": "Ratio undefined when either phase is zero."},
        {"key": "signal_amplitude",         "label": "Signal Amplitude",    "unit": "deg|norm_y",     "ideal_range": "uncalibrated_v0", "limitation": "2D projection; side-view compression."},
        {"key": "primary_joints_min_visibility", "label": "Min Visibility", "unit": "visibility",     "ideal_range": "uncalibrated_v0", "limitation": "Occlusion drives this below threshold."},
        {"key": "primary_joints_missing_frac",   "label": "Missing Frames", "unit": "frac",           "ideal_range": "uncalibrated_v0", "limitation": "High missingness flips rep to degraded."},
    ],
}

# Available sports
SPORT_CONFIGS: dict[str, SportConfig] = {
    "basketball": BASKETBALL_CONFIG,
    "gym": GYM_CONFIG,
    "tennis": TENNIS_CONFIG,
    "golf": GOLF_CONFIG,
}


def get_sport_config(sport_id: str) -> SportConfig:
    return SPORT_CONFIGS.get(sport_id, BASKETBALL_CONFIG)


def get_available_sports() -> List[dict]:
    """Return list of {id, name, available: bool} for UI dropdown."""
    return [
        {"id": "basketball", "name": "Basketball", "available": True},
        {"id": "gym", "name": "Gym", "available": True},
        {"id": "tennis", "name": "Tennis", "available": False},
        {"id": "golf", "name": "Golf", "available": False},
    ]
