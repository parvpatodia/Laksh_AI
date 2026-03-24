"""
Sports-agnostic configuration for Laksh.ai video analysis.
Each sport defines: event phases, key joints, metrics schema, and UI labels.
Adding a new sport = add a config entry + implement analyzer in physics_engine.
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
    "description": "Jump shot biomechanics: release velocity, arc, knee/elbow flexion, kinetic sync.",
    "event_phases": ["Setup", "Dip", "Drive", "Release", "Follow-through"],
    "min_clip_sec": 2.0,
    "recommended_aspect": "16:9 landscape, 45° front-offset",
    "pro_db_collection": "apex_oracle_v7",
    "metrics": [
        {"key": "release_velocity_mps", "label": "Release Velocity", "unit": "m/s", "ideal_range": "7–9", "limitation": "2D proxy; ball tracking would improve."},
        {"key": "shot_arc_deg", "label": "Shot Arc", "unit": "°", "ideal_range": "45–55", "limitation": "Side-view compresses arc."},
        {"key": "knee_angle", "label": "Knee Flexion", "unit": "°", "ideal_range": "140–165", "limitation": "MediaPipe 3D."},
        {"key": "elbow_angle", "label": "Elbow Flexion", "unit": "°", "ideal_range": "165–178", "limitation": "Single-camera."},
        {"key": "hip_rotation_deg", "label": "Hip Rotation", "unit": "°", "ideal_range": "5–15", "limitation": "View-dependent."},
        {"key": "kinetic_sync_ms", "label": "Kinetic Sync", "unit": "ms", "ideal_range": "120–250", "limitation": "Frame-rate dependent."},
        {"key": "balance_index", "label": "Balance Index", "unit": "/100", "ideal_range": "85–99", "limitation": "2D projection."},
        {"key": "fluidity_score", "label": "Fluidity Score", "unit": "/100", "ideal_range": "75–99", "limitation": "Pose noise."},
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

# Available sports
SPORT_CONFIGS: dict[str, SportConfig] = {
    "basketball": BASKETBALL_CONFIG,
    "tennis": TENNIS_CONFIG,
    "golf": GOLF_CONFIG,
}


def get_sport_config(sport_id: str) -> SportConfig:
    return SPORT_CONFIGS.get(sport_id, BASKETBALL_CONFIG)


def get_available_sports() -> List[dict]:
    """Return list of {id, name, available: bool} for UI dropdown."""
    return [
        {"id": "basketball", "name": "Basketball", "available": True},
        {"id": "tennis", "name": "Tennis", "available": False},
        {"id": "golf", "name": "Golf", "available": False},
    ]
