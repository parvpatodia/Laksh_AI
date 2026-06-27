"""
Shared constants for Laksh.ai.

These values define the ChromaDB embedding schema and are consumed by both
app/main.py (query time) and app/db_seeder.py (index time). Keeping them here
guarantees both sides always live in the same normalised space.

WARNING: Changing COLLECTION_NAME or FEATURE_WEIGHTS requires re-seeding
ChromaDB (delete chroma_db/ and restart) — old and new vectors are not
comparable after either change.
"""

# ChromaDB collection name — bump the version suffix to force a re-seed when
# the embedding schema changes.
COLLECTION_NAME = "apex_oracle_v7"

# 8D feature weights — equalise L2 distance variance across all dimensions.
# Each weight scales its dimension so the full biomechanical span maps to
# ~100 normalised units, preventing high-magnitude dimensions (e.g.
# kinetic_sync_ms ~300) from dominating the cosine similarity search.
#
# Vector schema (index → metric):
#   v0  release_velocity_mps   raw m/s,   span ~6     → ×16.6  → ~100
#   v1  shot_arc_deg           degrees,   span ~30    → ×3.3   → ~100
#   v2  knee_angle             degrees,   span ~80    → ×1.25  → ~100
#   v3  elbow_angle            degrees,   span ~60    → ×1.66  → ~100
#   v4  kinetic_sync_ms        ms,        span ~300   → ×0.33  → ~100
#   v5  fluidity_score         0–100,     span ~60    → ×1.66  → ~100
#   v6  hip_rotation_deg       degrees,   span ~45    → ×2.22  → ~100
#   v7  balance_index          0–100,     span ~50    → ×2.0   → ~100
FEATURE_WEIGHTS: list[float] = [
    16.6,   # v0: release_velocity_mps
    3.3,    # v1: shot_arc_deg
    1.25,   # v2: knee_angle
    1.66,   # v3: elbow_angle
    0.33,   # v4: kinetic_sync_ms
    1.66,   # v5: fluidity_score
    2.22,   # v6: hip_rotation_deg
    2.0,    # v7: balance_index
]

# Default metric values used when a measurement is None (e.g. fallback mode)
# before building the ChromaDB query vector. Represent league-average mechanics.
METRIC_DEFAULTS: dict[str, float] = {
    "release_velocity_mps": 7.0,
    "shot_arc_deg": 45.0,
    "knee_angle": 145.0,
    "elbow_angle": 165.0,
    "kinetic_sync_ms": 150.0,
    "fluidity_score": 65.0,
    "hip_rotation_deg": 5.0,
    "balance_index": 85.0,
}
