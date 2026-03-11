"""
Apex.ai Dynamic NBA Database Seeder.

Rate-limit-safe pipeline to seed ChromaDB collection 'apex_oracle_v7' with
active NBA players. Uses nba_api.LeagueDashPlayerStats for a single bulk
fetch. Deterministic heuristics synthesize 8D kinematic vectors from real stats.

8D Vector schema (aligned 1:1 with physics_engine output and main.py query_vector):
  v0: release_velocity_mps  (raw m/s,  4.0 – 9.0)
  v1: shot_arc_deg          (degrees, 38 – 55)
  v2: knee_angle            (degrees at dip, 135 – 175)
  v3: elbow_angle           (degrees at release, 150 – 178)
  v4: kinetic_sync_ms       (dip→release ms, 150 – 600)
  v5: fluidity_score        (0 – 100)
  v6: hip_rotation_deg      (XZ-plane yaw, −20 – +20)
  v7: balance_index         (0 – 100)
"""

import logging
import time
from typing import Any, Optional

import chromadb
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "apex_oracle_v7"
NBA_API_DELAY = 0.6  # seconds between requests (rate-limit safety)

# Expert feature weights — equalise L2 distance variance across all 8 dimensions.
# Each weight scales its dimension so the full biomechanical span maps to ~100 units,
# preventing high-magnitude dimensions (e.g. kinetic_sync_ms ~300) from dominating search.
# Applied to embeddings only; raw values are stored unchanged in metadata for the UI.
FEATURE_WEIGHTS = [
    16.6,  # v0: velocity_mps    (span ~6 m/s    → ×16.6 → ~100)
    3.3,   # v1: arc_deg         (span ~30 °      → ×3.3  → ~100)
    1.25,  # v2: knee_angle      (span ~80 °      → ×1.25 → ~100)
    1.66,  # v3: elbow_angle     (span ~60 °      → ×1.66 → ~100)
    0.33,  # v4: kinetic_sync_ms (span ~300 ms    → ×0.33 → ~100)
    1.66,  # v5: fluidity_score  (span ~60        → ×1.66 → ~100)
    2.22,  # v6: hip_rotation    (span ~45 °      → ×2.22 → ~100)
    2.0,   # v7: balance_index   (span ~50        → ×2.0  → ~100)
]


def translate_to_kinematics(row: dict[str, Any]) -> list[float]:
    """
    Deterministic heuristic: map NBA box-score stats to 8D kinematic vector.
    Output schema is aligned 1:1 with physics_engine.py output so ChromaDB
    similarity search compares like-for-like units.

    Elite shooters (Curry, Thompson) → ~[7.5, 48, 155, 168, 230, 92, 4, 90]
    """
    def _f(key: str, default: float = 0.0) -> float:
        v = row.get(key, default)
        try:
            return float(v) if v is not None and str(v) != "" else default
        except (ValueError, TypeError):
            return default

    reb = _f("REB", 5.0)
    ast = _f("AST", 2.0)
    tov = max(_f("TOV", 1.0), 0.1)
    fg3_pct = _f("FG3_PCT", 0.35)
    pts = _f("PTS", 10.0)
    gp = max(_f("GP", 10), 1)

    reb_pg = reb / gp
    ast_pg = ast / gp
    ast_tov = ast / tov
    guard_score = ast_pg / (reb_pg + 1)

    # v0: release_velocity_mps — raw m/s (no scaling).
    # Elite guards: 7.0-8.5 m/s, post bigs: 5.5-6.5 m/s.
    release_velocity_mps = 5.5 + guard_score * 1.2 + fg3_pct * 2.5
    release_velocity_mps = max(4.0, min(9.0, release_velocity_mps))

    # v1: shot_arc_deg — high 3P% correlates with textbook 45-50° arc.
    shot_arc_deg = 38.0 + fg3_pct * 22.0
    shot_arc_deg = max(38.0, min(55.0, shot_arc_deg))

    # v2: knee_angle at dip — guards use less knee bend (155-170°), bigs more (135-155°).
    knee_angle = 170.0 - min(reb_pg * 2.5, 30.0)
    knee_angle = max(135.0, min(175.0, knee_angle))

    # v3: elbow_angle at release — elite shooters fully extend 162-175°.
    elbow_angle = 158.0 + fg3_pct * 18.0
    elbow_angle = max(150.0, min(178.0, elbow_angle))

    # v4: kinetic_sync_ms — dip-to-release time in milliseconds.
    # Derived from guard_score: guards have quicker releases (~167-267 ms at 30 fps).
    # release_frames = 20 - min(guard_score * 6, 12) → 8-20 frames
    # kinetic_sync_ms = release_frames * (1000 / 30) → 267-667 ms
    release_frames = 20.0 - min(guard_score * 6.0, 12.0)
    release_frames = max(8.0, min(20.0, release_frames))
    kinetic_sync_ms = round(release_frames * (1000.0 / 30.0), 1)

    # v5: fluidity_score — smoothness of shooting motion (0-100).
    # High 3P% + high AST/TOV = repeatable, fluid mechanics.
    fluidity_score = 70.0 + ast_tov * 5.0 + fg3_pct * 35.0
    fluidity_score = max(60.0, min(98.0, fluidity_score))

    # v6: hip_rotation_deg — XZ-plane yaw of hip line at dip.
    # Squared-up shooters (high fg3_pct) have small positive rotation.
    # Post-heavy bigs (high reb_pg) rotate more toward basket (negative).
    hip_rotation_deg = (fg3_pct - 0.33) * 40.0 - (reb_pg - 5.0) * 0.5
    hip_rotation_deg = max(-20.0, min(20.0, hip_rotation_deg))

    # v7: balance_index — lateral stability proxy (0-100).
    # AST/TOV measures decision quality, which correlates with stable footwork.
    balance_index = 50.0 + ast_tov * 10.0
    balance_index = max(50.0, min(98.0, balance_index))

    return [
        round(release_velocity_mps, 2),
        round(shot_arc_deg, 1),
        round(knee_angle, 1),
        round(elbow_angle, 1),
        round(kinetic_sync_ms, 1),
        round(fluidity_score, 1),
        round(hip_rotation_deg, 2),
        round(balance_index, 1),
    ]


def seed_database(chroma_client: Optional[chromadb.Client] = None) -> int:
    """
    Fetch active NBA players, compute 8D vectors, seed ChromaDB.
    Returns number of players seeded. Idempotent (wipes collection first).
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
    except ImportError:
        logger.warning("nba_api not installed; skipping seed.")
        return 0

    time.sleep(NBA_API_DELAY)
    try:
        endpoint = leaguedashplayerstats.LeagueDashPlayerStats()
        df = endpoint.get_data_frames()[0]
    except Exception as e:
        logger.error("NBA API fetch failed: %s", e)
        return 0

    if df.empty:
        logger.warning("No player data returned.")
        return 0

    # Filter: minimum games played
    df = df[df["GP"] >= 5].reset_index(drop=True)
    if df.empty:
        return 0

    embeddings = []
    documents = []
    ids = []
    metadatas = []

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        vec = translate_to_kinematics(row_dict)
        name = str(row.get("PLAYER_NAME", f"Player_{idx}")).strip()
        if not name or name == "nan":
            continue
        player_id = int(row.get("PLAYER_ID", idx))
        doc_id = f"{player_id}_{row.get('TEAM_ID', idx)}"
        meta = {"player_id": player_id}
        for i, v in enumerate(vec):
            meta[f"v{i}"] = float(v)  # raw value kept for UI display
        weighted_vec = [v * w for v, w in zip(vec, FEATURE_WEIGHTS)]
        embeddings.append(weighted_vec)
        documents.append(name)
        ids.append(doc_id)
        metadatas.append(meta)

    if not embeddings:
        return 0

    client = chroma_client or chromadb.Client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    # ChromaDB add has limit; batch in chunks of 100
    chunk = 100
    for i in range(0, len(embeddings), chunk):
        end = min(i + chunk, len(embeddings))
        collection.add(
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            ids=ids[i:end],
            metadatas=metadatas[i:end],
        )

    logger.info("Seeded %d NBA players into %s", len(embeddings), COLLECTION_NAME)
    return len(embeddings)
