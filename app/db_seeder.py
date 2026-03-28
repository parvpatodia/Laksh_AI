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
import os
import time
from typing import Any, Optional

import chromadb
import pandas as pd

from app.constants import COLLECTION_NAME, FEATURE_WEIGHTS  # was: both defined inline here — moved to app/constants.py so main.py and db_seeder.py always use the same values

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)
NBA_API_DELAY = 0.6  # seconds between requests (rate-limit safety)
NBA_API_TIMEOUT = int(os.environ.get("NBA_API_TIMEOUT", "90"))  # cloud→stats.nba.com is slow
NBA_API_RETRIES = 2  # try twice before fallback

# Fallback seed when NBA API fails (timeout, rate-limit, cloud IP block).
# Synthetic stats run through translate_to_kinematics for consistent 8D vectors.
# Format: (name, player_id, {REB, AST, TOV, FG3_PCT, PTS, GP})
FALLBACK_PLAYERS = [
    ("Stephen Curry", 201939, {"REB": 5.2, "AST": 6.5, "TOV": 2.8, "FG3_PCT": 0.429, "PTS": 26.4, "GP": 69}),
    ("Klay Thompson", 202691, {"REB": 4.0, "AST": 2.2, "TOV": 1.6, "FG3_PCT": 0.412, "PTS": 21.5, "GP": 65}),
    ("LeBron James", 2544, {"REB": 8.3, "AST": 7.1, "TOV": 3.5, "FG3_PCT": 0.355, "PTS": 25.7, "GP": 55}),
    ("Kevin Durant", 201142, {"REB": 6.4, "AST": 5.0, "TOV": 2.9, "FG3_PCT": 0.389, "PTS": 27.1, "GP": 47}),
    ("Luka Doncic", 1629029, {"REB": 8.0, "AST": 8.8, "TOV": 4.0, "FG3_PCT": 0.368, "PTS": 28.4, "GP": 66}),
    ("Nikola Jokic", 203999, {"REB": 12.2, "AST": 9.8, "TOV": 3.6, "FG3_PCT": 0.332, "PTS": 26.2, "GP": 69}),
    ("Giannis Antetokounmpo", 203507, {"REB": 11.5, "AST": 5.7, "TOV": 3.2, "FG3_PCT": 0.272, "PTS": 30.4, "GP": 63}),
    ("Joel Embiid", 203954, {"REB": 11.0, "AST": 3.6, "TOV": 3.4, "FG3_PCT": 0.348, "PTS": 28.7, "GP": 39}),
    ("Jayson Tatum", 1628369, {"REB": 8.1, "AST": 4.6, "TOV": 2.6, "FG3_PCT": 0.375, "PTS": 26.9, "GP": 74}),
    ("Devin Booker", 1626164, {"REB": 4.2, "AST": 6.8, "TOV": 2.9, "FG3_PCT": 0.362, "PTS": 27.1, "GP": 68}),
    ("Anthony Edwards", 1630162, {"REB": 5.5, "AST": 5.1, "TOV": 3.0, "FG3_PCT": 0.358, "PTS": 25.9, "GP": 79}),
    ("Tyrese Haliburton", 1630169, {"REB": 3.9, "AST": 10.9, "TOV": 2.3, "FG3_PCT": 0.367, "PTS": 20.1, "GP": 69}),
    ("Damian Lillard", 203081, {"REB": 4.2, "AST": 7.2, "TOV": 2.8, "FG3_PCT": 0.378, "PTS": 25.1, "GP": 73}),
    ("Donovan Mitchell", 1628378, {"REB": 4.3, "AST": 5.3, "TOV": 2.8, "FG3_PCT": 0.368, "PTS": 26.6, "GP": 55}),
    ("Shai Gilgeous-Alexander", 1628983, {"REB": 5.5, "AST": 6.2, "TOV": 2.5, "FG3_PCT": 0.353, "PTS": 30.1, "GP": 75}),
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

    vec = [
        round(release_velocity_mps, 2),
        round(shot_arc_deg, 1),
        round(knee_angle, 1),
        round(elbow_angle, 1),
        round(kinetic_sync_ms, 1),
        round(fluidity_score, 1),
        round(hip_rotation_deg, 2),
        round(balance_index, 1),
    ]
    # Guard: ChromaDB silently accepts wrong-dimension vectors; catch schema drift here
    assert len(vec) == 8, f"translate_to_kinematics must return 8D vector, got {len(vec)}D"
    return vec


def _seed_fallback(chroma_client: chromadb.Client) -> int:
    """Seed ChromaDB with static fallback players when NBA API fails."""
    embeddings = []
    documents = []
    ids = []
    metadatas = []

    for i, (name, player_id, stats) in enumerate(FALLBACK_PLAYERS):
        vec = translate_to_kinematics(stats)
        meta = {"player_id": player_id}
        for j, v in enumerate(vec):
            meta[f"v{j}"] = float(v)
        weighted_vec = [v * w for v, w in zip(vec, FEATURE_WEIGHTS)]
        embeddings.append(weighted_vec)
        documents.append(name)
        ids.append(f"fallback_{player_id}_{i}")
        metadatas.append(meta)

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        embeddings=embeddings,
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )
    logger.info("Seeded %d fallback players (NBA API unavailable).", len(embeddings))
    return len(embeddings)


def _fetch_nba_data():
    """
    Fetch NBA player stats with extended timeout and retry.
    Returns DataFrame or None on failure.
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
    except ImportError:
        return None

    last_error = None
    for attempt in range(1, NBA_API_RETRIES + 1):
        try:
            time.sleep(NBA_API_DELAY)
            endpoint = leaguedashplayerstats.LeagueDashPlayerStats()
            # Extend nba_api timeout (default 30s) — cloud→stats.nba.com is often slow
            client = getattr(endpoint, "nba_response", None) or getattr(endpoint, "client", None)
            if client is not None and hasattr(client, "timeout"):
                client.timeout = NBA_API_TIMEOUT
            # Alternative: patch http module if available
            try:
                from nba_api.stats.library import http as nba_http
                if hasattr(nba_http, "NBAStatsHTTP"):
                    nba_http.NBAStatsHTTP.timeout = NBA_API_TIMEOUT
            except Exception:
                pass
            df = endpoint.get_data_frames()[0]
            if df is not None and not df.empty:
                return df
        except Exception as e:
            last_error = e
            if attempt < NBA_API_RETRIES:
                time.sleep(5)  # backoff before retry
    logger.warning("NBA API fetch failed after %d attempt(s): %s", NBA_API_RETRIES, last_error)
    return None


def seed_database(chroma_client: Optional[chromadb.Client] = None) -> int:
    """
    Fetch active NBA players, compute 8D vectors, seed ChromaDB.
    Returns number of players seeded. Idempotent (wipes collection first).
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
    except ImportError:
        logger.warning("nba_api not installed; using fallback seed.")
        return _seed_fallback(chroma_client) if chroma_client else 0

    df = _fetch_nba_data()
    if df is None or df.empty:
        logger.warning("No player data returned.")
        return _seed_fallback(chroma_client) if chroma_client else 0

    # Filter: minimum games played
    df = df[df["GP"] >= 5].reset_index(drop=True)
    if df.empty:
        return _seed_fallback(chroma_client) if chroma_client else 0

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
        return _seed_fallback(chroma_client) if chroma_client else 0

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
