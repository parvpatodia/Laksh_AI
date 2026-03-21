import os
import json
import time
import uuid
import base64
import io
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from google.genai.errors import APIError
import chromadb
from gtts import gTTS

from physics_engine import KinematicAnalyzer
from correction_engine import generate_correction_video

# Google Cloud TTS (Studio Voices) — optional; falls back to gTTS if credentials unavailable
try:
    from google.cloud import texttospeech
    _tts_client = texttospeech.TextToSpeechClient()
    _tts_available = True
except Exception:
    _tts_client = None
    _tts_available = False

logger = logging.getLogger(__name__)

COLLECTION_NAME = "apex_oracle_v7"
# Use absolute path: ./chroma_db is ambiguous when cwd differs in deployment (Railway, Docker)
PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# Module-level client and collection — populated in lifespan startup
chroma_client = None
_collection = None


def _get_collection():
    """Return the live collection; raises clearly if lifespan startup hasn't run yet."""
    if _collection is None:
        raise RuntimeError("ChromaDB collection not initialised — lifespan startup may have failed")
    return _collection


def calculate_market_index(vector: list[float], match_distance: float) -> str:
    """
    Deterministic valuation from cosine distance to nearest NBA pro.
    ChromaDB with hnsw:space='cosine' returns cosine distance in [0, 2]
    (1 − cosine_similarity). For well-matched weighted 8D vectors:
      0.00–0.15 = near-identical mechanics  (Elite)
      0.15–0.35 = strong stylistic overlap  (D1 Prospect)
      0.35–0.55 = moderate similarity       (High School Elite)
      0.55–0.75 = weak overlap              (Developmental)
      0.75+     = divergent mechanics       (Amateur)
    """
    if match_distance < 0.15:
        return "$1.2M - Elite Tier"
    if match_distance < 0.35:
        return "$450k - D1 Prospect"
    if match_distance < 0.55:
        return "$180k - High School Elite"
    if match_distance < 0.75:
        return "$45k - Developmental"
    return "$8k - Amateur"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise ChromaDB on startup; yield for request handling; clean up on shutdown."""
    _init_chroma()
    yield


# CORS: Production URL + local dev. Set CORS_ORIGINS env to override (comma-separated).
_DEFAULT_ORIGINS = [
    "https://lakshai-production.up.railway.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]
_CORS_ORIGINS = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", ",".join(_DEFAULT_ORIGINS)).split(",") if o.strip()
]

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

_DASHBOARD = Path(__file__).resolve().parent / "dashboard.html"


def _init_chroma():
    """Initialise ChromaDB client and collection. Tries PERSIST_DIR first, then /tmp (for read-only deploy fs)."""
    global chroma_client, _collection
    import shutil

    sqlite_path = os.path.join(PERSIST_DIR, "chroma.sqlite3")
    db_healthy = os.path.exists(sqlite_path) and os.path.getsize(sqlite_path) > 0
    logger.info("ChromaDB health check: db_healthy=%s, path=%s", db_healthy, PERSIST_DIR)

    if not db_healthy:
        logger.info("DATABASE NOT FOUND OR CORRUPT — wiping and rebuilding…")
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        os.makedirs(PERSIST_DIR, exist_ok=True)

    for path in [PERSIST_DIR, os.path.join(os.environ.get("TMPDIR", "/tmp"), "apex_chroma")]:
        try:
            os.makedirs(path, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=path)
            logger.info("ChromaDB initialised at %s", path)
            break
        except Exception as e:
            logger.warning("ChromaDB PersistentClient failed at %s: %s", path, e)
    else:
        raise RuntimeError("ChromaDB could not initialise — try CHROMA_PERSIST_DIR env var or check disk")

    if not db_healthy:
        # Let seed_database own collection creation (it deletes + recreates internally).
        # Fetch the live reference AFTER seeding so _collection points at the final UUID.
        try:
            from db_seeder import seed_database
            count = seed_database(chroma_client)
            logger.info("DATABASE SEEDING COMPLETE: %d players indexed.", count)
        except Exception as e:
            logger.warning("DB seed failed: %s", e)

    # Always resolve _collection from the client — this is the single source of truth
    # and survives seed_database's internal delete+recreate cycle.
    # Use get_or_create so deployment survives NBA API seeding failures (rate limits,
    # network blocks, ephemeral FS). Empty collection = no pro match, but app stays healthy.
    try:
        _collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        _collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Collection '%s' created empty (seeding may have failed).", COLLECTION_NAME)
    # Log count separately — don't let logging failures wipe _collection (fixes deployment crash)
    try:
        n = getattr(_collection, "count", None)
        cnt = n() if callable(n) else (n if isinstance(n, int) else "?")
        logger.info("Collection '%s' ready (%s items).", COLLECTION_NAME, cnt)
        # Single-line health summary for Railway logs — grep for "ChromaDB health"
        logger.info("ChromaDB health: OK | pre_seeded=%s | players=%s", db_healthy, cnt)
    except Exception:
        logger.info("Collection '%s' ready.", COLLECTION_NAME)


@app.get("/")
def root():
    return FileResponse(_DASHBOARD) if _DASHBOARD.exists() else {"status": "Apex Oracle Engine Active", "docs": "/docs"}


@app.get("/api")
def api_status():
    return {"status": "Apex Oracle Engine Active", "docs": "/docs"}


@app.get("/health")
def health():
    """
    Liveness probe for Railway / load balancers.
    Returns 200 if ChromaDB is ready; 503 if not.
    """
    try:
        coll = _get_collection()
        n = getattr(coll, "count", None)
        cnt = n() if callable(n) else (n if isinstance(n, int) else 0)
        return {
            "status": "ok",
            "chroma_ready": True,
            "collection_count": cnt,
        }
    except Exception as e:
        logger.warning("Health check failed: %s", e)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

# 8D vector schema (db_seeder v7 = physics_engine output = query_vector in /analyze-video):
# [release_velocity_mps, shot_arc_deg, knee_angle, elbow_angle, kinetic_sync_ms, fluidity_score, hip_rotation_deg, balance_index]

ORACLE_SCHEMA = {
    "type": "object",
    "properties": {
        "athlete_action": {"type": "string"},
        "stats": {
            "type": "object",
            "properties": {
                "release_velocity_mps": {"type": "number"},
                "shot_arc_deg": {"type": "number"},
                "market_index": {"type": "string"},
                "fluidity_score": {"type": "integer"},
            },
        },
        "scout_report": {"type": "string"},
        "athlete_feedback": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "category": {"type": "string"},
                    "observation": {"type": "string"},
                },
            },
        },
        "witty_catchphrase": {"type": "string"},
    },
    "required": ["athlete_action", "scout_report", "athlete_feedback", "witty_catchphrase"],
}


def _build_matched_pro(pro_name: str, player_id: Optional[int], meta: Optional[dict]) -> dict:
    """Build matched_pro object with name, image_url, and vector_stats (8 metrics).

    Schema (aligned with db_seeder v7 and physics_engine output):
      v0: release_velocity_mps  v1: shot_arc_deg   v2: knee_angle    v3: elbow_angle
      v4: kinetic_sync_ms       v5: fluidity_score v6: hip_rotation_deg v7: balance_index
    """
    image_url = None
    if player_id:
        image_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
    placeholder_silhouette = "https://lh3.googleusercontent.com/aida-public/AB6AXuAEYJtsjv0AVFJB2ba3U3KmEnL2mtWNkuzq2nhX4hWD30oy21DFwauSzmtFKRJ6r0ut_FxC9-MEjtmtIo1QRG_Ee485R2wiy_e4Q_sA8cMUlKpIPhjrwT3ZRwD6AvO4dvktAkSVmbLxuco8UsagMr0Ph0S0o6KzXTcXIpsYfQMOOFjX7zdlc_vD2p-zDv9QV5fqikJf1uG7gLsbX0f9OCNIHd32DeGv1u6tr1CmfPgTO7Ypq6xtpnw76ayjNdNbdYzTRxd-fTVcI8w"
    if not image_url:
        image_url = placeholder_silhouette

    vec = meta or {}
    vector_stats = {
        "release_velocity_mps": round(float(vec.get("v0", 7.0)), 2),
        "shot_arc_deg": round(float(vec.get("v1", 45.0)), 1),
        "knee_angle": round(float(vec.get("v2", 150.0)), 1),
        "elbow_angle": round(float(vec.get("v3", 165.0)), 1),
        "kinetic_sync_ms": round(float(vec.get("v4", 300.0)), 1),
        "fluidity_score": int(round(float(vec.get("v5", 75.0)))),
        "hip_rotation_deg": round(float(vec.get("v6", 5.0)), 2),
        "balance_index": int(round(float(vec.get("v7", 75.0)))),
    }
    return {
        "name": pro_name,
        "image_url": image_url,
        "vector_stats": vector_stats,
    }


def _normalize_analysis(
    data: dict,
    biomech: dict,
    market_index: str,
    pro_match: str,
    matched_pro: Optional[dict] = None,
) -> dict:
    stats = data.get("stats") or {}
    feedback = data.get("athlete_feedback") or []
    if not isinstance(feedback, list):
        feedback = [{"timestamp": "", "category": "general", "observation": str(feedback)}]
    telemetry = biomech.get("telemetry") or {}
    vq = telemetry.get("video_quality") or {}
    return {
        **data,
        "athlete_action": data.get("athlete_action") or "—",
        "witty_catchphrase": data.get("witty_catchphrase") or "",
        "stats": {
            "release_velocity_mps": biomech.get("release_velocity_mps"),
            "shot_arc_deg": biomech.get("shot_arc_deg"),
            "knee_angle": biomech.get("knee_angle"),
            "elbow_angle": biomech.get("elbow_angle"),
            "knee_angle_uncertainty": biomech.get("knee_angle_uncertainty"),
            "elbow_angle_uncertainty": biomech.get("elbow_angle_uncertainty"),
            "kinetic_sync_ms": biomech.get("kinetic_sync_ms"),
            "hip_rotation_deg": biomech.get("hip_rotation_deg"),
            "balance_index": biomech.get("balance_index"),
            "market_index": market_index,
            "fluidity_score": biomech.get("fluidity_score"),
        },
        "scout_report": data.get("scout_report") or "—",
        "athlete_feedback": feedback,
        "pro_match": pro_match,
        "matched_pro": matched_pro,
        "telemetry": telemetry,
        "video_quality_score": vq.get("video_quality_score"),
        "video_quality_label": vq.get("video_quality_label"),
        "confidence_factors": telemetry.get("confidence_factors", []),
    }


@app.post("/analyze-video")
async def analyze_video(
    video: UploadFile = File(...),
    start_sec: Optional[str] = Form(None),
    end_sec: Optional[str] = Form(None),
    athlete_name: Optional[str] = Form(None),
    sport: Optional[str] = Form(None),
):
    """Analyze video. Optional start_sec/end_sec restrict to user-selected clip (single shot)."""
    safe_name = os.path.join(tempfile.gettempdir(), f"laksh_{uuid.uuid4()}.mp4")
    with open(safe_name, "wb") as b:
        b.write(await video.read())
    try:
        start_val = None
        end_val = None
        if start_sec is not None and str(start_sec).strip():
            try:
                start_val = float(start_sec)
            except (TypeError, ValueError):
                pass
        if end_sec is not None and str(end_sec).strip():
            try:
                end_val = float(end_sec)
            except (TypeError, ValueError):
                pass
        biomech = KinematicAnalyzer(safe_name).analyze(start_sec=start_val, end_sec=end_val)

        # Query ChromaDB BEFORE Gemini so we can compute deltas for the prompt.
        # Weights must mirror db_seeder.FEATURE_WEIGHTS exactly so query and index
        # live in the same normalised L2 space.
        FEATURE_WEIGHTS = [16.6, 3.3, 1.25, 1.66, 0.33, 1.66, 2.22, 2.0]
        raw_vector = [
            float(biomech.get("release_velocity_mps", 7.0)),
            float(biomech.get("shot_arc_deg", 45.0)),
            float(biomech.get("knee_angle", 145.0)),
            float(biomech.get("elbow_angle", 165.0)),
            float(biomech.get("kinetic_sync_ms", 150.0)),
            float(biomech.get("fluidity_score", 65.0)),
            float(biomech.get("hip_rotation_deg", 5.0)),
            float(biomech.get("balance_index", 85.0)),
        ]
        query_vector = [v * w for v, w in zip(raw_vector, FEATURE_WEIGHTS)]

        collection = _get_collection()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        match_name = "—"
        meta = {}
        match_distance = 999.0
        confidence_score = 88.5

        # ChromaDB returns [[]] for empty collections — check inner list before indexing
        try:
            dists = results.get("distances") if results else []
            docs = results.get("documents") if results else []
            metas = results.get("metadatas") if results else []
            if dists and len(dists) > 0 and len(dists[0]) > 0:
                distance = dists[0][0]
                match_distance = float(distance)
                # Cosine distance ∈ [0, 2]; 0.0 = identical, ~1.0 = orthogonal.
                # Map to 0–100% confidence: 0.0→100%, 1.0→0% (clamped).
                confidence_score = round(max(0.0, min(100.0, 100.0 - (distance * 100))), 1)
            if docs and len(docs) > 0 and len(docs[0]) > 0:
                match_name = str(docs[0][0])
            if metas and len(metas) > 0 and len(metas[0]) > 0 and metas[0][0]:
                meta = dict(metas[0][0])
        except (IndexError, TypeError, KeyError) as e:
            logger.warning("ChromaDB parsing error: %s", e)

        player_id = meta.get("id") or meta.get("player_id")
        matched_pro = _build_matched_pro(match_name, player_id, meta) if match_name != "—" else None
        market_index = calculate_market_index(query_vector, match_distance)

        # Build pro_stats from meta (v0-v7) for delta calculation.
        # Schema: v0=vel_mps, v1=arc, v2=knee, v3=elbow, v4=ksync_ms, v5=fluidity, v6=hip, v7=balance
        pro_stats = {}
        if isinstance(meta, dict) and meta:
            try:
                pro_stats["release_velocity"] = round(float(meta.get("v0", 7.0)), 2)
                pro_stats["shot_arc"] = round(float(meta.get("v1", 45.0)), 1)
                pro_stats["knee_angle"] = round(float(meta.get("v2", 150.0)), 1)
                pro_stats["elbow_angle"] = round(float(meta.get("v3", 165.0)), 1)
                pro_stats["kinetic_sync_ms"] = round(float(meta.get("v4", 300.0)), 1)
                pro_stats["fluidity_score"] = int(round(float(meta.get("v5", 75.0))))
                pro_stats["hip_rotation_deg"] = round(float(meta.get("v6", 5.0)), 2)
                pro_stats["balance_index"] = int(round(float(meta.get("v7", 75.0))))
            except (TypeError, ValueError):
                pro_stats = {}

        user_stats = {
            "shot_arc_deg": biomech.get("shot_arc_deg"),
            "release_velocity_mps": biomech.get("release_velocity_mps"),
            "knee_flexion_at_dip": biomech.get("knee_angle"),
            "elbow_flexion_at_release": biomech.get("elbow_angle"),
            "kinetic_sync_ms": biomech.get("kinetic_sync_ms"),
            "balance_index": biomech.get("balance_index"),
            "hip_rotation_deg": biomech.get("hip_rotation_deg"),
            "fluidity_score": biomech.get("fluidity_score"),
        }

        deltas = {}
        if pro_stats:
            try:
                deltas["arc_gap"] = round(pro_stats.get("shot_arc", 45) - (user_stats.get("shot_arc_deg") or 45), 1)
                deltas["vel_gap"] = round(pro_stats.get("release_velocity", 7.0) - (user_stats.get("release_velocity_mps") or 7.0), 2)
                deltas["knee_gap"] = round(pro_stats.get("knee_angle", 150) - (user_stats.get("knee_flexion_at_dip") or 150), 1)
                deltas["elbow_gap"] = round(pro_stats.get("elbow_angle", 165) - (user_stats.get("elbow_flexion_at_release") or 165), 1)
                deltas["fluid_gap"] = round(pro_stats.get("fluidity_score", 80) - (user_stats.get("fluidity_score") or 80), 1)
                deltas["hip_gap"] = round(pro_stats.get("hip_rotation_deg", 5) - (user_stats.get("hip_rotation_deg") or 5), 1)
                deltas["ksync_gap"] = round(pro_stats.get("kinetic_sync_ms", 15) - (user_stats.get("kinetic_sync_ms") or 15), 1)
                deltas["bal_gap"] = round(pro_stats.get("balance_index", 80) - (user_stats.get("balance_index") or 80), 1)
            except Exception:
                deltas = {"error": "Delta calc failed"}

        athlete_label = (athlete_name or "").strip() or "the athlete"
        prompt = f"""
Act as an elite NBA Biomechanics Director with PhD-level expertise. Authoritative tone. Focus ruthlessly on causality (how input distortions affect output numbers).

Athlete: {athlete_label}
Oracle Match (nearest NBA pro by kinematic fingerprint): {match_name}.

USER STATS: {json.dumps(user_stats)}
PRO BASELINE: {json.dumps(pro_stats)}
KINEMATIC DELTAS (Pro minus User): {json.dumps(deltas)}

FORMATTING RULES:
- Wrap key numbers and recommendations in **double asterisks** for emphasis (e.g., **45° arc** or **increase knee flexion by 12°**).
- Maintain authoritative, PhD-level biomechanics expertise. Focus on causality.

Write the `scout_report` (technical overview) and `athlete_feedback`.
CRITICAL: The `athlete_feedback` array MUST contain exactly 3 items. They must strictly focus on closing the mathematical gaps in KINEMATIC DELTAS. Explain HOW each biomechanical difference causes outcome differences. Give tangible drills. Do not invent stats.

REQUIRED: Add `witty_catchphrase` — a short (max 8 words), fun, player-specific or basketball-trendy line based on the matched player. Examples: "Splash zone unlocked" or "Step-back energy, Trae-style."
"""

        try:
            video_file = client.files.upload(file=safe_name)
            while getattr(getattr(video_file, "state", None), "name", None) == "PROCESSING":
                time.sleep(1)
                video_file = client.files.get(name=video_file.name)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[video_file, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=ORACLE_SCHEMA),
            )
        except APIError as e:
            status = getattr(e, "code", 503)
            msg = "Analysis service temporarily unavailable. Please try again in a moment."
            if status == 429:
                msg = "Rate limit exceeded. Please try again later."
            raise HTTPException(status_code=min(status, 503), detail=msg)
        except Exception as e:
            raise HTTPException(status_code=503, detail="Analysis service error. Please try again.")

        data = json.loads(response.text)
        data["kinematic_deltas"] = deltas

        out = _normalize_analysis(data, biomech, market_index, match_name, matched_pro)
        out["athlete_name"] = (athlete_name or "").strip() or "Athlete"
        out["sport"] = sport or "basketball"
        # Reduce confidence when multiple people detected (improves pro-match reliability)
        det = (biomech.get("telemetry") or {}).get("detection_metadata") or {}
        if det.get("people_detected_max", 1) > 1:
            confidence_score = round(confidence_score * 0.85, 1)  # 15% penalty
        # Reduce confidence when validation warnings exist (video quality, pose visibility, etc.)
        vw = (biomech.get("telemetry") or {}).get("validation_warnings") or []
        if vw:
            confidence_score = round(confidence_score * max(0.7, 1.0 - len(vw) * 0.03), 1)
        out["confidence"] = confidence_score
        out["detection_metadata"] = det
        out["validation_warnings"] = vw
        # Phase 2: video quality, confidence factors for transparent attribution
        tel = biomech.get("telemetry") or {}
        vq = tel.get("video_quality") or {}
        out["video_quality_score"] = vq.get("video_quality_score")
        out["video_quality_label"] = vq.get("video_quality_label")
        out["confidence_factors"] = tel.get("confidence_factors") or []
        return out
    finally:
        if os.path.exists(safe_name):
            os.remove(safe_name)


def _placeholder_card_svg(match: str, score) -> str:
    safe = (match or "Prospect")[:40].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="400" height="533" viewBox="0 0 400 533">
      <rect width="400" height="533" fill="#0a0a0a" stroke="#334155" stroke-width="2"/>
      <text x="200" y="180" text-anchor="middle" fill="#94a3b8" font-family="sans-serif" font-size="14">FLUIDITY {score}</text>
      <text x="200" y="220" text-anchor="middle" fill="#64748b" font-family="sans-serif" font-size="12">{safe}</text>
      <text x="200" y="280" text-anchor="middle" fill="#475569" font-family="sans-serif" font-size="11">Image generation unavailable</text>
    </svg>"""
    return base64.b64encode(svg.encode("utf-8")).decode("utf-8")


@app.post("/generate-metric-card")
async def generate_metric_card(req: dict):
    # Pure background image — NO TEXT. Text overlays are rendered via CSS in the frontend.
    prompt = (
        "A cinematic, ultra-high definition vertical 9:16 holographic sports card background. "
        "Visual: A glowing neon-cyan wireframe silhouette of a basketball player in mid-jump shot. "
        "NO TEXT. DO NOT GENERATE ANY WORDS OR NUMBERS. "
        "Dark obsidian background, biometric HUD aesthetic, 8k resolution, photorealistic."
    )
    try:
        result = client.models.generate_images(
            model="imagen-4.0-generate-001",
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="9:16",
                output_mime_type="image/jpeg",
            ),
        )
        b64 = base64.b64encode(result.generated_images[0].image.image_bytes).decode("utf-8")
        return {"status": "success", "image_base64": b64}
    except Exception as e:
        print(f"Imagen Generation Error: {e}")
        return {"status": "fallback", "image_base64": _placeholder_card_svg((req or {}).get("match", "Prospect"), 0)}


@app.post("/generate-audio-brief")
async def generate_audio_brief(body: dict = None):
    body = body or {}
    text = (body.get("text") or body.get("evaluation") or "").strip()
    if not text:
        return {"status": "error", "message": "Missing 'text' in request body."}
    try:
        if _tts_available and _tts_client:
            # Expert TTS Configuration for hyper-natural, fast cadence
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Studio-O",  # Premium neural voice model
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.35,  # 35% faster for authoritative, rapid-fire engineering brief
                pitch=-1.5,  # Slightly deepen for professional scout authority
            )
            synthesis_input = texttospeech.SynthesisInput(text=text)
            response = _tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )
            audio_bytes = response.audio_content
        else:
            # Fallback to gTTS when Cloud TTS credentials unavailable
            tts = gTTS(text=text, lang="en", tld="co.uk")
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_bytes = fp.read()
        return {"status": "success", "audio_base64": base64.b64encode(audio_bytes).decode("utf-8")}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/generate-correction-video")
async def generate_correction_video_endpoint(
    video:            UploadFile = File(None),
    telemetry_json:   str        = Form(...),
    stats_json:       str        = Form(...),
    athlete_name:     str        = Form("Athlete"),
    kinematic_deltas_json: str   = Form("{}"),
    sport:            str        = Form("basketball"),
    pro_match:        str        = Form(""),
):
    """
    Phase 1 correction video endpoint.

    Accepts multipart/form-data:
      video                 — original uploaded video file (optional but strongly recommended)
      telemetry_json        — JSON string of telemetry block from /analyze-video
      stats_json            — JSON string of 8D metrics block
      athlete_name          — athlete display name
      kinematic_deltas_json — JSON string of kinematic_deltas (enables pro-matched correction)
      sport                 — sport ID (basketball / tennis / golf)
      pro_match             — matched pro name for subtitle

    Returns: { "status": "success", "video_base64": "<base64-mp4>" }
    """
    try:
        telemetry        = json.loads(telemetry_json or "{}")
        stats            = json.loads(stats_json or "{}")
        kinematic_deltas = json.loads(kinematic_deltas_json or "{}") or None
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Invalid JSON in form fields: {e}"}

    if not telemetry:
        return {"status": "error", "message": "Missing telemetry data."}

    # Save uploaded video to a temp file so OpenCV can read it frame-by-frame
    video_path = None
    tmp_video_path = None
    if video is not None:
        try:
            tmp_video_fd, tmp_video_path = tempfile.mkstemp(
                suffix=os.path.splitext(video.filename or ".mp4")[1] or ".mp4",
                dir=tempfile.gettempdir(),
            )
            with os.fdopen(tmp_video_fd, "wb") as f:
                content = await video.read()
                f.write(content)
            video_path = tmp_video_path
        except Exception as e:
            logger.warning("Could not save uploaded video for correction engine: %s", e)
            video_path = None

    try:
        video_bytes = generate_correction_video(
            telemetry, stats,
            athlete_name     = (athlete_name or "Athlete").strip() or "Athlete",
            kinematic_deltas = kinematic_deltas,
            sport            = sport or "basketball",
            pro_match        = pro_match or None,
            video_path       = video_path,
        )
        if not video_bytes:
            return {"status": "error", "message": "Correction video could not be rendered (no pose frames available)."}
        return {
            "status": "success",
            "video_base64": base64.b64encode(video_bytes).decode("utf-8"),
        }
    except Exception as e:
        logger.exception("Correction video generation failed")
        return {"status": "error", "message": str(e)}
    finally:
        if tmp_video_path and os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)
