import asyncio
import concurrent.futures
import os
import json
import re
import subprocess
import time
import uuid
import base64
import io
import logging
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from google import genai
from google.genai import types
from google.genai.errors import APIError
import chromadb
from gtts import gTTS

from app.api_contract import API_SCHEMA_VERSION
from app.api.v1 import router as v1_router
from app.logging_config import configure_logging
from app.physics_engine import KinematicAnalyzer
from app.correction_engine import generate_correction_video

# A6: Git SHA read from environment variable set at Docker build time.
# Do NOT use subprocess.run(["git", ...]) — git may not be installed in the
# production container and the .git directory is not mounted in Fly.io images.
_GIT_COMMIT_SHA: str = os.environ.get("GIT_COMMIT_SHA", "unknown")

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
# Repo root (parent of app/) — chroma_db lives next to requirements.txt, not inside app/
_REPO_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(_REPO_ROOT / "chroma_db")

# ── Confidence scoring constants ─────────────────────────────────────────────
# These are multiplicative engineering heuristics, NOT statistical
# probabilities. They compound: final_confidence = base × Π(penalties).
# Documented in evaluation/calibration_evidence_v0/basketball_literature_v0.md.
#
# Base confidence = (100 - cosine_distance × 100), clamped [0, 100].

CONF_MULTI_PERSON_FACTOR = 0.85
"""15% penalty when >1 person detected. Multi-person scenes cause
landmark swapping and unstable joint traces."""

CONF_WARNING_PENALTY_PER = 0.03
"""Per-validation-warning penalty (max total penalty capped at 30%).
Each warning (e.g. short clip, low fps, bad aspect) degrades the
input quality that metrics depend on."""

CONF_WARNING_FLOOR = 0.70
"""Floor multiplier for validation-warning penalties so confidence
never drops below 70% of its pre-warning value from warnings alone."""

CONF_RELIABILITY_BASE = 0.55
"""Minimum reliability multiplier even when all metrics are unavailable.
The remaining 0.45 is weighted by metric confidence + availability."""

CONF_RELIABILITY_METRIC_WEIGHT = 0.60
"""Within the reliability band (0.45), this fraction weights per-metric
confidence (from physics_engine._calibrate_metric_confidence)."""

CONF_RELIABILITY_AVAIL_WEIGHT = 0.40
"""Within the reliability band, this fraction weights the ratio of
available (non-unavailable) metrics to total metrics."""

CONF_PARTIAL_MODE_FACTOR = 0.90
"""10% penalty for partial analysis mode (enough detections to
compute some metrics but not all)."""

CONF_FALLBACK_CAP = 25.0
"""Hard cap on confidence in fallback mode (< 3 frames or < 2
detections). Prevents misleading high confidence on garbage input."""

CONF_DEGRADED_ORACLE_FACTOR = 0.88
"""12% penalty when oracle match is degraded (partial mode, ≥3
predicted metrics, or low mean metric confidence)."""

CONF_ORACLE_PREDICTED_THRESHOLD = 3
"""If this many or more metrics are 'predicted' (not measured),
the oracle match is marked degraded."""

CONF_ORACLE_MEAN_MC_THRESHOLD = 0.52
"""If mean metric confidence is below this, the oracle match is
marked degraded even if no individual metric is predicted."""

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
    """Configure logging, initialise ChromaDB, warm pose landmarker; yield for request handling."""
    configure_logging()
    _init_chroma()
    threading.Thread(target=_warm_pose_landmarker, daemon=True).start()
    yield


# CORS: explicit allowlist for stable origins, plus a regex that matches any
# Vercel deployment URL for the laksh-ai project.
#
# Why the regex: Vercel assigns each production deployment its own hashed
# subdomain (e.g. https://laksh-im4hx7f36-laksh-ai.vercel.app) in addition to
# the stable alias (https://laksh-ai.vercel.app). Without the regex we'd have
# to rotate CORS_ORIGINS on every deploy. Pattern is scoped to the laksh-ai
# project name so it does not allow arbitrary *.vercel.app sites.
_DEFAULT_ORIGINS = [
    "https://lakshai-production.up.railway.app",
    "https://laksh-ai.vercel.app",
    "https://laksh-ai-tawny.vercel.app",
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
# Matches stable alias, per-deployment URLs, and team preview hostnames.
_VERCEL_PREVIEW_REGEX = (
    r"^https://laksh-ai\.vercel\.app$"
    r"|^https://laksh(-[a-z0-9]+)+-laksh-ai\.vercel\.app$"
    r"|^https://laksh-ai-[a-z0-9-]+\.vercel\.app$"
    r"|^https://[a-z0-9-]+-laksh-ai\.vercel\.app$"
)

app = FastAPI(lifespan=lifespan)


class CrossOriginResourcePolicyMiddleware(BaseHTTPMiddleware):
    """Public API responses must be embeddable from COEP pages if we ever re-enable COEP on the web app."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return response


# CORP outermost, then CORS — so preflight and JSON/video responses all get ACAO + CORP.
app.add_middleware(CrossOriginResourcePolicyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=_VERCEL_PREVIEW_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Versioned API surface. All new clients should target /v1/*.
# The legacy /analyze-video route below remains in place for one release
# cycle while the v2-adapter for basketball lands.
app.include_router(v1_router)

# Fly often sets GOOGLE_API_KEY; local dev may use GEMINI_API_KEY — GenAI client needs one.
# Both spellings are checked so neither key naming convention breaks.
_genai_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not _genai_key:
    # This is a hard error that will cause 401 on every request; log loudly so it
    # is the FIRST thing visible in fly logs / docker logs rather than buried in traces.
    logger.error(
        "CRITICAL: No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY "
        "via `fly secrets set -a laksh-api 'GEMINI_API_KEY=...'`. "
        "All /analyze-video calls will fail with 401 until this is set."
    )
client = genai.Client(api_key=_genai_key)

_DASHBOARD = _REPO_ROOT / "static" / "dashboard.html"


def _fast_video_precheck(path: str) -> tuple[bool, str, dict]:
    """Cheap OpenCV-only pre-validation before committing to MediaPipe.

    Returns (ok, reason_code, actuals).  Uses only cv2 — no MediaPipe init.
    Rejects obviously bad inputs in < 0.3 s so the 25-40 s analysis budget is
    not burned on unreadable or pathologically short clips.

    Thresholds are deliberately lenient: the purpose is to catch corrupted files
    and clips where no biomechanics could possibly be computed (< 1 s, < 10 fps).
    We do NOT mirror the full A4 visibility/in-frame thresholds here because those
    require landmark data that only MediaPipe can produce.
    """
    import cv2 as _cv2
    try:
        cap = _cv2.VideoCapture(path)
        if not cap.isOpened():
            return False, "video_unreadable", {}
        fps = cap.get(_cv2.CAP_PROP_FPS) or 0.0
        total = cap.get(_cv2.CAP_PROP_FRAME_COUNT)  # may be -1 or 0 for WebM VP9
        # Read at least one frame to confirm the bitstream is valid.
        ok_read, _ = cap.read()
        cap.release()
        if not ok_read:
            return False, "video_no_frames", {}
        if fps < 10.0:
            return False, "preflight_fps_failed", {"fps_observed": round(fps, 1), "fps_floor": 10.0}
        # Duration check: skip when CAP_PROP_FRAME_COUNT returns 0/-1 (WebM VP9
        # containers from MediaRecorder do NOT embed frame-count metadata).
        # Falsely rejecting a valid clip as "too short" is worse than letting
        # the full pipeline handle it — the pipeline checks n_frames >= 3 itself.
        if fps > 0 and total > 0:
            duration_s = total / fps
            if duration_s < 1.0:
                return False, "preflight_too_short", {"duration_s": round(duration_s, 2), "min_duration_s": 1.0}
        duration_s = (total / fps) if (fps > 0 and total > 0) else None
        return True, "", {"fps_observed": round(fps, 1), "duration_s": round(duration_s, 1) if duration_s else "unknown"}
    except Exception as exc:
        logger.warning("Fast video pre-check failed: %s", exc)
        return True, "", {}  # On unexpected error: allow through; full analysis handles it


# Human-readable hints for each preflight/fallback failure code.
_PREFLIGHT_HINTS: dict[str, str] = {
    "video_unreadable": "The video file could not be decoded. Try re-recording or using a different format (MP4/H.264).",
    "video_no_frames": "No valid frames found in the video. Try re-recording.",
    "preflight_fps_failed": "Video frame rate is too low for reliable biomechanics. Record at 30 fps (most phone cameras default to this).",
    "preflight_too_short": "Clip is too short. Record at least 2-3 seconds of your shooting motion.",
    "low_detections": "Very few body landmarks were detected. Ensure your full upper body is visible and well-lit.",
    "low_visibility": "Landmark visibility was low throughout the clip. Move to a well-lit area and ensure your body is not obscured.",
    "short_clip": "Clip is too short for a complete shot-cycle analysis. Record a full jump-shot motion.",
    "decode_error": "Video decode failed. Try re-recording or converting to MP4.",
    "pose_init_failed": "Pose engine failed to initialize. This is a server issue — please retry.",
    "analysis_exception": "An unexpected analysis error occurred. Please retry; if the problem persists, try a different clip.",
}


def _warm_pose_landmarker() -> None:
    """Pre-load the MediaPipe pose model file into OS disk cache.

    Called in a daemon thread at startup. Failure is non-fatal -- the first
    video request will pay the cold-start cost instead.
    """
    try:
        from app.pose.mediapipe_common import create_pose_landmarker
        lm = create_pose_landmarker()
        lm.close()
        logger.info("[app.main] pose landmarker warm-load complete")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[app.main] pose landmarker warm-load failed: %s", exc)


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
            # Delete *contents* only -- the directory itself may be a mounted
            # volume (Fly.io block device) and rmtree on the mountpoint raises
            # OSError EBUSY. Keep the root directory, clear everything inside it.
            for _child in Path(PERSIST_DIR).iterdir():
                if _child.is_dir():
                    shutil.rmtree(_child)
                else:
                    _child.unlink()
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
            from app.db_seeder import seed_database
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
def root(request: Request):
    """API host: browsers → OpenAPI docs; optional legacy SPA via ?legacy_ui=1; JSON via Accept or ?format=json."""
    if request.query_params.get("legacy_ui") == "1" and _DASHBOARD.exists():
        return FileResponse(_DASHBOARD)
    accept = (request.headers.get("accept") or "").lower()
    if "application/json" in accept or request.query_params.get("format") == "json":
        return {
            "service": "laksh-api",
            "v1_health": "/v1/health",
            "openapi": "/docs",
            "legacy_basketball_analyze": "POST /analyze-video",
            "gym_canonical_video": "POST /v1/analyze/gym/video",
            "legacy_ui": "/?legacy_ui=1",
            "note": "GET / in a browser redirects to /docs. Legacy marketing UI: add ?legacy_ui=1",
        }
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/api")
def api_status():
    return {
        "status": "ok",
        "service": "laksh-api",
        "v1_health": "/v1/health",
        "openapi": "/docs",
        "legacy_basketball_analyze": "POST /analyze-video",
        "gym_canonical_video": "POST /v1/analyze/gym/video",
    }


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
            "api_schema_version": API_SCHEMA_VERSION,
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

# SDK-typed schema for GenerateContentConfig (some keys 400 if a raw dict is passed).
try:
    ORACLE_SCHEMA_GENAI: types.Schema | dict = types.Schema.model_validate(ORACLE_SCHEMA)
except Exception as _schema_exc:  # noqa: BLE001
    logger.warning("ORACLE_SCHEMA → types.Schema failed, using dict: %s", _schema_exc)
    ORACLE_SCHEMA_GENAI = ORACLE_SCHEMA


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
    feedback = data.get("athlete_feedback") or []
    if not isinstance(feedback, list):
        feedback = [{"timestamp": "", "category": "general", "observation": str(feedback)}]
    telemetry = biomech.get("telemetry") or {}
    vq = telemetry.get("video_quality") or {}
    metric_status = biomech.get("metric_status") or {}
    # Biomech scalar fields — emitted BOTH at the top level (for the
    # BasketballAnalyzeResponse TypeScript interface which reads them directly)
    # and inside the nested "stats" dict (for ChromaDB delta calculations and
    # backward compatibility). The top-level keys are the authoritative source.
    _biomech_scalars = {
        "release_velocity_mps": biomech.get("release_velocity_mps"),
        "shot_arc_deg": biomech.get("shot_arc_deg"),
        "knee_angle": biomech.get("knee_angle"),
        "elbow_angle": biomech.get("elbow_angle"),
        "knee_angle_uncertainty": biomech.get("knee_angle_uncertainty"),
        "elbow_angle_uncertainty": biomech.get("elbow_angle_uncertainty"),
        "balance_index_uncertainty": biomech.get("balance_index_uncertainty"),
        "fluidity_score_uncertainty": biomech.get("fluidity_score_uncertainty"),
        "hip_rotation_uncertainty": biomech.get("hip_rotation_uncertainty"),
        "kinetic_sync_ms": biomech.get("kinetic_sync_ms"),
        "hip_rotation_deg": biomech.get("hip_rotation_deg"),
        "balance_index": biomech.get("balance_index"),
        "fluidity_score": biomech.get("fluidity_score"),
    }
    out = {
        **data,
        # Top-level scalars: these are what BasketballReport.tsx reads via
        # result.release_velocity_mps etc. (BasketballAnalyzeResponse interface).
        **_biomech_scalars,
        "analysis_mode": biomech.get("analysis_mode") or "full",
        "fallback_reason_codes": biomech.get("fallback_reason_codes") or [],
        "metric_status": metric_status,
        # A4: actionable per-metric hints (only populated when source quality is
        # below "predicted" — tells the user exactly how to improve framing).
        "metric_hints": biomech.get("metric_hints") or {},
        # A4: pose detection quality status from post-analysis check.
        "preflight_status": biomech.get("preflight_status"),
        "preflight_hints": biomech.get("preflight_hints") or [],
        "athlete_action": data.get("athlete_action") or "—",
        "witty_catchphrase": data.get("witty_catchphrase") or "",
        "stats": {
            **_biomech_scalars,
            "market_index": market_index,
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
    ds = biomech.get("debug_summary")
    if ds is not None:
        out["debug_summary"] = ds
    return out


def _temp_video_path_and_gemini_mime(upload: UploadFile, raw: bytes) -> tuple[str, str]:
    """Temp path extension + MIME for Gemini ``files.upload``.

    The browser sends **WebM** (``video/webm``) from MediaRecorder as ``clip.webm``.
    Writing bytes to ``*.mp4`` makes Google's upload endpoint return **400** (type mismatch).
    """
    ct = (upload.content_type or "").split(";")[0].strip().lower()
    fname = (upload.filename or "").lower()

    def _magic() -> tuple[str, str] | None:
        if len(raw) >= 4 and raw[:4] == b"\x1a\x45\xdf\xa3":
            return ".webm", "video/webm"
        if len(raw) >= 12 and raw[4:8] == b"ftyp":
            return ".mp4", "video/mp4"
        return None

    ext_mime: tuple[str, str] | None = None
    if "webm" in ct or fname.endswith(".webm"):
        ext_mime = (".webm", "video/webm")
    elif "mp4" in ct or fname.endswith(".mp4") or fname.endswith(".m4v"):
        ext_mime = (".mp4", "video/mp4")
    elif "quicktime" in ct or fname.endswith(".mov"):
        ext_mime = (".mov", "video/quicktime")
    elif ct.startswith("video/"):
        if "mp4" in ct or "mpeg4" in ct:
            ext_mime = (".mp4", "video/mp4")
        elif "webm" in ct:
            ext_mime = (".webm", "video/webm")

    if ext_mime is None:
        ext_mime = _magic()
    if ext_mime is None:
        ext_mime = (".webm", "video/webm")

    ext, mime = ext_mime
    path = os.path.join(tempfile.gettempdir(), f"laksh_{uuid.uuid4().hex}{ext}")
    return path, mime


# Common misconfiguration: ``gemini-2.5`` is not a valid API id (must be e.g. ``gemini-2.5-flash``).
_GEMINI_MODEL_ALIASES: dict[str, str] = {
    "gemini-2.5": "gemini-2.5-pro",
    "gemini-2.0": "gemini-2.0-flash",
    "gemini-1.5": "gemini-1.5-pro",
    "gemini-flash": "gemini-2.5-flash",
    "gemini-pro": "gemini-2.5-pro",
}


def _normalize_gemini_model_id(raw: str) -> str | None:
    """Strip shell garbage and map shorthand ids to full model names."""
    s = raw.strip()
    if not s:
        return None
    s = re.sub(r"[^a-zA-Z0-9._-]", "", s)
    if not s:
        return None
    key = s.lower()
    if key in _GEMINI_MODEL_ALIASES:
        canon = _GEMINI_MODEL_ALIASES[key]
        if s != canon:
            logger.warning("Normalized GEMINI model id %r -> %r", raw, canon)
        return canon
    return s


def _oracle_gemini_models() -> list[str]:
    """Models to try in order.

    - ``GEMINI_ORACLE_MODELS`` — comma-separated list (highest priority).
    - ``GEMINI_ORACLE_MODEL`` — single model id.
    - Default — ``gemini-2.5-flash`` then ``gemini-2.0-flash``.

    **Why Flash and not Pro as default:**
    The oracle generates TEXT commentary from structured kinematic JSON that
    MediaPipe already computed.  This is NOT a complex reasoning task that
    requires Pro.  Flash 2.5 produces equivalent commentary quality at ~3-5 s
    vs Pro's ~12-20 s.  That 10+ s per-request saving matters for a live demo
    where MediaPipe already takes 15-30 s.  Judges who want Pro can set
    ``fly secrets set -a laksh-api 'GEMINI_ORACLE_MODEL=gemini-2.5-pro'``.

    **Shell:** always quote values — ``fly secrets set -a APP 'GEMINI_ORACLE_MODEL=gemini-2.5-flash'``.
    A trailing ``;`` starts a new shell command and corrupts the secret.
    """
    raw_m = (os.environ.get("GEMINI_ORACLE_MODELS") or "").strip()
    raw_1 = (os.environ.get("GEMINI_ORACLE_MODEL") or "").strip()
    pieces: list[str] = []
    if raw_m:
        pieces.extend(raw_m.split(","))
    elif raw_1:
        pieces.append(raw_1)
    else:
        pieces = ["gemini-2.5-flash", "gemini-2.0-flash"]

    out: list[str] = []
    for p in pieces:
        n = _normalize_gemini_model_id(p)
        if n:
            out.append(n)
    # Dedupe preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for m in out:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped if deduped else ["gemini-2.5-flash", "gemini-2.0-flash"]


def _gemini_upload_pipeline(src_path: str, fallback_mime: str) -> tuple[object | None, list[str]]:
    """Transcode src_path → MP4, upload to Gemini Files, wait for ACTIVE.

    Designed to run in a ThreadPoolExecutor in parallel with the MediaPipe
    analysis step so the two longest operations overlap rather than serialize.

    Returns:
        (video_file_ref | None, cleanup_paths) — caller must delete cleanup_paths.
    """
    cleanup: list[str] = []
    upload_path = src_path
    upload_mime = fallback_mime

    td = _transcode_for_gemini_upload(src_path)
    if td:
        upload_path, upload_mime = td
        cleanup.append(upload_path)

    try:
        ref = client.files.upload(
            file=upload_path,
            config=types.UploadFileConfig(mime_type=upload_mime),
        )
        deadline = time.time() + 120
        while ref is not None and time.time() < deadline:
            st = getattr(ref, "state", None)
            if getattr(st, "name", None) == "ACTIVE":
                break
            time.sleep(2)
            ref = client.files.get(name=ref.name)
        st = getattr(ref, "state", None)
        if getattr(st, "name", None) != "ACTIVE":
            logger.warning("Gemini file not ACTIVE after wait (state=%s); using text-only oracle", getattr(st, "name", st))
            return None, cleanup
        return ref, cleanup
    except APIError as upload_err:
        logger.warning("Gemini files.upload failed; using text-only oracle. message=%s", getattr(upload_err, "message", None))
        return None, cleanup
    except Exception as exc:
        logger.warning("Gemini upload pipeline error: %s", exc)
        return None, cleanup


def _transcode_for_gemini_upload(src_path: str) -> tuple[str, str] | None:
    """Remux/transcode to H.264 MP4 — Gemini uploads are flaky on arbitrary WebM codecs/containers."""
    dst = os.path.join(tempfile.gettempdir(), f"laksh_gemini_{uuid.uuid4().hex}.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        src_path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        dst,
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, timeout=180)
        if proc.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) < 32:
            tail = (proc.stderr or b"").decode("utf-8", errors="replace")[-800:]
            logger.warning("ffmpeg gemini transcode failed rc=%s stderr_tail=%s", proc.returncode, tail)
            if os.path.exists(dst):
                try:
                    os.remove(dst)
                except OSError:
                    pass
            return None
        return dst, "video/mp4"
    except Exception as exc:
        logger.warning("ffmpeg gemini transcode exception: %s", exc)
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except OSError:
                pass
        return None


def _strip_json_code_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t)
    if m:
        return m.group(1).strip()
    return t


def _extract_json_object_from_text(text: str) -> dict:
    """Parse JSON from model output; tolerate prose or fences around the object."""
    t = _strip_json_code_fence(text or "")
    try:
        out = json.loads(t)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass
    i = t.find("{")
    if i < 0:
        raise ValueError("no JSON object found in model output")
    depth = 0
    for j in range(i, len(t)):
        if t[j] == "{":
            depth += 1
        elif t[j] == "}":
            depth -= 1
            if depth == 0:
                out = json.loads(t[i : j + 1])
                if isinstance(out, dict):
                    return out
                raise ValueError("top-level JSON is not an object")
    raise ValueError("unbalanced braces in model output")


def _generate_oracle_gemini_response(video_part, prompt: str) -> tuple[object, str]:
    """Call Gemini with a latency-optimised, reliability-first attempt order.

    **Design rationale:**

    Schema mode (``response_schema``) frequently returns 400 when the SDK
    serialises the schema in a way the model rejects.  We skip schema mode
    entirely and use ``response_mime_type="application/json"`` which gives us
    structured JSON without the 400-prone schema enforcement.

    **Attempt order per model** (3 attempts, first success wins):
    1. ``video+json`` — model watches the clip AND returns JSON.  Best quality.
       Skipped if video upload failed.
    2. ``text+json``  — kinematic deltas are rich enough; no video needed.
       Primary path when video upload failed or ``video+json`` errors.
    3. ``text+plain`` — last resort if JSON mode 400s (rare with flash).

    **Token budget:** capped at 1 200 tokens.  The oracle needs ~600 tokens
    (scout_report 150 + 3 × feedback 100 + other fields 50).  1 200 gives 2×
    headroom and materially cuts flash latency vs the default unlimited budget.

    *video_part* may be ``None`` if ``files.upload`` failed.
    """
    _MAX_TOKENS = 1200
    # Prompt variant for text-only mode — self-contained with JSON instruction.
    text_prompt = (
        f"{prompt}\n\n"
        "Return ONLY a JSON object (no markdown fences, no extra text). "
        "Required keys: athlete_action (string), stats (object), scout_report (string), "
        "athlete_feedback (array of exactly 3 objects each with timestamp/category/observation), "
        "witty_catchphrase (string ≤8 words)."
    )
    last_err: APIError | None = None

    for model in _oracle_gemini_models():
        attempts: list[tuple[str, list, types.GenerateContentConfig]] = []

        # 1. Video-grounded JSON (best quality — model watches the clip)
        if video_part is not None:
            attempts.append((
                "video+json",
                [video_part, prompt],
                types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=_MAX_TOKENS,
                    temperature=0.1,
                ),
            ))

        # 2. Text-only JSON (always included; primary path when video is unavailable)
        attempts.append((
            "text+json",
            [text_prompt],
            types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=_MAX_TOKENS,
                temperature=0.1,
            ),
        ))

        # 3. Text-only plain (last resort — handles models that 400 on json mime type)
        attempts.append((
            "text+plain",
            [text_prompt],
            types.GenerateContentConfig(
                max_output_tokens=_MAX_TOKENS,
                temperature=0.1,
            ),
        ))

        for tag, contents, cfg in attempts:
            try:
                r = client.models.generate_content(model=model, contents=contents, config=cfg)
                return r, f"{model}/{tag}"
            except APIError as err:
                last_err = err
                logger.warning(
                    "Gemini oracle attempt model=%s tag=%s code=%s message=%s details=%s",
                    model,
                    tag,
                    err.code,
                    getattr(err, "message", None),
                    getattr(err, "details", None),
                )
    assert last_err is not None
    raise last_err


@app.post("/analyze-video")
async def analyze_video(
    video: UploadFile = File(...),
    start_sec: Optional[str] = Form(None),
    end_sec: Optional[str] = Form(None),
    athlete_name: Optional[str] = Form(None),
    sport: Optional[str] = Form(None),
):
    """Analyze video. Optional start_sec/end_sec restrict to user-selected clip (single shot)."""
    raw = await video.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty video upload")
    safe_name, gemini_mime = _temp_video_path_and_gemini_mime(video, raw)
    extra_cleanup: list[str] = []
    with open(safe_name, "wb") as b:
        b.write(raw)
    try:
        # --- A4: Fast pre-check (< 0.3 s, OpenCV only) --------------------------------
        # Catches corrupted/too-short/low-fps inputs before burning 25-40 s of MediaPipe.
        _pre_ok, _pre_code, _pre_actuals = _fast_video_precheck(safe_name)
        if not _pre_ok:
            hint = _PREFLIGHT_HINTS.get(_pre_code, "Please re-record and try again.")
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "preflight_failed",
                    "reason_code": _pre_code,
                    "hint": hint,
                    "actuals": _pre_actuals,
                },
            )

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
        # Run MediaPipe analysis and Gemini upload pipeline in parallel —
        # both are blocking/CPU-bound and completely independent of each other.
        # Sequential execution was the root cause of the 180s client timeout:
        # MediaPipe(~60s) + Gemini_upload(~60s) + generate_content(~45s) = 165s.
        # Parallel: max(60, 60) + 45 = ~105s, well within budget.
        loop = asyncio.get_event_loop()
        _start_sec = start_val
        _end_sec = end_val
        _safe_name = safe_name
        _gemini_mime = gemini_mime

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pool:
            biomech_fut = loop.run_in_executor(
                _pool,
                lambda: KinematicAnalyzer(_safe_name).analyze(start_sec=_start_sec, end_sec=_end_sec),
            )
            gemini_fut = loop.run_in_executor(
                _pool,
                lambda: _gemini_upload_pipeline(_safe_name, _gemini_mime),
            )
            try:
                _biomech_result, _gemini_result = await asyncio.wait_for(
                    asyncio.gather(biomech_fut, gemini_fut, return_exceptions=True),
                    timeout=220,  # 220 s < gunicorn 300 s worker timeout; gives 80 s margin
                )
            except asyncio.TimeoutError:
                logger.error("analyze-video: biomech+gemini exceeded 220 s timeout")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail={
                        "hint": "Analysis timed out. Try a shorter clip (5-10 seconds).",
                        "reason_code": "analysis_timeout",
                    },
                )

        # Handle biomech failures gracefully instead of crashing the endpoint.
        if isinstance(_biomech_result, Exception):
            logger.error("KinematicAnalyzer failed: %s", _biomech_result)
            biomech = {}
        else:
            biomech = _biomech_result

        video_file_ref, _gemini_cleanup = (
            _gemini_result if not isinstance(_gemini_result, Exception) else (None, [])
        )
        extra_cleanup.extend(_gemini_cleanup)

        # --- A4: Post-analysis quality check -----------------------------------------
        # If MediaPipe found too few landmarks to run analysis, surface actionable hints
        # in the response rather than returning bare null/fallback fields.
        # We return 200 (not 422) so the frontend shows a warning state, not an error.
        _analysis_mode = biomech.get("analysis_mode", "full")
        _fallback_codes = biomech.get("fallback_reason_codes") or []
        if _analysis_mode == "fallback" and _fallback_codes:
            # Build hints from the reason codes so the user knows exactly what to fix.
            _pose_hints = [
                _PREFLIGHT_HINTS[c] for c in _fallback_codes if c in _PREFLIGHT_HINTS
            ]
            biomech["preflight_status"] = "pose_detection_failed"
            biomech["preflight_hints"] = _pose_hints or [
                "Ensure your full body is visible, well-lit, and you perform a clear shooting or curl motion."
            ]

        # Query ChromaDB BEFORE Gemini so we can compute deltas for the prompt.
        # Weights must mirror db_seeder.FEATURE_WEIGHTS exactly so query and index
        # live in the same normalised L2 space.
        FEATURE_WEIGHTS = [16.6, 3.3, 1.25, 1.66, 0.33, 1.66, 2.22, 2.0]
        def _num(v, default):
            try:
                if v is None:
                    return float(default)
                return float(v)
            except (TypeError, ValueError):
                return float(default)
        raw_vector = [
            _num(biomech.get("release_velocity_mps"), 7.0),
            _num(biomech.get("shot_arc_deg"), 45.0),
            _num(biomech.get("knee_angle"), 150.0),
            _num(biomech.get("elbow_angle"), 165.0),
            _num(biomech.get("kinetic_sync_ms"), 300.0),
            _num(biomech.get("fluidity_score"), 75.0),
            _num(biomech.get("hip_rotation_deg"), 5.0),
            _num(biomech.get("balance_index"), 75.0),
        ]
        query_vector = [v * w for v, w in zip(raw_vector, FEATURE_WEIGHTS)]

        match_name = "—"
        meta = {}
        match_distance = 999.0
        confidence_score = 88.5

        # ChromaDB query + result parsing — fully wrapped so a DB failure never
        # crashes the request.  On any exception we degrade to no pro-match
        # (match_name="—") and keep the biomech metrics intact.
        try:
            collection = _get_collection()
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )
            # ChromaDB returns [[]] for empty collections — check inner list.
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
        except Exception as chroma_err:
            # DB down, not initialized, or corrupt result — log loudly but
            # do NOT raise.  Biomech data and Gemini oracle still return.
            logger.error(
                "ChromaDB query/init failed — pro match degraded: %s", chroma_err
            )

        player_id = meta.get("id") or meta.get("player_id")
        try:
            matched_pro = _build_matched_pro(match_name, player_id, meta) if match_name != "—" else None
        except Exception as _pro_err:
            # Malformed ChromaDB metadata (non-numeric v0-v7) — degrade gracefully.
            logger.warning("_build_matched_pro failed: %s", _pro_err)
            matched_pro = None
            match_name = "—"
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
                # Only compute a delta when the user's measurement is non-null.
                # If the physics engine returned None (unavailable), we skip that
                # delta entirely so Gemini does not coach on un-measured values.
                def _gap(pro_key, user_key, pro_default, label):
                    user_val = user_stats.get(user_key)
                    if user_val is None:
                        return  # no measurement — suppress the delta
                    deltas[label] = round(float(pro_stats.get(pro_key, pro_default)) - float(user_val), 2)

                _gap("shot_arc", "shot_arc_deg", 45, "arc_gap")
                _gap("release_velocity", "release_velocity_mps", 7.0, "vel_gap")
                _gap("knee_angle", "knee_flexion_at_dip", 150, "knee_gap")
                _gap("elbow_angle", "elbow_flexion_at_release", 165, "elbow_gap")
                _gap("fluidity_score", "fluidity_score", 80, "fluid_gap")
                _gap("hip_rotation_deg", "hip_rotation_deg", 5, "hip_gap")
                _gap("kinetic_sync_ms", "kinetic_sync_ms", 300.0, "ksync_gap")
                _gap("balance_index", "balance_index", 80, "bal_gap")
            except Exception:
                deltas = {"error": "Delta calc failed"}

        athlete_label = (athlete_name or "").strip() or "the athlete"
        stats_available = any(v is not None for v in user_stats.values())

        if not stats_available:
            # Biomech pipeline failed (short clip, occlusion, VFR fallback, etc.).
            # Do NOT send null stats — Gemini will hallucinate "Coaching point N" placeholders.
            # Instead request honest general coaching without quantitative references.
            prompt = f"""
Act as an elite NBA Biomechanics Director.

The video clip for {athlete_label} did not yield measurable biomechanics (clip too short, joint occlusion, or poor lighting).

Write a `scout_report` (1-2 sentences) that honestly acknowledges no quantitative measurements were captured and recommends re-recording with better framing (full body visible, good lighting, 5+ seconds of motion).

Write `athlete_feedback` with exactly 3 universally applicable basketball shooting technique tips — focus on shot arc (target 45-55°), knee drive for vertical power, and wrist snap at release. Do NOT reference any specific numbers from this athlete.

Write `witty_catchphrase` — a short (max 8 words) motivational line.

Respond ONLY with valid JSON, no markdown fences:
{{"scout_report": "...", "athlete_feedback": [{{"title": "...", "feedback": "...", "drill": "..."}}, {{"title": "...", "feedback": "...", "drill": "..."}}, {{"title": "...", "feedback": "...", "drill": "..."}}], "witty_catchphrase": "..."}}
"""
        else:
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

Respond ONLY with valid JSON, no markdown fences:
{{"scout_report": "...", "athlete_feedback": [{{"title": "...", "feedback": "...", "drill": "..."}}, {{"title": "...", "feedback": "...", "drill": "..."}}, {{"title": "...", "feedback": "...", "drill": "..."}}], "witty_catchphrase": "..."}}
"""

        gemini_upload_path = safe_name
        gemini_upload_mime = gemini_mime
        td = _transcode_for_gemini_upload(safe_name)
        if td:
            gemini_upload_path, gemini_upload_mime = td
            extra_cleanup.append(gemini_upload_path)

        video_file_ref = None
        try:
            video_file_ref = client.files.upload(
                file=gemini_upload_path,
                config=types.UploadFileConfig(mime_type=gemini_upload_mime),
            )
            # 30 s is sufficient for a short sports clip; 120 s added too much latency
            # to the total response time and masked upload failures as slow processing.
            deadline = time.time() + 30
            while video_file_ref is not None and time.time() < deadline:
                st = getattr(video_file_ref, "state", None)
                st_name = getattr(st, "name", None) if st is not None else None
                if st_name == "ACTIVE":
                    break
                if st_name == "FAILED":
                    logger.warning("Gemini file processing FAILED server-side; using text-only oracle")
                    video_file_ref = None
                    break
                time.sleep(1)
                video_file_ref = client.files.get(name=video_file_ref.name)
            if video_file_ref is not None:
                st = getattr(video_file_ref, "state", None)
                if getattr(st, "name", None) != "ACTIVE":
                    logger.warning(
                        "Gemini file not ACTIVE after wait (state=%s); using text-only oracle",
                        getattr(st, "name", st),
                    )
                    video_file_ref = None
        except APIError as upload_err:
            logger.warning(
                "Gemini files.upload failed; using text-only oracle. message=%s",
                getattr(upload_err, "message", None),
            )

        # Oracle generation: always return biomech data even when Gemini fails.
        # A Gemini error is logged loudly but does NOT propagate as HTTPException —
        # the frontend receives biomechanical metrics and stub oracle fields instead
        # of an empty 503.  This is the "canonical results never show empty" contract.
        _oracle_error_msg: str | None = None
        data: dict = {}
        try:
            response, oracle_mode = _generate_oracle_gemini_response(video_file_ref, prompt)
            logger.info("Gemini oracle succeeded mode=%s", oracle_mode)
            try:
                data = _extract_json_object_from_text(response.text or "")
            except (json.JSONDecodeError, ValueError) as je:
                logger.error(
                    "Gemini JSON parse failed: %s text=%s", je, (response.text or "")[:2000]
                )
                _oracle_error_msg = "Oracle commentary could not be parsed — biomechanical data is complete."
        except APIError as e:
            try:
                err_code = int(getattr(e, "code", 503) or 503)
            except (TypeError, ValueError):
                err_code = 503  # code attr is a non-int string (e.g. "RESOURCE_EXHAUSTED")
            logger.error(
                "Gemini APIError after retries: code=%s message=%s details=%s",
                err_code,
                getattr(e, "message", None),
                getattr(e, "details", None),
            )
            if err_code == 401:
                _oracle_error_msg = "Oracle unavailable: API key missing or invalid. Set GEMINI_API_KEY on the server."
            elif err_code == 429:
                _oracle_error_msg = "Oracle unavailable: rate limit reached. Biomechanical data shown below."
            else:
                _oracle_error_msg = "Oracle commentary temporarily unavailable. Biomechanical data is complete."
        except Exception as exc:
            logger.exception("Gemini generate_content failed: %s", exc)
            _oracle_error_msg = "Oracle commentary temporarily unavailable. Biomechanical data is complete."

        data["kinematic_deltas"] = deltas

        out = _normalize_analysis(data, biomech, market_index, match_name, matched_pro)
        if _oracle_error_msg:
            # Surface a friendly degraded-oracle notice in the scout_report field
            # so the frontend always has something to show (not an empty card).
            out["scout_report"] = (
                out.get("scout_report")
                or f"[Oracle commentary unavailable] {_oracle_error_msg}"
            )
            out["oracle_error"] = _oracle_error_msg
        out["athlete_name"] = (athlete_name or "").strip() or "Athlete"
        out["sport"] = sport or "basketball"
        analysis_mode = biomech.get("analysis_mode") or "full"
        # Reduce confidence when multiple people detected (improves pro-match reliability)
        det = (biomech.get("telemetry") or {}).get("detection_metadata") or {}
        if det.get("people_detected_max", 1) > 1:
            confidence_score = round(confidence_score * CONF_MULTI_PERSON_FACTOR, 1)
        vw = (biomech.get("telemetry") or {}).get("validation_warnings") or []
        if vw:
            warning_factor = max(CONF_WARNING_FLOOR, 1.0 - len(vw) * CONF_WARNING_PENALTY_PER)
            confidence_score = round(confidence_score * warning_factor, 1)
        metric_status = biomech.get("metric_status") or {}
        if metric_status:
            available = [m for m in metric_status.values() if m.get("source") != "unavailable"]
            availability_ratio = len(available) / max(1, len(metric_status))
            mean_metric_conf = (
                sum(float(m.get("confidence", 0.0)) for m in available) / len(available)
                if available else 0.0
            )
            reliability_factor = CONF_RELIABILITY_BASE + (
                (1.0 - CONF_RELIABILITY_BASE) * (
                    CONF_RELIABILITY_METRIC_WEIGHT * mean_metric_conf
                    + CONF_RELIABILITY_AVAIL_WEIGHT * availability_ratio
                )
            )
            confidence_score = round(confidence_score * reliability_factor, 1)
        if analysis_mode == "partial":
            confidence_score = round(confidence_score * CONF_PARTIAL_MODE_FACTOR, 1)
        elif analysis_mode == "fallback":
            confidence_score = min(confidence_score, CONF_FALLBACK_CAP)

        ms_all = biomech.get("metric_status") or {}
        av_metrics = [m for m in ms_all.values() if isinstance(m, dict) and m.get("source") != "unavailable"]
        n_predicted = sum(1 for m in ms_all.values() if isinstance(m, dict) and m.get("source") == "predicted")
        mean_mc = (
            sum(float(m.get("confidence", 0.0)) for m in av_metrics) / len(av_metrics)
            if av_metrics
            else 0.0
        )
        oracle_match_degraded = (
            analysis_mode in ("partial", "fallback")
            or n_predicted >= CONF_ORACLE_PREDICTED_THRESHOLD
            or (len(av_metrics) >= 1 and mean_mc < CONF_ORACLE_MEAN_MC_THRESHOLD)
        )
        out["oracle_match_degraded"] = oracle_match_degraded
        if oracle_match_degraded:
            if analysis_mode == "fallback":
                out["oracle_caveat"] = (
                    "Pro comparison is not reliable for this clip — fix pose detection or re-record with clearer framing."
                )
            else:
                out["oracle_caveat"] = (
                    "Pro comparison is approximate: side angles and compressed video often force 2D or partial metrics. "
                    "Re-record from a 45° front offset for a tighter vector match."
                )
            if analysis_mode != "fallback":
                confidence_score = round(confidence_score * CONF_DEGRADED_ORACLE_FACTOR, 1)
        else:
            out["oracle_caveat"] = None

        out["confidence"] = confidence_score
        out["analysis_reliability_score"] = round(float(confidence_score), 1)
        out["detection_metadata"] = det
        # Append a clear warning when biomech produced no measurements so the
        # UI can surface an honest "measurement failed" banner.
        if not stats_available:
            vw = list(vw) + ["Biomechanics not measured: clip too short, joint occlusion, or VFR decode failure."]
        out["validation_warnings"] = vw
        # Phase 2: video quality, confidence factors for transparent attribution
        tel = biomech.get("telemetry") or {}
        vq = tel.get("video_quality") or {}
        out["video_quality_score"] = vq.get("video_quality_score")
        out["video_quality_label"] = vq.get("video_quality_label")
        out["confidence_factors"] = tel.get("confidence_factors") or []
        # A5: Multi-shot segmentation result (None when segment_shots failed or
        # the physics_engine is in fallback mode). Surfaced as-is; the UI chip
        # reads n_shots_detected/valid/degraded. Biomech metrics remain from
        # the single dominant-shot detection (honest: not claiming per-shot median).
        out["shot_segmentation"] = biomech.get("shot_segmentation")
        out["api_schema_version"] = API_SCHEMA_VERSION

        # A6: Provenance block for reproducibility and judge auditability.
        # mediapipe_model_sha is not embedded in the runtime analysis dict;
        # it is SHA-pinned at build time. The GIT_COMMIT_SHA env var (set at
        # Docker build time) is the primary audit anchor.
        seg_info = biomech.get("shot_segmentation") or {}
        out["provenance"] = {
            "git_commit_sha": _GIT_COMMIT_SHA,
            "analysis_mode": "canonical_backend",
            "signals_used": ["wrist_y_nadir", "elbow_velocity_peak"],
            "n_shots_detected": seg_info.get("n_shots_detected"),
            "n_shots_valid": seg_info.get("n_shots_valid"),
            "n_shots_degraded": seg_info.get("n_shots_degraded"),
        }
        return out
    finally:
        for path in {safe_name, *extra_cleanup}:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


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
        logger.warning("Imagen generation failed, using placeholder: %s", e, exc_info=True)
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
    video:                 UploadFile = File(None),
    telemetry_json:        str        = Form(...),
    stats_json:            str        = Form(...),
    athlete_name:          str        = Form("Athlete"),
    kinematic_deltas_json: str        = Form("{}"),
    sport:                 str        = Form("basketball"),
    pro_match:             str        = Form(""),
    clip_start_sec:        float      = Form(0.0),
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
        render_result = generate_correction_video(
            telemetry, stats,
            athlete_name     = (athlete_name or "Athlete").strip() or "Athlete",
            kinematic_deltas = kinematic_deltas,
            sport            = sport or "basketball",
            pro_match        = pro_match or None,
            video_path       = video_path,
            clip_start_sec   = float(clip_start_sec or 0.0),
        )
        if not render_result:
            return {"status": "error", "message": "Correction video could not be rendered (no pose frames available)."}
        video_bytes = render_result.get("video_bytes")
        if not video_bytes:
            return {"status": "error", "message": "Correction video could not be rendered (empty output)."}
        return {
            "status": "success",
            "video_base64": base64.b64encode(video_bytes).decode("utf-8"),
            "render_mode": render_result.get("render_mode", "observed"),
            "render_confidence": render_result.get("render_confidence", 0.9),
        }
    except Exception as e:
        logger.exception("Correction video generation failed")
        return {"status": "error", "message": str(e)}
    finally:
        if tmp_video_path and os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)
