# Laksh.ai Oracle Engine

**A computer-vision pipeline that turns a phone video of a basketball jump shot into shot-form feedback — built end to end (pose estimation → physics features → vector search → LLM report → web app).**

Upload a jump-shot video. The system extracts per-frame body pose, derives shot-form features (joint angles, release timing, arc), generates coaching feedback, and — for fun — tells you which NBA player's *statistical archetype* your form lands nearest.

> ### What this is, and what it isn't
>
> **This is:** a solo-built, end-to-end ML system and a *practice-feedback* tool. The knee/elbow angles are real goniometry from MediaPipe pose (coarse — roughly ±5–15°, side-view dependent). Video feedback genuinely helps motor learning.
>
> **This is NOT** lab-grade biomechanics, and the "NBA match" is **not** a biomechanical comparison. The NBA player vectors are synthesized from box-score stats (points, rebounds, 3P%) and contain zero motion data, so the match is an *entertainment* feature — "which pro's stat-archetype does your form land nearest?" — not "your mechanics are like Player X's." Accurate markerless biomechanics needs multiple calibrated cameras (e.g. [OpenCap](https://www.opencap.ai/)); a single uncalibrated phone cannot match that.
>
> **Not medical advice.** General fitness and practice feedback only — not a medical device, not injury screening, and not a substitute for a qualified coach, athletic trainer, or physician.

---

## Table of Contents

- [Project layout](#project-layout)
- [What It Does](#what-it-does)
- [Architecture Overview](#architecture-overview)
- [What's Implemented](#whats-implemented)
- [What's Coming](#whats-coming)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Technical Limitations](#technical-limitations)
- [Evaluation reel](#evaluation-reel)
- [License](#license)

---

## Project layout

| Path | Purpose |
|------|---------|
| `app/` | Application package: `main.py` (FastAPI), `physics_engine.py`, `correction_engine.py`, `db_seeder.py`, `sport_configs.py` |
| `static/` | `dashboard.html` (SPA UI) |
| `tests/` | Pytest suite |
| `scripts/` | Dev/ops helpers (pose download, benchmarks, golden JSON) |
| `evaluation/` | Benchmark manifests and (local) clip folder |
| `docs/` | Guides + `docs/planning/` archive |
| Repo root | `requirements.txt`, `Dockerfile`, `Makefile`, `chroma_db/`, `pose_landmarker_heavy.task` (downloaded) |

---

## What It Does

1. **Ingest** — You upload a short video of a basketball jump shot (MP4 or similar).
2. **Extract** — MediaPipe Pose extracts joint trajectories; a physics pipeline computes shot-form features (release arc, knee/elbow angles, release timing, fluidity/balance proxies). These are directional cues from a single camera, not lab measurements.
3. **Stat-archetype match (for fun)** — Your feature vector is compared (cosine nearest-neighbor in ChromaDB) against ~500 NBA players whose vectors are *synthesized from box-score stats, not motion capture*. The nearest neighbor is a playful "stat-personality" match — **not** a biomechanical one. See [Technical Limitations](#technical-limitations).
4. **Explain** — Gemini 2.5 Flash turns your features, the matched archetype, and the deltas into a readable report and three coaching cues. Treat the output as directional practice feedback, not a verified biomechanical prescription.
5. **Present** — A React dashboard renders metrics, radar charts, the archetype comparison, and optional audio brief (TTS) plus a generative metric card (Imagen 4).

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Video Upload   │────▶│  physics_engine  │────▶│  8D Vector  │
│  (MP4)          │     │  MediaPipe Pose  │     │  (raw)      │
└─────────────────┘     └──────────────────┘     └──────┬──────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  dashboard.html │◀────│  main.py          │◀────│  ChromaDB   │
│  (React SPA)     │     │  FastAPI          │     │  cosine NN   │
└─────────────────┘     └────────┬─────────┘     └─────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ Gemini   │ │ Imagen 4 │ │ Cloud TTS │
              │ 2.5 Flash│ │ (card)   │ │ or gTTS   │
              └──────────┘ └──────────┘ └──────────┘
```

---

## What's Implemented

| Component | Description | Technology |
|-----------|-------------|------------|
| **Physics Engine** (`app/physics_engine.py`) | 3D pose extraction, Savitzky–Golay smoothing, dimensionless velocity/arc derivation, kinetic chain event detection | MediaPipe Pose (Heavy), OpenCV, NumPy, SciPy, Pandas |
| **8D Vector Schema** | release_velocity_mps*, shot_arc_deg, knee_angle, elbow_angle, kinetic_sync_ms, fluidity_score, hip_rotation_deg, balance_index (*proxy — see Limitations) | Aligned in shape across `app/physics_engine`, `app/db_seeder`, and `app/main` |
| **NBA stat-archetype index** | Deterministic heuristics map NBA box-score stats → 8D archetype vectors (NOT motion capture); ChromaDB cosine search returns nearest archetype | nba_api, Chromadb |
| **AI Scout Report** | Structured JSON output: scout_report, athlete_feedback (3 items), witty_catchphrase | Gemini 2.5 Flash, response schema |
| **Generative Asset** | Holographic 9:16 metric card with personalized overlay | Imagen 4, fallback SVG |
| **Audio Brief** | Text-to-speech of scout report | Google Cloud TTS Studio Voices (en-US-Studio-O) or gTTS fallback |
| **Dashboard** | Hash-routed SPA: Ingestion, Biomechanics, Archetype Match | React 18, Tailwind CDN, Babel standalone |
| **Deployment** | Dockerfile, .dockerignore, .gitignore | Python 3.11-slim, uvicorn |

---

## What's Coming

- **Honest-metric pass** — rename `release_velocity_mps` → `release_power_index` (it is a 2D proxy, not a true velocity); demote single-camera hip rotation to a clearly-labeled low-confidence field.
- **A real validation point** — hand-label joint angles on a small clip set and report angle MAE vs manual measurement (concurrent validity), instead of self-consistency only.
- **Longitudinal trends** — session storage and across-session consistency tracking (the part with actual retention value).
- **3D volumetric mapping** — improved accuracy from 45° front-offset camera angles; calibration-aware confidence scoring.
- **Production hardening** — CORS origins restriction, rate limiting, optional authentication.

---

## Prerequisites

- **Python 3.11+**
- **Gemini API key** — [Get one](https://ai.google.dev/gemini-api/docs/api-key) from Google AI Studio.
- **Optional:** Google Cloud TTS credentials for Studio Voices (otherwise gTTS is used).

---

## Quick Start

### Local Development

```bash
# Clone and enter project
cd Laksh_AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and add your Gemini API key
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your-key-here

# Start server (with auto-reload)
./run.sh
# Or: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). On first run, the app seeds ChromaDB from the NBA API (~30s cold start); subsequent starts reuse the persisted DB.

### Docker

```bash
docker build -t laksh-oracle .
docker run -p 8000:8000 -e GEMINI_API_KEY=your-key laksh-oracle
```

---

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google AI Studio API key for Gemini and Imagen |
| `GOOGLE_APPLICATION_CREDENTIALS` | No | Path to service account JSON for Cloud TTS (enables Studio Voices) |
| `CORS_ORIGINS` | No | Comma-separated origins (default: production + localhost) |
| `LOG_LEVEL` | No | Python log level: `DEBUG`, `INFO` (default), `WARNING`, … |

---

## Testing

```bash
pytest tests/ -v
```

Regression test requires a golden video. See [docs/GOLDEN_VIDEO_GUIDE.md](docs/GOLDEN_VIDEO_GUIDE.md) for how to create one. Without it, the test is skipped.

> **Note on what the regression test proves:** the golden values are generated from the analyzer's *own* output, so the test enforces **self-consistency** (the pipeline hasn't drifted), **not** accuracy against marker-based ground truth. Concurrent-validity testing is listed under [What's Coming](#whats-coming).

```bash
make test
# or: pytest tests/ -q
```

---

## Evaluation reel

Bench the pose pipeline on your own clips (optional): [evaluation/README.md](evaluation/README.md).

```bash
python scripts/download_pose_model.py   # model at repo root if missing
make eval-bench                         # needs evaluation/clips/*.mp4
# or: bash scripts/run_evaluation_local.sh
```

Spec and pass/fail guidance: [docs/evaluation_set_spec.md](docs/evaluation_set_spec.md). Validation: [docs/VALIDATION_STRATEGY.md](docs/VALIDATION_STRATEGY.md).

**Architecture & contributing:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves dashboard HTML |
| `/api` | GET | API status |
| `/health` | GET | Liveness probe (ChromaDB readiness); 503 if unavailable |
| `/docs` | GET | OpenAPI/Swagger UI |
| `/analyze-video` | POST | `video` (multipart): Returns full analysis with archetype match, scout report, feedback |
| `/generate-metric-card` | POST | `{ "match": "Player Name" }`: Returns Imagen 4 card or SVG fallback |
| `/generate-audio-brief` | POST | `{ "text": "..." }`: Returns base64 MP3 |

---

## Technical Limitations

- **The NBA match is NOT biomechanical.** Player vectors are deterministic heuristics over box-score stats (points, rebounds, 3P%) — they contain no motion-capture data. A player's "elbow angle" in this DB is literally a function of their 3-point percentage. The match shares axis *labels* with your measured form but not *meaning*; it is an entertainment feature, not a biomechanical pro-comparison.
- **`release_velocity_mps` is a proxy, not a measured velocity.** It is a dimensionless 2D pixel ratio (wrist travel / torso length) scaled into a plausible range — no ball tracking, no camera calibration, no real time-base. The misleading unit will be renamed (`release_power_index`).
- **Monocular depth limits.** Single-camera pose compresses depth: knee/elbow angles are coarse (~±5–15° on a clean side view) and **hip rotation (a depth-axis quantity) is low-confidence — closer to noise than measurement.**
- **No ground-truth validation yet.** The regression test enforces self-consistency, not accuracy vs marker-based motion capture. A concurrent-validity study is future work.
- **Single-pose assumption.** The pipeline expects one visible shooter; crowded or occluded frames degrade metrics.
- **External APIs.** Gemini, Imagen, and NBA API are subject to rate limits and availability.

---

## License

MIT © 2026 Parv Patodia
