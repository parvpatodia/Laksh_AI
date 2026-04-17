# Laksh.ai Oracle Engine

**Biomechanical intelligence for basketball shot mechanics.**

Upload a jump-shot video. The system extracts an 8-dimensional kinematic fingerprint, matches you against active NBA professionals in vector space, and delivers an AI-powered scout report with actionable coaching feedback.

---

## Table of Contents

- [Project layout](#project-layout)
- [What It Does](#what-it-does)
- [Architecture Overview](#architecture-overview)
- [What's Implemented](#whats-implemented)
- [What's Coming](#whats-coming)
- [Roadmap](#roadmap)
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
2. **Extract** — MediaPipe Pose extracts 3D joint trajectories; a custom physics pipeline computes eight biomechanical metrics (release velocity, shot arc, knee/elbow angles, kinetic sync, fluidity, hip rotation, balance).
3. **Match** — Your 8D vector is queried against ChromaDB, which holds ~500 active NBA players. The nearest neighbor by cosine similarity becomes your "Oracle Match."
4. **Explain** — Gemini 2.5 Flash consumes your stats, the matched pro's baseline, and kinematic deltas to produce a structured scout report and three drill-focused coaching insights.
5. **Present** — A React dashboard renders metrics, radar charts, pro comparison, and optional audio brief (TTS) plus a generative metric card (Imagen 4).

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
| **8D Vector Schema** | release_velocity_mps, shot_arc_deg, knee_angle, elbow_angle, kinetic_sync_ms, fluidity_score, hip_rotation_deg, balance_index | Aligned across `app/physics_engine`, `app/db_seeder`, and `app/main` |
| **NBA Oracle** | Deterministic heuristics map NBA box-score stats → 8D vectors; ChromaDB cosine search returns nearest pro | nba_api, Chromadb |
| **Market Index** | L2-distance–based valuation tiers (Elite → Amateur) | Calibrated thresholds |
| **AI Scout Report** | Structured JSON output: scout_report, athlete_feedback (3 items), witty_catchphrase | Gemini 2.5 Flash, response schema |
| **Generative Asset** | Holographic 9:16 metric card with personalized overlay | Imagen 4, fallback SVG |
| **Audio Brief** | Text-to-speech of scout report | Google Cloud TTS Studio Voices (en-US-Studio-O) or gTTS fallback |
| **Dashboard** | Hash-routed SPA: Ingestion, Biomechanics, Oracle Match | React 18, Tailwind CDN, Babel standalone |
| **Deployment** | Dockerfile, .dockerignore, .gitignore | Python 3.11-slim, uvicorn |

---

## What's Coming

- **3D volumetric mapping** — Improved accuracy from 45° front-offset camera angles; calibration-aware confidence scoring.
- **Multi-sport expansion** — Generalized kinematic schemas beyond basketball (e.g., tennis serve, golf swing).
- **Production hardening** — CORS origins restriction, rate limiting, optional authentication.
- **Mobile-native capture** — On-device recording flow and UX optimizations.
- **Historical trends** — Session storage and longitudinal comparison.

---

## Roadmap

Full plan: [docs/product-grade_laksh_roadmap_05e7df02.plan.md](docs/product-grade_laksh_roadmap_05e7df02.plan.md) — phased path from basketball demo to defensible CV product with a measurement contract, tiered data, and operational honesty. Milestones tracked in [GOALS.md](GOALS.md).

| Phase | Focus | Status |
|---|---|---|
| **A** — Repository excellence | Locked deps, two-speed CI (`pr.yml` + `nightly.yml`), regression scorecard bundle (`make scorecard`), ruff + mypy in CI, pose model SHA-256 pin | **Largely landed** (see `scripts/build_scorecard.py`, `.github/workflows/`, [CONTRIBUTING.md](CONTRIBUTING.md)) |
| **B** — Data moat | Tiered L0 / L1 / L2 datasets; subject-level splits; jitter + calibration reports; human coaching rubric | **Scaffolded** — tooling in `scripts/pose_jitter.py`, `scripts/pose_calibration.py`, `scripts/subject_split_check.py`, rubric in [docs/HUMAN_RUBRIC.md](docs/HUMAN_RUBRIC.md); L1 capture + L2 labels in progress |
| **C** — Pose stack completion | P2 person isolation + P3 canonical joints default in `KinematicAnalyzer` | **Parity telemetry + promotion gate shipping** behind `LAKSH_USE_CANONICAL_JOINTS`; reduce a bench JSONL with `make pose-parity-report JSONL=...`. ADR [0002](docs/adr/0002-p3-canonical-in-kinematic-analyzer.md). |
| **D** — Gym MVP | Exercise v0 freeze, rep segmentation, per-rep feature vector with valid/degraded/unknown semantics | **Exercise v0 frozen** (12 compound movements, [`evaluation/exercise_v0_manifest.json`](evaluation/exercise_v0_manifest.json), schema 1.0.0). **Rule-based rep segmenter shipping** in [`app/gym/rep_segmenter.py`](app/gym/rep_segmenter.py) with valid / degraded / unknown reason codes; labeled F1/IoU eval harness + per-rep feature vector next. See [GOALS.md](GOALS.md) Milestone 1. |
| **E** — Product operations | Async job queue, structured logs, p50/p95 SLO, shadow / canary | **Decision doc landed** ([docs/adr/0003-observability-and-async-jobs.md](docs/adr/0003-observability-and-async-jobs.md)); implementation follows Phase A + partial B |
| **F** — Generative media | LTX / before-after video conditioned on pose | **Deferred** — ships only after pose tracks are stable and legal review closes |

Grading rubric, honest gap analysis, and dependencies between phases are in the roadmap doc.

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
# Optional (matches CI lint + types): pip install -r requirements-dev.txt

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
| `/analyze-video` | POST | `video` (multipart): Returns full analysis with pro match, scout report, feedback |
| `/generate-metric-card` | POST | `{ "match": "Player Name" }`: Returns Imagen 4 card or SVG fallback |
| `/generate-audio-brief` | POST | `{ "text": "..." }`: Returns base64 MP3 |

---

## Technical Limitations

- **NBA vectors are heuristic.** Pro embeddings derive from box-score stats (pts, reb, ast, fg3_pct, etc.), not motion capture. Matching is indicative, not ground-truth.
- **2D camera constraints.** Pure side-profile views compress depth; knee/hip angles can be underestimated. A 45° front-offset improves 3D inference.
- **Single-pose assumption.** The pipeline expects one visible shooter; crowded or occluded frames may degrade metrics.
- **External APIs.** Gemini, Imagen, and NBA API are subject to rate limits and availability.

---

## License

MIT © 2026 Parv Patodia
