# System architecture

## Request flow (happy path)

1. **Client** uploads MP4 to `POST /analyze-video` (optional clip window).
2. **`app/physics_engine.KinematicAnalyzer`**: FFmpeg normalize → multi-pass frame extraction + MediaPipe Pose → phases (jump vs set shot) → metrics + `metric_status` + telemetry.
3. **`app/main`**: Build weighted 8D vector → **ChromaDB** nearest NBA pro → kinematic deltas → **Gemini** JSON (scout + feedback) → merge with biomech, confidence, Oracle caveat flags.
4. **Client** renders **`static/dashboard.html`** (biomechanics, Oracle, optional correction video).

## Key directories

| Path | Responsibility |
|------|----------------|
| `app/main.py` | FastAPI, Chroma lifecycle, Gemini/Imagen/TTS, analysis orchestration |
| `app/physics_engine.py` | Video, pose, metrics, reliability fields |
| `app/correction_engine.py` | Correction / projected motion video |
| `app/db_seeder.py` | NBA → vector seeding for Chroma |
| `static/dashboard.html` | SPA UI |
| `tests/` | Pytest (`pytest.ini` sets `pythonpath`) |
| `evaluation/` | Benchmark manifests + local clips (media gitignored) |

## External dependencies

- **Gemini / Imagen** — narrative and card image; require `GEMINI_API_KEY`.
- **ChromaDB** — persisted under repo `chroma_db/` (or `/tmp` fallback in constrained FS).
- **MediaPipe** — `pose_landmarker_heavy.task` at repo root (see `scripts/download_pose_model.py`).
- **Optional** — Google Cloud TTS for studio voice; else gTTS.

## Honest boundary

Oracle **pro similarity** uses **heuristic NBA stats → 8D**, not motion capture. User vectors come from **monocular pose**. Treat match as **stylistic / indicative**, not ground-truth biomechanical identity.
