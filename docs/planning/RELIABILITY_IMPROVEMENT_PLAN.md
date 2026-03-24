# Laksh.ai Reliability Improvement Plan

> **Purpose**: A PhD-level, scientist-grade roadmap to transform this sports analytics platform into the most reliable, trustworthy basketball shot analysis system in the industry. Written from the perspective of a biomechanics researcher and a principal engineer who has deployed platforms at scale.

---

## Executive Summary

After a full codebase scan, the platform is **structurally sound** with a deterministic physics engine, MediaPipe pose extraction, 8D vector matching, and Gemini for qualitative feedback. The main gaps are: **operational reliability** (errors, edge cases, observability), **scientific validation** (reproducibility, uncertainty, ground truth), and **user trust** (transparent limitations, graceful degradation). This plan prioritizes changes that deliver the highest reliability gain per unit effort.

**LiteLLM / config.yaml**: Not used in the live pipeline. Safe to ignore or remove later; no impact on reliability.

---

## 1. Current State (As Built)

### Architecture

```
User uploads video → FastAPI /analyze-video
  → KinematicAnalyzer (physics_engine.py): MediaPipe Pose + physics math
  → 8 metrics + telemetry (dip, release, per-frame joints)
  → ChromaDB 8D vector search → NBA pro match
  → Gemini 2.5 Flash: scout_report + athlete_feedback
  → Dashboard: React (UMD) + Tailwind, hash router
```

### What Already Works Well

| Component | Status | Notes |
|-----------|--------|------|
| **Clip selection** | ✅ | `start_sec`/`end_sec` wired end-to-end |
| **Segment selection** | ✅ | Physics engine `extract_frames()` respects clip range |
| **Validation warnings** | ✅ | Video quality, pose visibility, biological plausibility |
| **Multi-person awareness** | ✅ | `_count_people_sampled`, confidence penalty |
| **Metric tooltips** | ✅ | What/why/ideal/limitation per metric |
| **Overlay** | ✅ | Joint labels, release point, shooting arm, phase |
| **Fallback** | ✅ | Graceful degradation when pose fails |
| **Docker** | ✅ | Pose model pre-downloaded; ChromaDB seeded at build |

### Deployment (Railway + GitHub)

- Railway auto-deploys on push to connected repo
- Single container: FastAPI + uvicorn on port 8000
- Dashboard served at `/`, API at `/analyze-video`, etc.
- CORS: Second middleware allows `*` — first (lakshai-production) is effectively overridden

---

## 2. Gaps & Risks (Honest Assessment)

### A. Operational Reliability

| Risk | Impact | Likelihood | Current Mitigation |
|------|--------|------------|--------------------|
| **Gemini API failure** | User sees 503 | Medium | Generic error message; no retry |
| **NBA API timeouts at build** | ChromaDB empty | Medium | Fallback 15 players; build succeeds |
| **Large video upload** | Timeout / OOM | Medium | No size limit documented |
| **MediaPipe model missing** | Crash on first request | Low | Docker pre-downloads; local may download |
| **ChromaDB disk full (ephemeral)** | Startup fails | Low | Tries `/tmp/apex_chroma` as fallback |
| **No health endpoint** | Ops can't probe liveness | High | None |

### B. Scientific / Accuracy

| Gap | Impact | Notes |
|-----|--------|------|
| **No uncertainty quantification** | Overconfident numbers | "147°" not "147° ± 6°" |
| **Release velocity = wrist proxy** | Systematic error | No ball tracking; m/s approximate |
| **Single-shot assumption** | Wrong subject in multi-shot | Uses first dip/release in clip |
| **No regression test** | Regressions slip through | No golden video in CI |
| **market_index uses wrong param** | Minor | `calculate_market_index(vector, distance)` — `vector` unused; works but confusing |

### C. User Experience

| Gap | Impact |
|-----|--------|
| **No upload size limit** | Large files can timeout |
| **No progress indicator** | User waits blindly for analysis |
| **Error messages generic** | "Analysis service error" — not actionable |
| **sport_configs not wired to UI** | Basketball hardcoded; extensibility unused |

### D. Codebase Hygiene

| Item | Notes |
|------|-------|
| Duplicate CORS middleware | First restricts to Railway domain; second allows `*` — redundant |
| `calculate_market_index` | First arg unused; consider renaming to `_` |
| `sport_configs.py` | Exists but `main.py` doesn't import it; future tennis/golf ready |
| Many roadmap docs | IMPLEMENTATION_ROADMAP, TECHNICAL_ROADMAP, VALIDATION_STRATEGY, etc. — consider consolidating |

---

## 3. Prioritized Improvement Roadmap

### Tier 1: Critical for Reliability (0–2 weeks)

**1.1 Health & Observability**

- Add `GET /health`:
  - Returns `200` if ChromaDB connected and collection has items
  - Returns `503` if DB unhealthy
  - Lightweight (no video processing)
- Add `GET /api` or extend it with: `version`, `chroma_ready`, `collection_count`
- **Why**: Railway and any load balancer need a liveness probe. Without it, restarts and deploys are blind.

**1.2 Upload Limits & Timeouts**

- Enforce max upload size (e.g. 50 MB) at FastAPI level
- Return `413 Payload Too Large` with clear message: "Video must be under 50 MB. Trim your clip or compress."
- Set reasonable timeouts for `/analyze-video` (e.g. 120 s) so clients don't hang forever

**1.3 Error Handling & User Messages**

- Differentiate: `429` (rate limit), `503` (Gemini down), `500` (internal)
- Return structured `{ "error": "...", "code": "...", "retry_after": null }` for client display
- Frontend: show "Rate limit — try again in a minute" vs "Analysis temporarily unavailable"

**1.4 Regression Test**

- Add `tests/` with one golden video (or synthetic minimal clip)
- Assert: pipeline runs, returns stats dict with expected keys, no crash
- Run in CI (e.g. GitHub Actions on push) — blocks deploy if test fails

---

### Tier 2: Trust & Transparency (2–4 weeks)

**2.1 Video Quality Score (0–100)**

- Combine: resolution, FPS, aspect ratio, pose visibility, people count
- Single number in API response and UI: "Video quality: 78/100 — good for analysis"
- When low: "Re-record for better accuracy: use 720p+, single person, 45° angle"

**2.2 Uncertainty Ranges (Where Feasible)**

- From pose confidence or frame-to-frame variance: e.g. "Knee: 147° ± 5°"
- Start with 1–2 metrics (knee, elbow) — bootstrap or analytic propagation
- Display in UI: "147° ± 5°" instead of bare "147°"

**2.3 Method Cards**

- One-page doc per metric: formula, assumptions, limitations, typical error
- Link from tooltips: "How we calculate this"
- Supports investor due diligence and user trust

---

### Tier 3: Accuracy Breakthroughs (1–3 months)

**3.1 Ball Tracking (YOLO or similar)**

- Biggest accuracy gain for **release velocity** and **shot arc**
- Current: wrist proxy → approximate
- With ball: true release point, trajectory, arc
- Effort: medium–high; new dependency, training or fine-tuning

**3.2 Multi-Shot Detection**

- Detect each release event; segment video into N shots
- Let user pick which shot to analyze (or analyze all, show best/worst)
- Eliminates "wrong subject" when video has multiple shots

**3.3 RTMPose Option**

- When GPU available (e.g. Railway add-on): use RTMPose instead of MediaPipe
- Better joint accuracy; validated in basketball studies
- Design: pose-model abstraction so MediaPipe and RTMPose share the same output schema

---

### Tier 4: Platform Evolution (3–12 months)

- Lab validation partnership (mocap comparison)
- Longitudinal analytics (same athlete over time)
- Export annotated video (overlay rendered to MP4)
- Multi-sport (tennis, golf) via `sport_configs`

---

## 4. Immediate Action Checklist

Before investing in new features, fix these **quick wins**:

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 1 | Add `GET /health` | 1–2 h | Ops probe; deploy confidence |
| 2 | Enforce 50 MB upload limit | 30 min | Prevents OOM / timeout |
| 3 | Differentiate error responses (429, 503) | 1 h | Better UX |
| 4 | Fix `calculate_market_index` call (or drop unused param) | 5 min | Code clarity |
| 5 | Add basic regression test (golden video or mock) | 2–4 h | Prevents regressions |
| 6 | Consolidate CORS (single middleware, env-based origins) | 15 min | Cleaner config |

---

## 5. What NOT to Do (Yet)

- **LiteLLM migration**: You're not using it; no need to remove or refactor. Leave as-is.
- **Pose model switch (RTMPose/ViTPose)**: Higher accuracy but more complexity. Do after reliability basics.
- **Multi-sport backend**: `sport_configs` is ready; wire when you add tennis/golf.
- **Rewrite frontend**: React UMD works; focus on backend and API first.

---

## 6. Deployment Notes (Railway)

- **Environment**: Ensure `GEMINI_API_KEY` is set in Railway
- **Health check**: Configure Railway to use `GET /health` as the health check path (when added)
- **Build**: Dockerfile uses Python 3.11, pre-seeds ChromaDB — if NBA API fails, fallback players used
- **Logs**: `logger.info("ChromaDB health: OK | ...")` — grep Railway logs for "ChromaDB health"

---

## 7. Summary

| Tier | Focus | Timeline |
|------|-------|----------|
| **1** | Health, limits, errors, regression test | 0–2 weeks |
| **2** | Quality score, uncertainty, method cards | 2–4 weeks |
| **3** | Ball tracking, multi-shot, RTMPose | 1–3 months |
| **4** | Lab validation, longitudinal, export | 3–12 months |

**Lead with reliability.** A platform that rarely crashes, fails gracefully, and tells users when to re-record will build more trust than extra features with opaque failures.

---

*Document version: 1.0 | Generated after full codebase scan*
