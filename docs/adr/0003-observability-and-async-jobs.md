# ADR 0003: Observability and async job queue for `/analyze-video`

## Status

**Proposed.** Decision doc only — no implementation in this change. Tracks
[product-grade_laksh_roadmap](../product-grade_laksh_roadmap_05e7df02.plan.md)
Phase E ("API + job queue + SLO + shadow / canary").

## Context

`POST /analyze-video` is synchronous today: one request holds the worker
until pose, metrics, and LLM calls finish. This blocks scale in three ways:

1. **Cold start + pose model load** dominate latency on first request per
   container (up to 30 s). We cannot time out below that without degrading UX.
2. **Gemini + Imagen + TTS** are external; their p99 is unbounded from our
   perspective. A sync HTTP handler holding a connection for 40+ seconds is
   hostile to load balancers and mobile networks.
3. **Per-container concurrency** is effectively 1 for the pose path
   (MediaPipe is not safely thread-reentrant in our wrapper — see
   `app/pose/*`). This maps 1:1 to uploads, so a spike queues against
   uvicorn workers rather than a purpose-built queue.

We do not have p50/p95 latency metrics today. Without them we cannot claim
"improved" after a change; see roadmap rubric row "Product → B+".

## Decision (when implemented)

### 1. Split the API into submit + poll

- `POST /analyze-video` -> returns `{job_id, status: "queued"}` immediately.
- `GET /jobs/{job_id}` -> `{status: queued|running|done|failed, result?}`.
- Result persisted keyed by `job_id` with a retention window (default 7 d).

### 2. Choose Redis + RQ (not Celery) for MVP

- **RQ** has ~200 LOC of state, one dependency (redis-py), and first-class
  Python stack traces. Celery's scheduling flexibility is not needed for a
  single "pose + LLM + TTS" pipeline.
- Self-host Redis on the same box for MVP (`redis:7-alpine`, maxmemory
  policy `allkeys-lru`, persistence off — jobs are disposable).
- Worker container runs `rq worker --with-scheduler` reading the same image.
- Escape hatch: if volume requires it, Celery behind the same submit+poll
  HTTP contract is a drop-in replacement because the contract hides the
  queue choice.

### 3. Observability (structured logs + trace id + latency breakdown)

Every job emits one JSON log line per phase with a shared `job_id` field:

| Phase | Fields emitted |
|---|---|
| `upload.received` | `bytes`, `content_type`, `client_ip_hash` |
| `decode.started` | `ffmpeg_version`, `preprocess_applied` |
| `pose.started` | `backend`, `n_frames`, `device` |
| `pose.finished` | `detection_rate`, `pose_usable_heuristic`, `reason_codes`, `ms` |
| `metrics.finished` | `analysis_mode`, `ms` |
| `llm.finished` | `model`, `tokens_in`, `tokens_out`, `ms` |
| `tts.finished` | `provider`, `ms` |
| `job.finished` | `status`, `total_ms` |

Aggregation: p50 / p95 / p99 per phase via log ingestion (Datadog, Grafana
Loki, or `jq` + SQLite for self-hosted). **Do not** roll our own metrics
endpoint — `prometheus_client` is a one-line add if we keep everything
structured now.

### 4. Latency budget

Target for a 3-second clip at 30 fps on current CPU container:

| Phase | Budget p50 | Budget p95 |
|---|---|---|
| decode + preprocess | 0.8 s | 2.0 s |
| pose (90 frames) | 2.5 s | 6.0 s |
| metrics + vector search | 0.2 s | 0.6 s |
| LLM (Gemini) | 2.0 s | 8.0 s |
| TTS | 1.0 s | 4.0 s |
| **total (sync-equivalent)** | **6.5 s** | **20.6 s** |

SLO: 95% of 3-second-clip jobs finish under 15 s wall clock at load <= 10
concurrent jobs. If pose dominates, GPU is the lever. If LLM dominates,
prompt-length + streaming are the levers. **Measure first.**

### 5. Shadow / canary before making canonical-pose path default

When P3 canonical path becomes the default (see ADR 0002), shadow-run the
new stack on uploaded clips without user-facing output for one week. Compare
`detection_rate` / `pose_usable_heuristic` / `reason_codes` distributions
against the live stack. Only promote when deltas match the claim tier
recorded in the scorecard.

## Consequences

- **New ops surface:** Redis + worker container + result store. Dockerfile
  gains a multi-target build (`api` vs `worker` same image, different CMD).
- **Client contract changes:** frontend (`static/dashboard.html`) must
  handle `job_id` + polling. Acceptable because current UX already shows a
  loading state for the full duration of the sync request.
- **Observability cost:** structured logs add ~200 bytes per phase per job.
  At 1k jobs/day this is ~1.4 MB/day — negligible.
- **Calendar:** MVP implementation estimated 1 focused day for submit+poll
  + RQ wiring; 1 day for structured log schema; 1 day for UX update. Not
  done in this change — roadmap says ship after Phase A + partial B land.

## What we are deliberately NOT doing

- Not picking Kubernetes. A single-VM docker-compose file is the target
  deploy surface; k8s is premature.
- Not building a custom metrics dashboard. Structured JSON logs + an
  aggregator is the smallest thing that catches regressions.
- Not adding auth or rate limits in this ADR (tracked separately in
  roadmap Phase E).

## Related

- ADR 0001 (Phase A MediaPipe baseline)
- ADR 0002 (P3 canonical in KinematicAnalyzer)
- [docs/POSE_UPGRADE_EXECUTION_PLAN.md](../POSE_UPGRADE_EXECUTION_PLAN.md) §4 analysis_mode
- [docs/product-grade_laksh_roadmap_05e7df02.plan.md](../product-grade_laksh_roadmap_05e7df02.plan.md) Phase E
