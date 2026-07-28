# Sponsor Technology Usage — Laksh.ai (Wizard Hackathon)

Exactly which sponsor APIs Laksh.ai uses, where in the code, and how to verify each
is live. Both integrations are deployed and verified against the live backend.

- **Live app:** https://laksh-ai-tawny.vercel.app
- **Live API:** https://laksh-api.fly.dev
- **Track:** Sports Arena

---

## 🟦 InsForge — *Best Use of InsForge* ($500)

**The backend is provisioned entirely through code, not a dashboard.** It authenticates
to InsForge and provisions the whole backend live via the InsForge CLI and OAuth:

- Created the Postgres schema — `sessions` + `rep_results` — via `insforge db import`
  (migration: [`infra/insforge/0001_init.sql`](infra/insforge/0001_init.sql)).
- Created a private `laksh-clips` storage bucket via `insforge storage create-bucket`.
- Auth (Google / GitHub / email) is InsForge-native.

**Where it's wired in the product:**
- [`app/persistence/insforge_store.py`](app/persistence/insforge_store.py) —
  `InsForgeSessionStore` implements the app's `SessionStore` interface over the
  InsForge REST API (`POST/GET /api/database/records/{table}`, `Authorization: Bearer`,
  PostgREST `order`/`limit`/`eq`/`not.is.null` filters).
- [`app/persistence/store.py`](app/persistence/store.py) — the backend factory
  selects InsForge when `LAKSH_PERSISTENCE_BACKEND=insforge` (set on Fly), behind the
  same interface as the SQLite fallback (clean dependency inversion).
- Every analysis persists a session row (+ per-rep rows); `GET /v1/leaderboard` reads
  the best `form_index` per exercise from InsForge.

**Verified live:** REST insert/select/delete returned `201/200/204`; a fresh analysis
through the live API wrote a real `sessions` row and the leaderboard count went
`4 → 5` with the new row ranked correctly (`backend: insforge`). Project:
`Laksh_AI` (`q9p6qk2x.us-west`).

**Verify it yourself:**
```bash
curl "https://laksh-api.fly.dev/v1/leaderboard?exercise_id=back_squat" | jq '{backend, count}'
# -> {"backend": "insforge", "count": 5+}
```

---

## 🟪 You.com — *Best Use of You.com* ($1,000)

**Grounded, cited coaching.** Laksh detects form faults **deterministically** from
measured reps (each cue shows its rule + evidence — the honesty contract). For each
fault + exercise, the backend calls the **You.com Search API** (`GET /v1/search`,
`livecrawl`/`freshness`) for real sports-science sources and attaches **resolvable
citations** to the cue. Detection is never LLM-invented; You.com grounds the *remediation*.

**Where it's wired:**
- [`app/coaching/you_search.py`](app/coaching/you_search.py) — `YouComClient` over the
  You.com Search API (`Authorization: Bearer`, defensive `results.web[]` parsing).
- [`app/coaching/grounding.py`](app/coaching/grounding.py) — maps each detected fault to
  a search query, attaches citations, preserves the measured cue text verbatim.
- [`app/api/v1/coaching.py`](app/api/v1/coaching.py) — `POST /v1/coaching/ground`.
- [`web/components/FormInsights.tsx`](web/components/FormInsights.tsx) — renders the
  cited source links + a "Sourced via You.com" badge.

**Verified live:** a real "fast eccentric" fault on a back squat returned 3 relevant
citations — including a **peer-reviewed MDPI paper** on eccentric back-squat velocity —
resolving to live URLs (`grounding_enabled: true`, `reason_codes: ["you_com_grounded","freshness:year"]`).

**Verify it yourself:**
```bash
curl -s -X POST https://laksh-api.fly.dev/v1/coaching/ground \
  -H 'Content-Type: application/json' \
  -d '{"exercise_id":"back_squat","faults":[{"id":"tempo-fast-eccentric","title":"Control the lowering phase","cue":"Slow your descent."}]}' \
  | jq '{grounding_enabled, backend, citations: .cues[0].citations}'
```

---

## ⚪ Nebius (Token Factory) — not wired

Phase 3 (swap the report LLM to a Nebius open model behind a config flag with a Gemini
fallback) is designed but **not enabled** — Nebius API access requires card details we
deferred. The flag + fallback path are the planned integration point; we do not claim
Nebius usage.

---

## Honesty note (a feature, not a caveat)
Every metric carries `{value, unit, status, reason_codes}`; uncalibrated metrics are shown
without a grade; the leaderboard `form_index` is explicitly an *uncalibrated relative
index*, not a validated score. The sponsor integrations strengthen this rather than mask
it: InsForge stores the full provenance (incl. `git_commit_sha`) per session, and You.com
makes the coaching auditable to real sources.
