# HANDOFF — Laksh.ai Realtime Demo (recovered from Claude Code, Apr 17–18)

**Recovered:** 2026-04-19 from Claude Code session `71166b16-d5ec-4953-a922-36af6a0c1834`
(1,973 records, ~24 hours of work, Apr 17 07:51 UTC → Apr 18 06:19 UTC).
**Recovered by:** Cursor agent reading `~/.claude/projects/-Users-parvpatodia-Laksh-AI--claude-worktrees-unruffled-torvalds/`.

---

## 1. The exact moment work stopped

**Apr 18, 06:19 UTC** — last assistant turn before usage limit hit.

You were running `vercel --prod` in `~/Laksh_AI/web` and the CLI rejected your project name twice:

```
? What's your project's name? Laksh
Error: Project names ... must be lowercase ...
? What's your project's name? LAKSH
Error: Project names ... must be lowercase ...
```

The last instruction Claude gave you was:

> Run this:
> ```bash
> cd ~/Laksh_AI/web && vercel --prod --name laksh-ai
> ```
> If `--name` isn't accepted, answer the prompt with `laksh-ai` (all lowercase, hyphen not space).

**Status check now:** `~/Laksh_AI/web/.vercel/project.json` already exists and contains:
```json
{"projectId":"prj_m6HY5o1ikENFX0jhX8LhlqmdjSmE","orgId":"team_BEEJZwelMU97zFA4FbdZWBCU","projectName":"laksh-ai"}
```
So a project named `laksh-ai` **was** created at some point (linked Apr 17 23:19). What never completed was the **redeploy of the days 8–11 changes** — the parity-probe UI, warm-loader backend, failure-mode cards.

---

## 2. The big picture (your stated goal)

**Project:** Laksh.ai — sports biomechanics analyzer, basketball + gym, with the honesty-first measurement spine you've been building since March.

**Driving deadline:** Northeastern Research Showcase, 8–14 days out (as of Apr 17).
Award category: **Research Award** — judged 40 % Execution & Rigor / 30 % Significance & Innovation / 30 % Presentation & Collaboration.

**Demo target:** A judge walks up, picks a sport, does a movement (jump shot or bicep curl) on the spot, and sees:
1. **Live skeleton overlay + ghost metrics** in the browser within ~1 s (in-browser MediaPipe `LIVE_STREAM`, lite model).
2. **Capture button** uploads the last few seconds of WebM to the backend.
3. **Canonical report** comes back in ≤ 3 s with per-field `{value, unit, status, reason_codes}` and a **parity-probe block** showing how close the live preview was to the canonical pass.

**Tone you keep asking for:** "andraj karpathy level technical ability and expertise" — no quality compromise, no shortcuts on rigor, every decision explainable.

---

## 3. Architecture (built, mostly shipped)

```
Judge browser (Chrome/Safari)
  │ getUserMedia → @mediapipe/tasks-vision LIVE_STREAM (lite, ~30 fps)
  │ canvas overlay + client-side rep counter (TS port of squat hip-y / wrist y)
  │ MediaRecorder captures WebM → POST /v1/analyze
  ▼
Vercel (Next.js 14, App Router, TS, Tailwind, shadcn) — project: laksh-ai
  ▼
Fly.io (FastAPI, single iad machine, shared-cpu-1x, 1 GB, always-on) — app: laksh-api
  ▼
Cloudflare R2 (laksh-clips bucket, 7-day lifecycle)
  ▼
ChromaDB (in-container persistent volume `laksh_chroma`)
```

**Cost ceiling:** ~$5–20 / month total. Domain: deferred — using `laksh-ai.vercel.app` and `laksh-api.fly.dev` until showcase day.

---

## 4. What's actually built (commit-by-commit, branch `feat/realtime-demo` on `~/Laksh_AI`)

52 commits on this branch, last 10:

| Commit | Title |
|---|---|
| `10c00e3` | feat(day10-11): pose warm-loader, eager preload, failure-mode cards |
| `e0142b6` | docs(showcase): rewrite all 4 docs for web demo architecture |
| `4ceeb2d` | feat(day8): wire parity probe into upload endpoint and surface in UI |
| `1c47e73` | feat(day7): add video upload endpoint and CanonicalReport component |
| `6d96bb5` | feat(web): add repCounter and GhostMetricsPanel for ghost metrics |
| `48606b8` | feat(web): add PoseCamera with MediaPipe VIDEO mode and skeleton overlay |
| `cbb0366` | feat(web): scaffold Next.js 14 app with sport selector and API client |
| `47cdf2c` | fix(chroma): clear volume contents instead of rmtree mountpoint |
| `d540b69` | feat(deploy): add R2 client, fly.toml, and gunicorn entrypoint |
| `307e37e` | feat(parity): add realtime-vs-canonical parity probe and ADR 0004 |

### Frontend (`~/Laksh_AI/web/`)
- `app/page.tsx` — landing + sport selector + failure-mode cards
- `app/[sport]/page.tsx` — capture page per sport
- `components/PoseCamera.tsx` — MediaPipe LIVE_STREAM, canvas skeleton overlay
- `components/GhostMetricsPanel.tsx` — live preview metrics
- `components/CanonicalReport.tsx` — backend result with status chips
- `components/ParityProbePanel.tsx` — realtime-vs-canonical numeric agreement
- `components/FailureModeCards.tsx` — 3 demo cards (occluded joint, no reps, multi-person)
- `lib/api.ts` — typed client for `/v1/analyze`, `/v1/jobs`
- `lib/realtime/repCounter.ts` — TS port of squat/bicep/basketball signal logic
- `lib/failureModes.ts` — fixtures
- `lib/pose/` — MediaPipe loader + landmark drawing helpers

Stack: Next.js 14.2.20, React 18.3, TS strict, Tailwind 3.4, shadcn primitives, lucide icons.

### Backend additions
- `app/api/v1/analyze.py` — sport dispatch, calls `KinematicAnalyzer` (basketball) or gym pipeline.
- `app/api/v1/jobs.py` — in-memory job registry + SSE.
- `app/gym/pipeline.py` — orchestrator wrapping segmenter → features → calibration.
- `app/storage/r2_client.py` — boto3 to R2 endpoint, signed URLs.
- `app/parity/realtime.py` — parity-probe block builder.
- `app/main.py` — `_warm_pose_landmarker()` daemon thread on startup.
- `tests/test_api_v1_analyze.py`, `tests/test_parity_realtime.py` — green.
- `docs/adr/0004-realtime-dual-path.md` — design record for the dual-path pattern.

### Infra
- `fly.toml` — `app="laksh-api"`, region `iad`, always-on, 1 GB shared-cpu, internal port 8000, force HTTPS.
- `Dockerfile` — entrypoint runs `gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8000 app.main:app`.
- Deployed once successfully — your message Apr 18 05:10: *"okay the deployment is successful now, i can see the old UI that i had for the website"*.

### Showcase docs (rewritten in commit `e0142b6`)
- `docs/showcase/poster.md` — rebalanced ~50/50 basketball/gym, dual-path diagram, parity-probe section.
- `docs/showcase/pitch_5min.md` — judge-walks-up-and-moves flow.
- `docs/showcase/demo_runbook.md` — open `laksh-ai.vercel.app` → pick sport → Start → move → Stop → result <3 s.
- `docs/showcase/judge_qa.md` — Q&A crib sheet.

---

## 5. Open threads (what was actually unfinished when the limit hit)

1. **🔴 Vercel deploy of latest code (`feat/realtime-demo` HEAD = `10c00e3`).**
   Project linked, last build never completed. **Action:** `cd ~/Laksh_AI/web && vercel --prod` → name `laksh-ai`.
2. **🟡 Fly.io deploy of latest backend.** The Apr 18 04:47 UTC fly deploy hit `deadline_exceeded: context deadline exceeded` mid-build but the image was created (`registry.fly.io/laksh-api:deployment-01KPFE2QSHG8STSA8HDACJWVAJ`, 1.1 GB). Machine `e28691e4c0e328` was launching. **Action:** `fly status -a laksh-api` to confirm it stabilized; if not, `fly deploy` again.
3. **🟡 R2 secrets.** `fly.toml` lists required secrets: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY`, `R2_SECRET`, `R2_BUCKET=laksh-clips`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`. Confirm with `fly secrets list -a laksh-api`.
4. **🟡 `NEXT_PUBLIC_API_BASE` on Vercel.** Must be `https://laksh-api.fly.dev` in Vercel project env, otherwise frontend talks to `localhost:8000` (the default in `.env.local.example`).
5. **🟢 Days 12–14 in the original plan** = dress rehearsal + buffer. No code work.

---

## 6. The honesty contract you keep insisting on

Every metric the system reports carries `{value, unit, status, reason_codes}`. Calibration is `uncalibrated_v0` and **forbids** ranges on uncalibrated entries (enforced in `app/gym/calibration_v0.py:CalibrationEntry.__post_init__`). Reason-code taxonomy is frozen in `app/pose/reason_codes.py` — the realtime layer added exactly two new codes (`realtime_preview`, `canonical_backend_overrode`). Every `result.json` carries `git_commit_sha`, `pose_baseline_version`, `exercise_manifest_sha`, `calibration_manifest_sha`, model SHA-256 — so any judge can re-run analysis offline against the same clip + commit and get byte-identical features.

The **research contribution surface** for this milestone is the `parity_probe` block — numeric agreement between the realtime-preview path and the canonical-backend pass. ADR 0004 is the design record.

---

## 7. The other plans Claude Code is still tracking (out of scope here, FYI)

- `~/.claude/plans/twinkly-hugging-corbato.md` (Apr 17 22:16) — **Suchita Tribute site** ("Wow Factor Elevation", typewriter letter, mic blow-out candle). Separate project under `clever-turing` worktree. Not Laksh.ai.
- `~/.claude/plans/proud-pondering-zephyr.md` (Apr 10) — older, unrelated.

---

## 8. The workflow system you set up on Apr 17

You spent the morning of Apr 17 building a per-laptop Claude Code workflow (Phase 3 of an external `BUILD_GUIDE.md` from `~/Downloads/`). Components landed:
- `~/.claude/CLAUDE.md` — global operating-system rules (the file you can see loaded into this Cursor session).
- `~/.claude/skills/`, `~/.claude/commands/` — skills and slash-commands.
- `~/Laksh_AI/.claude/commands/`, `~/Laksh_AI/.claude/rules/` — project-scoped overrides.
- A `/repo` slash-command you built for ingesting GitHub repos shared by friends (Ritik, Yash). Yash's reference repos:
  - https://github.com/parvpatodia/AI-basketball-analysis
  - https://github.com/vinod-polinati/basketball-vision-analytics
  - https://github.com/AggieSportsAnalytics/ShotFormCorrecter
  - https://github.com/Ed-Zh/Basketball-Analytics
- Auto-push behavior tuned to commit as `parvpatodia` (your account) and never leave Claude's signature.
- A teaching agent meant to explain concepts at "Stanford lecture" depth as you build.

Recurring cost constraint you stated: **$300/month university Claude budget, +$100 extra on request**. You explicitly told Claude *not* to trade away accuracy/rigor for cost.

---

## 9. Repo geography (important — Cursor is in the wrong place)

| Path | Branch | State |
|---|---|---|
| `~/Laksh_AI` | `feat/realtime-demo` | **The actual current work.** 52 commits ahead of `main`. One uncommitted edit to `web/.gitignore`. |
| `~/Laksh_AI/.claude/worktrees/unruffled-torvalds` | `claude/unruffled-torvalds` | **Where this Cursor window is rooted.** Clean, up to date with `origin/main`, does **not** have the realtime-demo work. The `web/` directory does **not exist** here. |
| `~/Laksh_AI/.claude/worktrees/brave-jemison` | `claude/brave-jemison` | Older worktree. |

**Recommendation:** open the main repo (`~/Laksh_AI` on `feat/realtime-demo`) in Cursor instead of this worktree, or `cd ~/Laksh_AI && git checkout feat/realtime-demo` to continue the deploy. All the Vercel/Fly/web/ work is there.

---

## 10. Next 3 actions to ship the showcase

1. `cd ~/Laksh_AI/web && vercel --prod` → confirm name is `laksh-ai` → wait for deploy URL.
2. `cd ~/Laksh_AI && fly status -a laksh-api && fly deploy` → confirm backend is reachable: `curl https://laksh-api.fly.dev/health`.
3. In Vercel dashboard → project `laksh-ai` → Settings → Environment Variables → set `NEXT_PUBLIC_API_BASE=https://laksh-api.fly.dev` → trigger a redeploy.

Then run the demo runbook in `docs/showcase/demo_runbook.md` end-to-end on a fresh laptop.
