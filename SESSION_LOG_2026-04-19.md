# SESSION LOG — Laksh.ai live-demo hardening
**Date:** 2026-04-19
**Purpose:** Single source of truth for everything done in the Cursor session "live demo hardening" (Apr 19), so context survives across Cursor windows / new chats.

---

## 1. Canonical URL (read this first)

| What | URL |
|---|---|
| **Frontend (use this for the showcase)** | **`https://laksh-ai-tawny.vercel.app`** |
| Backend API | `https://laksh-api.fly.dev` |
| Backend health | `https://laksh-api.fly.dev/health` |

**Why not `https://laksh-ai.vercel.app`?** That slug is owned by a different Vercel account. Vercel auto-assigned us `laksh-ai-tawny.vercel.app` (verified via `vercel project ls` → "Latest Production URL: https://laksh-ai-tawny.vercel.app"). Trying to use `laksh-ai.vercel.app` returns either 404 (different app) or the wrong content. **Do not use it.**

If you want a cleaner URL for the showcase poster: register a custom domain (`laksh.ai`, `laksh-ai.com`, etc.) and add it via `vercel domains add <name>` then set DNS. ~10 min if you already own a domain. The `-tawny` slug is fine for the demo itself.

---

## 2. What was wrong (in priority order, with verified root causes)

| # | Symptom | Root cause (verified) | Fix |
|---|---|---|---|
| 1 | "Failed to fetch" on the canonical result | `NEXT_PUBLIC_API_BASE` was **never set on Vercel** → the bundle defaulted to `http://localhost:8000` per `web/lib/api.ts:107` | Set env var, redeployed |
| 2 | CORS preflight 400 from preview/deploy URLs | Backend allowlist only had `https://laksh-ai.vercel.app` (which we don't own), exact-match | Added `allow_origin_regex` matching all `*-laksh-ai.vercel.app` and `laksh-ai-*.vercel.app` patterns |
| 3 | Backend going to sleep mid-demo | Fly machine config drifted: ran with autostop=on despite `fly.toml` saying off | `fly machine update --autostop=off --autostart` |
| 4 | Camera frame too small for full body | `max-w-5xl` container + `aspect-video` (16:9) + `object-cover` (crops feet) | `max-w-7xl`, responsive 3:4/4:5/16:10 aspect, `object-contain`, framing tip |
| 5 | "Many reps but counter shows 3" | (a) Wrist visibility drops below 0.4 when ball is held → signal nulls → counter pauses. (b) When signal returns, stale `prevDelta` causes phantom or missed sign-flip. (c) `handleLandmarks` callback identity changed every rep → re-mounted PoseCamera's RAF loop → frame gap | Gap-reset (>500 ms with no signal → re-prime EMA), Schmitt-trigger hysteresis on delta threshold, vis 0.4 → 0.3, removed `repCount` from useCallback deps |
| 6 | Trust signals buried in `<details>` | Existing `provenance`, `parity_probe`, `calibration` fields rendered only after a click | New `TrustPanel` component sits above metrics by default |

All six are deployed and verified live as of 2026-04-19 21:42 UTC.

---

## 3. Files touched

### Backend (`~/Laksh_AI/`)
- `app/main.py` (CORS regex, lines ~90–115). Single change.

### Frontend (`~/Laksh_AI/web/`)
- `lib/realtime/repCounter.ts` — gap-reset, hysteresis, lower vis threshold, new state fields
- `app/[sport]/page.tsx` — wider container, decoupled callback, mounted TrustPanel, updated loading skeleton aspect
- `components/PoseCamera.tsx` — bigger box (3:4 → 16:10 responsive), `object-contain`, framing tip overlay
- `components/TrustPanel.tsx` — **new file**

### Infra (no source changes)
- Fly: `fly machine update e28691e4c0e328 --autostop=off --autostart`
- Fly: `fly secrets set CORS_ORIGINS=...` (set in previous session, still in effect)
- Vercel: `vercel env add NEXT_PUBLIC_API_BASE production` = `https://laksh-api.fly.dev`
- Vercel: `vercel --prod --yes` (twice — once before env var, once after)

---

## 4. Verified live (curl-tested, not just hoped)

```
A. Stable URL → HTTP 200
   curl -I https://laksh-ai-tawny.vercel.app/basketball
B. CORS preflight from canonical URL → HTTP 200, allow-origin echoed
   curl -X OPTIONS -H "Origin: https://laksh-ai-tawny.vercel.app" \
        https://laksh-api.fly.dev/v1/analyze
C. CORS preflight from random vercel.app → HTTP 400, no allow-origin (security preserved)
D. End-to-end POST /v1/analyze/gym → HTTP 200, full canonical_backend response with
   schema_version, provenance, segment, feature_vectors, calibration, parity_probe
E. Backend health → {"status":"ok","chroma_ready":true,"collection_count":550}
F. Fly machine state: started, autostop=off, 1/1 healthy
```

---

## 5. Uncommitted changes

**Nothing committed yet** (per CLAUDE.md operating rules — only commit when user explicitly asks).
When ready, suggested commit breakdown:
1. `fix(api): allow Vercel preview/deploy URLs via CORS regex` — `app/main.py`
2. `fix(realtime): gap-reset and hysteresis in rep counter` — `web/lib/realtime/repCounter.ts`, `web/app/[sport]/page.tsx`
3. `feat(web): trust panel and full-body camera framing` — `web/components/TrustPanel.tsx`, `web/components/PoseCamera.tsx`, `web/app/[sport]/page.tsx`

---

## 6. Demo runbook (60 seconds, do this end-to-end before judges arrive)

1. Open **`https://laksh-ai-tawny.vercel.app/basketball`** (hard refresh: ⌘⇧R).
2. Click **Start camera**, allow permission. Verify:
   - Big portrait/landscape camera box (not tiny 16:9)
   - Framing tip "Stand 6–10 ft back…" visible
   - Pose model loads (`Pose tracking active` text after ~3 s)
3. Walk back. Hold a basketball. Do **5 jump-shot motions**.
4. Click **Record**, do 5 more reps, click **Stop & Analyse**.
5. Verify, in this order:
   - Ghost panel shows ≈ 5 (off by 1 is OK; that's the realtime preview tradeoff)
   - Upload spinner appears, then resolves in ~10–20 s (no "Failed to fetch")
   - **TrustPanel** appears above metrics with: model, parity status, calibration, manifest SHAs
   - Per-rep cards appear with `valid`/`degraded` chips and reason codes
6. Repeat with `gym → dumbbell_curl` and a dumbbell.

If anything fails: `curl https://laksh-api.fly.dev/health` first. If that's slow (>1s), the machine cold-started despite the autostop fix — re-run `fly machine status` and `fly logs`.

---

## 7. How to continue this work in a new Cursor window

You're worried about losing context if you open `~/Laksh_AI` in a fresh Cursor window. Here's the truth:

- **Files**: nothing is lost. All edits in this session were made directly in `~/Laksh_AI/` (the main repo), NOT in the worktree at `~/Laksh_AI/.claude/worktrees/unruffled-torvalds/`. Open the new window on `~/Laksh_AI` and you'll see every change in `git status`.
- **Chat memory**: this Cursor chat is workspace-scoped and **will not** transfer. The new window will have a fresh chat.
- **Bridge**: `HANDOFF.md` (existing) + this `SESSION_LOG_2026-04-19.md` (new) are the bridge. In the new window, just say to the new chat: *"Read HANDOFF.md and SESSION_LOG_2026-04-19.md, then continue."*
- **Transcripts**: Cursor stores chat transcripts on disk under `~/.cursor/projects/<workspace>/agent-transcripts/`. They survive but are not auto-loaded.

### Recommended switch sequence
1. Make sure you're on the right branch in this worktree: `git status` (we've been working on `feat/realtime-demo`, the `~/Laksh_AI` checkout is on the same branch).
2. In the new Cursor window, open the `~/Laksh_AI` folder directly (not the worktree).
3. First message to new chat:
   > Read `HANDOFF.md` and `SESSION_LOG_2026-04-19.md`. The demo URL is `https://laksh-ai-tawny.vercel.app`. We're 4 days from the Northeastern Research Showcase. Pick up from "remaining work" in the session log.
4. New chat will have full context.

---

## 8. Remaining work before showcase (Apr 23)

| Priority | Task | Effort |
|---|---|---|
| P0 | Smoke-test the live URL with a real basketball + dumbbell, confirm rep counts within ±1 of canonical | 10 min |
| P0 | Commit the 3 logical changes from §5 above and push | 5 min |
| P1 | Add 1–2 calibration cohorts so the TrustPanel shows `within_reference` instead of `uncalibrated_v0` for at least one exercise | 1–2 hr |
| P1 | Decide on poster URL: `laksh-ai-tawny.vercel.app` vs custom domain | 10–60 min |
| P2 | Add per-frame logging hook (toggleable) for debugging rep counter on stage if needed | 30 min |
| P2 | Practice the 5-min demo script 3× | 30 min |

---

## 9. Honest open questions / known limitations

- **Rep counter is a ghost preview, not the canonical truth.** It will sometimes off-by-one on fast reps. The canonical backend (after upload) is the source of truth and is graded on validity. This is by design and surfaced in the TrustPanel via the parity probe.
- **`uncalibrated_v0`** status: we report values but suppress reference-range comparisons because we don't have a calibration cohort yet. This is the honest behavior, not a bug.
- **One Fly machine** in `iad` region. If it dies during the demo there's no automatic failover. For a 5-min demo this is acceptable; for production it would need `min_machines_running ≥ 2` across regions.
- **Vercel deployment protection** is on for non-canonical deploy URLs (anonymous traffic gets 401). This is expected; only the canonical alias `laksh-ai-tawny.vercel.app` is fully public.
