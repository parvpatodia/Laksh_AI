# Laksh.ai — On-Stage Demo Script (Wizard Hackathon)

> **Track:** Sports Arena. **Prizes targeted:** Best Use of InsForge ($500), Best
> Use of You.com ($1k), Best Technical Execution, Best Overall.
> **Judges read for:** "is this fundable + does it actually work, live." This script
> is engineered so the *real* system looks its most impressive — no faked data,
> because this panel probes. The honesty spine is sold as a premium feature.

**One-liner:** "Laksh.ai reads your body in motion — 33-landmark pose, in your
browser, with a per-rep biomechanics report that's honest about exactly what it
measured. Strava-grade leaderboard, agent-run backend, coaching grounded in real
sports-science sources."

**Live URLs**
- App: https://laksh-ai-tawny.vercel.app
- Backend health: https://laksh-api.fly.dev/v1/health
- Leaderboard: https://laksh-ai-tawny.vercel.app/leaderboard

---

## Pre-stage checklist (reliability — do this before you walk up)

- [ ] **Do ONE analysis ~5 min before** going on (keeps the MediaPipe model hot on
      the Fly machine — first video request after idle is the slow one).
- [ ] **Do NOT redeploy right before the demo** (a fresh boot reloads the pose model).
- [ ] **Exercise = Back Squat.** It has literature-cited reference ranges (NSCA/ACSM)
      so the report shows full "in range" context, and the hip-y signal segments
      cleanly → reliably `valid` reps. Avoid exercises that show "no reference yet".
- [ ] **Lighting:** face the light. No strong backlight (washes the skeleton).
- [ ] **Framing:** full body, side view, ~2.5 m back. Confirm the skeleton tracks
      before you start the real reps.
- [ ] **Keep the clip short: 2–3 reps, ~6–8 seconds.** Short clip → analysis returns
      in ~15–25 s instead of 60 s. Talk through the progress bar while it runs.
- [ ] **Fallback clip saved** (screen recording of a clean run) in a pinned tab.
- [ ] Tabs: (1) app homepage, (2) `/leaderboard`, (3) fallback clip, (4) `/v1/health`.
- [ ] Venue WiFi tested; phone hotspot ready.

---

## The 2-minute flow (beat by beat)

### Beat 1 — Homepage, the hook (15s)
Open the homepage. Say:
> "Every fitness app guesses your form from population averages. We measure *your*
> body — 33 landmarks, 30fps, running in the browser right now. And we're honest
> about exactly what we measured."

Point to the live pose visual. Move to the Gym card.

### Beat 2 — The magic: live skeleton (20s)
Click **Back Squat** → **Start camera**. Stand side-on.
> "That's MediaPipe pose in the browser — no server round-trip. Watch the live rep
> counter and the pose-confidence badge update as I move."

Do one slow half-squat so the skeleton + ghost metrics visibly react. **This is the
wow moment — let them see it track in real time.**

### Beat 3 — Record + the honest report (40s, talk through the spinner)
Click **Record** → do **2–3 clean squats** → **Stop & Analyse** → **Analyse my form**.
While the staged progress runs:
> "It just captured a WebM clip and is running the *heavy* pose model server-side —
> the same 33 landmarks, but full-precision — then a per-rep biomechanics pass."

When the report lands, point to:
1. **Scorecard** — "3 reps detected, 3 full valid reps."
2. **A rep card** — "Tempo ratio 2.0 — controlled lowering. Range of motion, pose
   confidence — every number carries a status."
3. **The Verified badge + calibration note** — "Reference ranges are cited from NSCA
   and ACSM literature. Where we *don't* have a reference, we say so — we never
   invent a grade. That's why a coach can trust it."
4. **Live Counter Check (parity probe)** — "We ran pose twice — fast in your browser,
   precise on the server — and report the numeric agreement. No other open-source
   sports tool does this."

### Beat 4 — Leaderboard, the product (15s)
Open `/leaderboard`. Refresh.
> "Your session just persisted and ranked — here you are. Best form indices per
> exercise. The ranking is a transparent, measured-only index — it even refuses to
> fake a form grade."

(Invite a judge to do a squat → they appear on the board. Huge engagement moment.)

### Beat 5 — The sponsor + agent story (20s)
> "The whole backend — Postgres schema, auth, clip storage — was provisioned **live
> by an AI agent through InsForge's MCP**: the agent operates the backend. And the
> coaching cues are **grounded in real sources via You.com Search** — every cue links
> to live sports-science references you can click."

(If a key isn't live yet: demo the deterministic coaching + say "grounding flips on
with the You.com key — here's the cited version," and show a prepared example.)

### Close (10s)
> "Real-time pose, honest measurement, an agent-run backend, and grounded coaching —
> production-deployed, today. That's Laksh.ai."

---

## Judge Q&A crib (they WILL probe — answer confident + honest)

- **"Is this real-time / on-device?"** → "Yes. Pose runs in-browser via MediaPipe
  Tasks (WASM). The report is a second, heavier server pass — we show the agreement
  between the two."
- **"How accurate is it?"** → "Pose confidence is reported per rep from the model's
  own landmark visibility. We don't claim accuracy we can't measure — uncalibrated
  metrics are shown without a grade. That honesty is the point."
- **"What's the moat?"** → "Real-time pose that actually works + a measurement spine
  where every number is auditable to a frozen taxonomy + calibration manifest (SHAs
  in the report). Reproducible byte-for-byte."
- **"Business?"** → "B2C form coaching + B2B (gyms, PT, sports clinics) that need
  defensible, audit-trail biomechanics, not a black box."
- **"How'd you build it so fast?"** → "Agent-operated infra (InsForge MCP) +
  grounded generation (You.com) — the agent provisioned and wired the backend."

---

## Failure recovery
- **Backend slow / first request:** "The heavy model is warming up." Wait 20s; if
  still stalling, play the **fallback clip**.
- **Skeleton not tracking:** step back, fix lighting, Cmd+Shift+R once. Worst case,
  fallback clip.
- **WiFi:** switch to the localhost fallback (pre-started) or hotspot.
- **Report shows Partial/Unknown:** re-record with better framing; or narrate it as
  "the system is telling us the framing degraded these reps — that's the honesty
  contract working," then re-run.
