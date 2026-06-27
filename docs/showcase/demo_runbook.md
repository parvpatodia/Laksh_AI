# Live Demo Runbook -- Laksh.ai Showcase

**System: Next.js 14 frontend (Vercel) + FastAPI backend (Fly.io)**
**Demo surface: https://laksh-ai.vercel.app  |  fallback: http://localhost:3000**

---

## Pre-showcase checklist (do this before stepping up)

- [ ] Laptop charged >= 80%. Bring charger. Reserve outlet spot.
- [ ] Browser tabs open and loaded (Chrome or Safari, NOT Firefox -- MediaRecorder
      WebM compatibility is better in Chrome):
      - Tab 1: https://laksh-ai.vercel.app  (pinned, already at homepage)
      - Tab 2: https://laksh-api.fly.dev/v1/health  (confirms backend is live)
      - Tab 3: http://localhost:3000  (fallback, see Fallback section below)
- [ ] Backend health check: navigate to https://laksh-api.fly.dev/v1/health
      Confirm `"status": "ok"`. If you see a 503, wait 30 s -- Fly cold-start.
- [ ] Camera permission already granted on this machine / browser profile.
      Open https://laksh-ai.vercel.app/gym?exercise=back_squat, click "Start camera"
      once, confirm skeleton appears, then close and re-stage tab at homepage.
- [ ] Room light: face the light source. Avoid strong backlight (washes skeleton).
- [ ] Font size: N/A -- this is a web app, not a terminal. Browser zoom at 100%.
- [ ] Tested WiFi: load the Vercel URL from venue WiFi once. If slow, use hotspot.

---

## The 90-second demo flow

### T+0:00 -- Homepage (30 s)

Open **Tab 1** (https://laksh-ai.vercel.app).

Say:
> "This is Laksh.ai -- sports biomechanics, honest by construction.
>  Three research contributions on this landing page."

Point to the three cards at the bottom:

1. **Measurement spine**: "Every number carries `value, unit, status, reason_codes`.
   No bare floats. You can see whether a measurement is `valid`, `degraded`, or
   `unknown` -- not just what the number is."

2. **Calibration honesty**: "`uncalibrated_v0` entries cannot claim reference
   ranges. The policy is enforced at serialisation time. The system cannot silently
   invent ideal bands."

3. **Parity probe**: "We run pose in the browser in real time AND on the server
   with the full model. We report the numerical delta between the two. That's
   the novel contribution -- no other open-source sports pipeline does this."

### T+0:30 -- Gym page + exercise picker (10 s)

Click **Gym** card.

Say:
> "Twelve compound movements. I'll use Back Squat."

Click **Back Squat** -- URL becomes `/gym?exercise=back_squat`.

### T+0:40 -- Camera on, skeleton overlay (10 s)

Click **"Start camera"** button (top-right).

Say:
> "MediaPipe Tasks Vision JS loads in the browser -- that's the lite model,
>  33 landmarks at ~30 FPS. No server round-trip for this."

Point to skeleton overlay on video. Show a partial squat to confirm joints track.

Point to **Ghost metrics** panel (left side):
> "Rep count, current phase (eccentric / concentric / rest), and signal level --
>  all with `realtime_preview` label so the UI distinguishes them from the
>  authoritative backend result."

### T+1:00 -- Do 2-3 squats, show live metrics (20 s)

Perform 2 or 3 slow back squats in front of the camera.

While squatting:
> "The rep counter is detecting phase from the hip-y signal derivative.
>  Each completed rep gives duration, eccentric phase, concentric phase,
>  and tempo ratio -- all stamped `realtime_preview`."

After reps, point to Ghost metrics panel showing:
- Rep count: 2 or 3
- Last rep's `rep_duration_s`, `eccentric_duration_s`, `tempo_ratio_ecc_over_con`
- Status chips: `valid` or `degraded`

### T+1:20 -- Record, Stop & Analyse (20 s)

Click **"Record"** button inside the PoseCamera component. Do one clean squat.
Click **"Stop & Analyse"** (or camera stop button).

Say:
> "MediaRecorder just captured a WebM clip. Now I'll send it to the backend."

Click **"Analyse clip"** button in the Canonical result panel (right side).

While the spinner runs (~10-15 s):
> "The backend is at laksh-api.fly.dev -- FastAPI on Fly.io, 1 GB shared CPU.
>  It's running MediaPipe heavy model in VIDEO mode, then the full gym pipeline:
>  rep segmenter, feature extractor, calibration layer."

### T+1:40 -- Show canonical result (30 s)

When result appears, point to **Canonical result** panel:

1. **Rep cards**: "Each rep is a card -- start frame to end frame, status chip.
   Inside: `rep_duration_s`, `tempo_ratio_ecc_over_con`, all with status."

2. **Parity probe block**: "Here is the parity probe. It compares the ghost
   metrics I showed you 30 seconds ago to the canonical backend result.
   `p90_abs_delta` is the 90th-percentile absolute difference across all
   matched fields. If it says `within_tolerance` -- the fast browser path
   agreed with the slow heavy model. If `outside_tolerance` -- there's
   systematic divergence and we can see which fields caused it."

3. **Calibration notice**: "The calibration block says `uncalibrated_v0`.
   That is honest: we measured tempo ratio, we do NOT have labeled cohort
   data to say whether 1.7 is good or bad for this movement. The contract
   prevents us from inventing a reference range."

4. **Provenance**: Click the "Provenance" disclosure. Show `exercise_manifest`
   and `calibration_manifest` SHA prefixes, schema version 2.0.0.

Done. Step back.

---

## Failure recovery

### Backend returns 503 or times out
- Wait 30 s. Fly.io cold-starts take up to 20 s on the first request.
- If still failing after 60 s, switch to fallback (see below).
- Say: "The backend is waking up -- first request to a sleeping Fly machine
  can take 15-30 seconds."

### Camera permission denied
- Browser address bar > lock icon > Camera > Allow.
- If Chrome blocks: Settings > Privacy > Camera > add laksh-ai.vercel.app.

### Skeleton not appearing / MediaPipe load failure
- Check browser console for CORS or WASM errors.
- Try force-refresh (Cmd+Shift+R). The lite model file is ~5 MB and caches after first load.
- Fallback: show the skeleton from a second device (phone) at localhost:3000.

### "Analyse clip" fails with API error
- Check Tab 2 (health endpoint) -- if backend is down, use local backend fallback.
- Common error: 413 if clip is > 50 MB (shouldn't happen for a 5-second clip).
- Common error: 422 if MediaPipe could not extract frames (bad lighting, extreme
  occlusion). Re-record with better framing and resubmit.

### WiFi too slow for Vercel frontend load
- Switch to localhost:3000 (local Next.js dev server, pre-started -- see below).

---

## Offline / localhost fallback

**Pre-start before leaving for the venue:**

```
# Terminal 1 -- backend
cd /Users/parvpatodia/Laksh_AI
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2 -- frontend pointing at local backend
cd /Users/parvpatodia/Laksh_AI/web
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Open http://localhost:3000 in Tab 3. Demo flow is identical. The only visible
difference: the URL shows localhost and no Vercel banner.

Verify backend health before the demo:
```
curl http://localhost:8000/v1/health
```
Expect: `{"status":"ok","v1_schema_version":"2.0.0",...}`

---

## Print-ready command card (cut out and keep in pocket)

```
PRODUCTION URLS
  Frontend : https://laksh-ai.vercel.app
  Backend  : https://laksh-api.fly.dev/v1/health

LOCAL FALLBACK
  Backend  : cd /Users/parvpatodia/Laksh_AI
             uv run uvicorn app.main:app --port 8000

  Frontend : cd /Users/parvpatodia/Laksh_AI/web
             NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
             => http://localhost:3000

DEMO PATH
  / -> click Gym -> click Back Squat
  -> Start camera -> do 2-3 squats -> observe ghost metrics
  -> Record -> squat -> Stop & Analyse
  -> Analyse clip -> show canonical result + parity probe

BACKEND ENDPOINTS
  GET  /v1/health
  GET  /v1/sports
  POST /v1/analyze/gym/video    (multipart: exercise_id, video, realtime_vectors_json)
  POST /v1/analyze/gym          (JSON: frames_json source)
```
