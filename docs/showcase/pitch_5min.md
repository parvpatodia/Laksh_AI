# 5-Minute Pitch Script -- Laksh.ai
## Research Showcase -- Northeastern University, Khoury College of Computer Sciences

**Total time: 5:00**
**Pace guide: ~130 words/min. Practice each block to its timestamp.**

---

### [0:00 - 0:30] HOOK

"Every serious sports-AI paper since 2017 starts the same way:
'We extract skeleton landmarks from video.'
And then it goes quiet about what comes next.

Landmarks are not biomechanics.
A hip joint coordinate is a float.
A measurement of eccentric-phase duration with an explicit uncertainty label
and a calibration provenance trail -- that is biomechanics infrastructure.

The gap between those two things is where every app that says
'AI coaching' lives. They have the landmarks. They hardcode the ranges.
Nobody is honest about the gap.

Laksh.ai starts at the gap."

*(pause 1 second)*

---

### [0:30 - 1:30] LIVE DEMO

*[Switch to browser -- Tab 1, https://laksh-ai.vercel.app]*

"This is the system running in production.
Next.js 14 on Vercel. FastAPI backend on Fly.io."

*[Click Gym -> Back Squat]*

"I'll do a back squat. The browser is running MediaPipe Tasks Vision JS --
the lite model, 33 landmarks at about 30 FPS. Watch the left panel."

*[Click Start camera. Do 2-3 squats.]*

"Ghost metrics: rep count, phase, tempo ratio -- all labeled
`realtime_preview` so you know these are indicative, not authoritative.

Now I record and send it to the server."

*[Click Record. One squat. Stop & Analyse. Click Analyse clip.]*

"The backend is running MediaPipe heavy model in VIDEO mode,
then the full gym pipeline. Takes 10-15 seconds. Here it comes."

*[When result loads, point to Canonical result panel.]*

"Per-rep cards. Each field has a status chip -- valid, degraded, or unknown.
And here: the parity probe. That block compares the ghost metrics
from 30 seconds ago to what the heavy model computed.
p90 absolute delta: that number is the reproducible research contribution."

---

### [1:30 - 2:30] RESEARCH CONTRIBUTION 1: MEASUREMENT SPINE

"Every measured number in the response envelope has this shape:

    {
      'value':        1.73,
      'unit':         'ratio',
      'status':       'valid',
      'reason_codes': []
    }

Not a bare float. A FieldValue. The status field is not cosmetic --
it propagates. If MediaPipe drops landmarks for 25% of a rep's frames,
that rep's `primary_joints_missing_frac` exceeds the threshold,
the rep status becomes `degraded`, and downstream code knows not to
treat it as authoritative.

The rep segmenter uses scipy.signal.find_peaks on a 1D signal extracted
from the skeleton -- hip y-coordinate for squats, elbow angle for pulls.
NaN-safe centered moving average handles frames where MediaPipe drops joints.

Seven features per rep. Duration, eccentric phase, concentric phase,
tempo ratio, signal amplitude, min visibility, missingness fraction.
All deterministic. Same video, same result, every run.

205 tests pass. Zero learned parameters in the measurement spine.
The value of the system is not in the model. It is in the pipeline."

---

### [2:30 - 3:30] RESEARCH CONTRIBUTION 2: CALIBRATION HONESTY CONTRACT

"Look at the calibration block in the response."

*[Point to the calibration notice in the UI, or read aloud:]*

    evidence_status: 'uncalibrated_v0'

"This is deliberate. The system measured tempo ratio at 1.73.
It does NOT say 1.73 is good or bad. Because we don't have labeled data yet.
No biomechanist has reviewed enough back squat clips to say
'this range is within normal' for this population.

And I have made it impossible to accidentally violate this contract.
The Pydantic model at serialisation time rejects any calibration entry
that has `evidence_status='uncalibrated_v0'` AND non-null reference ranges.
You cannot ship vaporware ranges silently. The code rejects the payload.

When labeled data exists -- when a biomechanist reviews clips from
`evaluation/gym_manifest_hard.csv` and annotates quality scores --
a calibration entry graduates from `uncalibrated_v0` to `cited`
with an `evidence_source` pointer to the exact scorecard and eval run.

No evidence, no range. That is the contract."

*(pause)*

---

### [3:30 - 4:30] RESEARCH CONTRIBUTION 3: DUAL-PATH PARITY PROBE

"The third contribution is the one that required building the web frontend.

Most biomechanics research either ships a real-time system with no
offline validation, or an offline pipeline with no real-time surface.
We built both, deliberately, and then compared them numerically.

The browser runs `repCounter.ts` -- an EMA smoother on the landmark stream,
sign-change peak detection, five ghost features per rep.
Label: `realtime_preview`.

The backend runs the full Python gym pipeline.
Label: `canonical_backend`.

When the clip is submitted, `app.parity.realtime.probe_reps` pools all
valid-status field pairs across matched reps and computes:

  - `p90_abs_delta`  -- 90th-percentile absolute difference
  - `max_abs_delta`  -- worst-case field
  - `status`         -- within_tolerance | outside_tolerance | insufficient_data

Default thresholds: p90 <= 0.15, max <= 0.50.

This gives a judge a factual number: the lite browser path agreed with
the heavy server path to within X units on these five fields.
That number is what makes the dual-path architecture a research contribution,
not just a UX choice."

---

### [4:30 - 5:00] WHAT'S NEXT / MILESTONE 2

"Milestone 2 is labeled data.

We need a biomechanist or experienced coach to annotate rep boundaries
and quality scores on clips from the gym manifest. Once that cohort exists:

- Calibration entries graduate from `uncalibrated_v0` to `cited`.
- The rep segmenter gets an F1 / IoU harness against labeled boundaries.
- The parity probe thresholds tighten as we know what tolerance is clinically meaningful.

The infrastructure is ready to receive that evidence.
The schema is already parameterised for it.
The constraint that blocks fake calibration is already enforced.

Laksh.ai is not trying to replace a coach.
It is the measurement layer between the camera and the coach --
honest, testable, and reproducible.

Frontend: https://laksh-ai.vercel.app
Backend:  https://laksh-api.fly.dev/v1/health
Code:     github.com/parvpatodia/Laksh_AI

I'm happy to go deep on any of the three contributions."

*(step back, stop)*

---

## Timing reference

| Segment              | Duration | Cumulative |
|----------------------|----------|------------|
| Hook                 | 0:30     | 0:30       |
| Live demo            | 1:00     | 1:30       |
| Measurement spine    | 1:00     | 2:30       |
| Calibration honesty  | 1:00     | 3:30       |
| Parity probe         | 1:00     | 4:30       |
| Next steps / close   | 0:30     | 5:00       |

## Rehearsal notes

- Slow down on "valid, degraded, unknown" -- say each word separately.
- Do not read the JSON blob during the demo. Point to specific fields, narrate.
- "No evidence, no range. That is the contract." -- let it land, then continue.
- If the parity probe shows `insufficient_data`: explain it as expected
  behavior when not enough fields survived the `valid` filter. Not a failure.
- If canonical analysis takes > 20 s: narrate the pipeline phases while waiting.
  Do not apologise for the latency -- cite ADR 0003 and say async queue is designed.
- Practice the squat rep count to match the ghost metrics panel update rate.
  Two clean reps at normal speed work better than three rushed ones.
