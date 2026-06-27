# Judge Q&A Crib Sheet

Anticipated hard questions from technical and non-technical judges.
**Answer style: short direct answer first, depth only if they push.**

---

## ON THE CORE SYSTEM

**Q: Why not just use an existing pose estimation library and call it done?**

Pose estimation gives you coordinates. Biomechanics is about what those coordinates
mean over time -- when does a rep start, how long is the eccentric phase, what is
the range of motion. No general-purpose library does that. The measurement spine
is the contribution.

---

**Q: What makes your rep segmentation better than a simple threshold?**

Three things. First: scipy.signal.find_peaks with prominence filtering avoids
noise-triggered false positives a threshold would catch. Second: NaN-safe
smoothing handles frames where MediaPipe drops landmarks (common at the bottom
of a squat where limbs overlap). Third: every rep gets a status -- VALID,
DEGRADED, or UNKNOWN -- so downstream code knows when to trust the segmentation
and when to flag it for review. A threshold gives you a boundary and no
uncertainty estimate.

---

**Q: Why seven features? Why not more?**

These seven are the ones we can measure reliably from a single-camera 2D skeleton
without additional sensors or calibration rig. More features would require either
a multi-camera setup (depth estimation) or EMG sensors (muscle activation). We are
scoped to a standard webcam or smartphone. The seven fields cover the things a
coach actually looks at: duration, phase balance, depth, visibility quality.

---

**Q: Your calibration is all uncalibrated_v0. Isn't that just shipping nothing?**

It is shipping the container that will hold the evidence when we have it, along with
a contract that prevents anyone from inventing numbers to fill that container.
The alternative -- hardcoding ideal bands with no evidence -- is what every other
system does. We chose honest infrastructure over confident theater.

---

**Q: How do you plan to get labeled data for calibration?**

Milestone 2: a biomechanist or experienced coach reviews clips from
`evaluation/gym_manifest_hard.csv` and annotates rep quality scores. We compute
reference ranges from the labeled distribution. Those ranges commit to the JSON
with `evidence_status = "cited"` and an `evidence_source` pointer to the exact
scorecard and eval run that justified the numbers. Every range has a traceable source.

---

## ON THE DUAL-PATH ARCHITECTURE

**Q: Why does the web frontend matter for research? Couldn't you just publish a script?**

Two reasons. First: the web surface makes the measurement accessible to subjects
without a Python environment, which is required for a labeled cohort at Milestone 2.
Second: the browser's MediaPipe lite path is the only way to get real-time feedback
at ~30 FPS without a GPU on the server. If we published only a server script, we
could not run the parity probe -- there would be nothing to compare the canonical
result against. The frontend is what makes the dual-path contribution possible.

---

**Q: Why compare realtime vs canonical? What does that tell you scientifically?**

It tells you the systematic bias of the fast path. The browser repCounter is a
simple EMA smoother with sign-change peak detection and no rep segmenter. The
backend runs scipy.signal.find_peaks on smoothed landmarks from the heavy model.
If `p90_abs_delta` is small -- say 0.05 ratio units on tempo -- then the fast path
is a good proxy and could substitute for the full backend in latency-critical
settings. If `p90_abs_delta` is large -- 0.4+ -- then the lite model or the EMA
detector is introducing systematic error that would mislead a real-time coach.
We do not know which case is true until we run both on the same clip and compare.
That is the experiment.

---

**Q: What does p90 delta specifically tell you?**

It is the 90th-percentile absolute difference, pooled across all matched
(field, rep) pairs where both the ghost and canonical values had `status="valid"`.
The p90 mirrors the ADR 0002 gate used for the offline canonical-vs-legacy path.
Using p90 rather than mean makes the statistic robust to one-off large errors on
degraded reps -- a single DEGRADED rep with a large delta should not dominate
the summary if the other reps agreed well.

The `max_abs_delta` is also reported for the worst-case single field,
so a judge can see both the typical agreement and the tail.

---

**Q: Analysis takes 10-15 seconds. That seems slow. Why is it synchronous?**

Honest answer: the current endpoint blocks until MediaPipe heavy finishes,
which on a 1 GB Fly shared-CPU machine takes 8-15 s for a 5-second 720p clip.
This is acceptable for the showcase (the judge is still standing there).
ADR 0003 has the async design: split into `POST /analyze` (returns job_id
immediately) and `GET /jobs/{job_id}` (polls). The bottleneck is the sequential
MediaPipe decode -- splitting the endpoint does not change how long pose extraction
takes, but it unblocks the HTTP connection and allows the UI to show live progress.
We deliberately did not implement the async queue pre-showcase because it adds
Redis + worker container complexity without changing the latency for the demo case.

---

**Q: Why Next.js on Vercel? Why not just serve a static HTML file?**

Static HTML is what the old architecture used. The new system needs:
(1) server-side environment variables for the Fly.io API URL (not exposed to the
browser raw); (2) dynamic routing for `/gym?exercise=back_squat` with search-param
state; (3) lazy-loading PoseCamera so the ~5 MB WASM file only downloads when the
user navigates to a sport page. Next.js 14's App Router handles all three cleanly.
Vercel handles preview deploys on every PR, which is how we validated the Day-7
and Day-8 feature additions before the showcase. A judge can also load the URL on
their phone and interact with it -- a bare Python server on a laptop cannot do that.

---

**Q: Can someone else reproduce your results?**

Yes, with caveats. The measurement spine is deterministic: same video, same result.
The provenance block in every response includes a `git_commit_sha`, the SHA-256 of
the exercise manifest, and the SHA-256 of the calibration manifest. A reproducer
who checks out the same commit and pinned manifests gets the same pipeline.
The only non-deterministic element is the pose extractor: MediaPipe's CPU
interpolation can vary slightly across OS versions. We do not currently
SHA-pin the MediaPipe model weights, only the pipeline logic. That is on the Milestone 2
roadmap.

---

## ON TECHNICAL CHOICES

**Q: Why MediaPipe instead of RTMPose or ViTPose?**

MediaPipe Tasks Vision JS is the only pose backend that runs in the browser without
a server round-trip. That is a hard requirement for real-time ghost metrics.
On the server side, RTMPose and ViTPose have higher accuracy on gym clips (especially
occlusion recovery), and the backend is backend-agnostic: the canonical joint
vocabulary is defined by `exercises_v0.py`, not by the pose library, so swapping
backends does not change the measurement spine or calibration contract.

---

**Q: scipy.signal.find_peaks is not ML. Why not train a rep detector?**

Two reasons. First: we do not have a labeled dataset yet -- Milestone 2 is building
it. You cannot train a detector without ground-truth rep boundaries. Second: the
deterministic segmenter gives us a reproducibility baseline. When we do train a
learned detector, we will compare its IoU against the find_peaks baseline. If it
does not beat it on the labeled set, we do not ship it.

---

**Q: How do you handle occlusion? What if MediaPipe misses a joint?**

Three layers. The segmenter's NaN-safe smoothing uses linear interpolation to fill
short gaps and weights the signal by missingness. The feature extractor tracks
`primary_joints_missing_frac` per rep -- if it exceeds 25% the rep is DEGRADED.
The calibration layer emits `"unavailable"` for any field whose underlying
measurement had `status="unknown"`. Occlusion is surfaced at every layer, not
silently filled.

---

**Q: How many parameters does your model have?**

The rep segmenter and feature extractor have zero learned parameters -- they are
deterministic algorithms. MediaPipe Pose Landmarker heavy variant is approximately
5M parameters. The repCounter.ts in the browser uses MediaPipe Tasks Vision JS
lite variant (~3M parameters). The system's value is in the pipeline architecture
and calibration contract, not in model parameters.

---

**Q: How many tests do you have?**

205 tests pass deterministically; 1 is skipped (MediaPipe-GPU path, not available
on the demo machine). All 205 run in under 30 seconds on CPU with no GPU required.
The test suite covers the rep segmenter, feature extractor, calibration validator,
parity probe, API schema validation, and provenance builder.

---

## ON BUSINESS AND VISION

**Q: How is this different from Whoop / Strava / Garmin?**

Those systems use accelerometers and heart rate -- they know you moved, not how you
moved. Laksh.ai uses computer vision to measure form. The nearest equivalent is
Dartfish or Hudl, which are manual video review tools for elite sports.
We are building the automated version for everyday athletes with a webcam.

---

**Q: What is your go-to-market?**

Out of scope for this showcase -- we are presenting research infrastructure.
The measurement spine is the defensible asset. Any coaching app can attach a camera.
Very few have a calibrated, uncertainty-aware biomechanics layer underneath.
The data moat comes from the labeled clips tied to coaching outcomes that Milestone 2
begins collecting.

---

## IF THINGS GET UNCOMFORTABLE

**"This seems like a lot of infrastructure for not much output."**

> The output is a measurement you can trust. A system that tells you a number
> and hides its uncertainty is worse than no system. We built the foundation
> that lets the number mean something when the calibration is done.

**"Why should a judge care about your testing setup?"**

> 205 deterministic tests mean any collaborator or future student can change the
> segmenter and immediately see if they broke something. Research that cannot be
> reproduced is not research.

**"Have you validated this on real athletes?"**

> Not yet -- that is Milestone 2. What we have validated is that the pipeline
> produces sensible output on gym-clip data, that the parity probe runs
> end-to-end in the browser-to-server flow, and that the infrastructure is
> ready to receive labeled evidence. Premature validation on a small unlabeled
> sample would give us false confidence in the calibration, which the contract
> explicitly prevents.

**"Why build a web app instead of just the paper?"**

> Because a paper cannot run on a judge's phone. The web surface is how we
> demonstrate the parity probe live -- the ghost metrics accumulate while you
> watch, the canonical result arrives, and the delta is a real number from a
> real clip, not a synthetic fixture.

---

## SAFE CLOSING LINE (use if time runs out mid-answer)

> "The short answer is: we do not know yet, and the system says so. That is the
> point. When we have the evidence, the architecture is ready to use it."
