# Judge Q&A Crib Sheet

Anticipated hard questions from technical and non-technical judges.
**Answer style: short direct answer first, depth only if they push.**

---

## ON THE CORE SYSTEM

**Q: Why not just use an existing pose estimation library and call it done?**

Pose estimation gives you coordinates. Biomechanics is about what those coordinates
mean over time - when does a rep start, how long is the eccentric phase, what is
the range of motion. No general-purpose library does that. The measurement spine
is the contribution.

---

**Q: What makes your rep segmentation better than a simple threshold?**

Three things. First: scipy.signal.find_peaks with prominence filtering avoids
noise-triggered false positives a threshold would catch. Second: NaN-safe
smoothing handles frames where MediaPipe drops landmarks (common at the bottom
of a squat where limbs overlap). Third: every rep gets a status - VALID,
DEGRADED, or UNKNOWN - so downstream code knows when to trust the segmentation
and when to flag it for review. A threshold gives you a boundary and no
uncertainty estimate.

---

**Q: Why seven features? Why not more?**

These seven are the ones we can measure reliably from a single-camera 2D skeleton
without additional sensors or calibration rig. More features would require either
multi-camera setup (depth estimation) or EMG sensors (muscle activation). We're
scoped to smartphone video. The seven fields cover the things a coach actually
looks at: duration, phase balance, depth, visibility quality.

---

**Q: Your calibration is all uncalibrated_v0. Isn't that just shipping nothing?**

It's shipping the container that will hold the evidence when we have it, along with
a contract that prevents anyone from inventing numbers to fill that container.
The alternative - hardcoding ideal bands with no evidence - is what every other
system does. We chose to ship honest infrastructure over confident theater.

---

**Q: How do you plan to get labeled data for calibration?**

Milestone 2: a labeled subset from `evaluation/gym_manifest_hard.csv`. A
biomechanist (or experienced coach) reviews clips and annotates rep boundaries
and quality scores. We compute reference ranges from the labeled distribution.
Those ranges commit to the JSON with `evidence_status = "cited"` and a pointer
to the exact scorecard that justified them. Every range has a traceable source.

---

## ON TECHNICAL CHOICES

**Q: Why MediaPipe instead of RTMPose or ViTPose?**

MediaPipe is the only pose backend that runs on-device on an iPhone without a
server. That matters for the product vision (real-time, no upload). RTMPose
is better on gym clips - higher accuracy, better occlusion handling - and we
have infrastructure for it (ADR 0002, Phase B/C). The canonical joint vocabulary
is backend-agnostic so we can swap backends without changing the measurement
spine.

---

**Q: scipy.signal.find_peaks is not ML. Why not train a rep detector?**

Two reasons. First: we don't have a labeled dataset yet - Milestone 2 is building
it. You can't train a detector without ground truth rep boundaries. Second: the
deterministic segmenter gives us a reproducibility baseline. When we do train
a learned detector, we'll compare its IoU against the find_peaks baseline. If it
doesn't beat it, we don't ship it.

---

**Q: How do you handle occlusion? What if MediaPipe misses a joint?**

Three layers. The segmenter's NaN-safe smoothing uses linear interpolation to
fill short gaps and weights the signal by missingness. The feature extractor
tracks `primary_joints_missing_frac` per rep - if it exceeds 25% the rep is
DEGRADED. The calibration layer emits `"unavailable"` for any field whose
underlying measurement had unknown status. Occlusion is surfaced, not silently
filled.

---

**Q: What is the Tukey fence p90 gate?**

When we flip to canonical joints (ADR 0002 Phase C), we want to verify the new
path doesn't change biomechanical metrics more than expected. The parity report
runs both paths on the same video, computes per-frame joint position differences,
and uses Tukey's method to identify outlier frames. If p90 delta exceeds 2
degrees, the flag fails. This gates a production flip without needing labeled
"correct" data - just a stability signal.

---

## ON BUSINESS AND VISION

**Q: How is this different from Whoop / Strava / Garmin?**

Those systems use accelerometers and heart rate - they know you moved, not how
you moved. Laksh.ai uses computer vision to measure form, not just effort.
The nearest equivalent is Dartfish or Hudl, which are manual video review tools
for elite sports. We're building the automated version for everyday athletes
with a smartphone.

---

**Q: What is your go-to-market?**

Out of scope for this showcase - we're presenting research infrastructure.
But: the measurement spine is the defensible asset. Any coaching app can add
a camera; very few have a calibrated, uncertainty-aware biomechanics layer
underneath. The data moat comes from labeled clips tied to coaching outcomes.

---

**Q: Can this work in real time?**

The segmenter is offline (post-hoc over a clip). Real-time would require a
streaming segmenter that detects rep boundaries frame-by-frame. That's a
different architecture - possible with a learned detector trained on the Milestone
2 labels. The current system is designed for the async case: record, upload,
analyze.

---

**Q: How many parameters does your model have?**

The segmenter and feature extractor have zero learned parameters - they are
deterministic algorithms. MediaPipe Pose Landmarker (heavy variant) is ~5M
parameters. The system's value is in the pipeline architecture and calibration
contract, not in model parameters.

---

## IF THINGS GET UNCOMFORTABLE

**"This seems like a lot of infrastructure for not much output."**

> The output is a measurement you can trust. A system that tells you a number
> and hides its uncertainty is worse than no system. We built the foundation
> that lets the number mean something when the calibration is done.

**"Why should a judge care about your testing setup?"**

> 187 deterministic tests mean any collaborator, any future student working on
> this project, can change the segmenter and immediately see if they broke
> something. Research that can't be reproduced isn't research.

**"Have you validated this on real athletes?"**

> Not yet - that is Milestone 2. What we have validated is that the pipeline
> produces sensible output on synthetic and gym-clip data, and that the
> infrastructure is ready to receive labeled evidence. Premature validation
> on a small sample would give us false confidence.

---

## SAFE CLOSING LINE (use if time runs out mid-answer)

> "The short answer is: we don't know yet, and the system says so. That's the
> point. When we have the evidence, the architecture is ready to use it."
