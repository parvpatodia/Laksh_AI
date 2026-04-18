# Laksh.ai -- Dual-Path Sports Biomechanics with Honest Calibration
## A Measurement Spine with Explicit Uncertainty and a Realtime Parity Probe

**Parv Patodia** | Khoury College of Computer Sciences | Northeastern University
`github.com/parvpatodia/Laksh_AI`  |  `https://laksh-ai.vercel.app`

---

## ABSTRACT

Pose estimation extracts landmarks from video; it does not produce biomechanics.
We built Laksh.ai, a web-deployed gym biomechanics system that closes this gap
through a layered measurement spine: every measured quantity carries an explicit
status (`valid`/`degraded`/`unknown`) and calibration provenance
(`uncalibrated_v0` until labeled cohort data exists). To enable real-time judge
interaction, we run a lightweight JavaScript rep counter in the browser alongside
the canonical Python backend, then report a numeric parity probe (p90 absolute
delta) between the two paths on the same clip. This dual-path architecture is the
system's primary research contribution: it quantifies the accuracy cost of
real-time convenience and makes that cost auditable per clip.

---

## MOTIVATION

**The measurement gap.** Over 1 billion gym memberships worldwide. Personal
coaching cannot scale 1:1 to that population. Computer vision reaches smartphone
scale -- but existing sports AI systems make one of two errors:

1. **Vaporware feedback.** Hardcoded ideal ranges ("knee angle should be 90 deg")
   with no labeled evidence, no calibration provenance, no uncertainty. The system
   looks confident. The user trusts it. The number is invented.

2. **Detection only.** Outputs raw landmarks with no biomechanical interpretation --
   no rep segmentation, no phase timing, no quality assessment.

**Central question:** Can we build a measurement pipeline that is simultaneously
honest about uncertainty (no fake calibration), real-time (sub-frame feedback for
a live demo), and numerically auditable (we report how well the two paths agree)?

---

## SYSTEM ARCHITECTURE

```
Browser (https://laksh-ai.vercel.app -- Next.js 14, Vercel)
  |
  |-- getUserMedia -> PoseCamera component
  |     MediaPipe Tasks Vision JS (lite model, ~3M params)
  |     33 landmarks @ ~30 FPS, LIVE_STREAM mode
  |     |
  |     +-- repCounter.ts (EMA smoother, sign-change peak detect)
  |           -> GhostRepMetrics per rep
  |              { value, unit, status="valid|degraded|unknown",
  |                reason_codes=["realtime_preview"] }
  |           -> GhostMetricsPanel (live display)
  |
  |-- MediaRecorder -> WebM blob (on "Stop & Analyse")
  |     + ghost rep vectors (JSON)
  |
  v  POST /v1/analyze/gym/video  (multipart)
     https://laksh-api.fly.dev  (FastAPI, Fly.io, 1 GB shared CPU)
       |
       +-- extract_canonical_frames()
       |     MediaPipe Pose Landmarker heavy (~5M params, VIDEO mode)
       |     -> canonical frames list[frame_dict | None]
       |
       +-- analyze_gym_clip()
       |     rep_segmenter.py     (scipy.signal.find_peaks, NaN-safe)
       |     -> RepSpan[] with VALID | DEGRADED | UNKNOWN
       |     rep_features.py
       |     -> RepFeatureVector (7 fields, each FieldValue{value,unit,status,reason_codes})
       |     calibration_v0.py   (versioned JSON, policy-enforced honesty)
       |     -> CalibrationBlock (uncalibrated_v0 | cited)
       |
       +-- probe_reps()  [if ghost vectors submitted]
             pool valid-status pairs across matched reps
             -> ParityProbeModel { p90_abs_delta, max_abs_delta, status,
                                   fields_compared }
       |
       v  AnalyzeResponseModel (schema v2.0.0)
          -> CanonicalReport (per-rep cards, status chips, parity probe, provenance)
```

---

## METHODS

### Measurement Spine

Every measured quantity in the response envelope carries:

```json
{
  "value":        1.73,
  "unit":         "ratio",
  "status":       "valid",
  "reason_codes": []
}
```

Status is not cosmetic -- it propagates. A rep with `primary_joints_missing_frac`
> 25% receives `rep_status = "degraded"`. Downstream code (calibration, parity
probe) filters on status before computing statistics. Degraded measurements are
not silently promoted to valid.

### Rep Segmenter

`rep_segmenter.py` extracts a 1D signal per exercise:

| Exercise type     | Signal                                      |
|-------------------|---------------------------------------------|
| cyclic_vertical   | Mean hip y-coordinate (squats, hinges)      |
| cyclic_angle      | Elbow/shoulder/wrist triplet angle (press)  |
| duration/isometric| Body-midline variance proxy (plank)         |

`scipy.signal.find_peaks` with prominence filtering detects rep extrema.
NaN-safe centered moving average fills short landmark-drop gaps before peak
detection. Every RepSpan carries a status tag.

### Calibration Honesty Contract

`calibration_v0.py` enforces a policy at Pydantic model instantiation time:
a `CalibrationEntry` with `evidence_status = "uncalibrated_v0"` cannot carry
non-null `reference_ranges`. The validator raises at serialisation -- it is
impossible to ship vaporware reference bands without changing `evidence_status`
to `"cited"` and supplying a valid `evidence_source` pointer.

Current deployment: all 12 exercises ship as `uncalibrated_v0`. When labeled
cohort data (Milestone 2) produces reference distributions, the entry flips to
`"cited"` with a SHA-traceable link to the scorecard.

### Dual-Path Parity Probe

`app/parity/realtime.py` -- `probe_reps(realtime_vecs, canonical_vecs)`:

1. Match ghost and canonical reps by `rep_index`.
2. For each matched pair, collect all fields where both sides have `status="valid"`.
3. Pool absolute deltas across all matched (field, rep) pairs.
4. Compute `p90_abs_delta` (90th percentile) and `max_abs_delta`.
5. Apply thresholds: p90 <= 0.15 and max <= 0.50 -> `within_tolerance`;
   else `outside_tolerance`; fewer than `MIN_COMPARABLE_FIELDS` valid pairs
   -> `insufficient_data`.

Default thresholds are conservative for v0 (no calibrated tolerance exists).
They are keyword-overridable to tighten as reference data accumulates.

---

## RESULTS

| Metric                         | Value                                  |
|--------------------------------|----------------------------------------|
| Deterministic tests passing    | 205 (1 skipped: GPU path)              |
| Test runtime (CPU, no GPU)     | ~28 s                                  |
| Realtime pose FPS (browser)    | ~30 FPS (MediaPipe lite, LIVE_STREAM)  |
| Canonical analysis latency     | 8-15 s (5-second 720p clip, 1 GB CPU)  |
| Learned parameters in pipeline | 0 (segmenter + features are deterministic algorithms) |
| Learned parameters (pose model)| ~5M (MediaPipe heavy, server)          |
|                                | ~3M (MediaPipe lite, browser)          |
| Features per rep               | 7 (all FieldValue with status)         |
| Exercises supported            | 12 compound movements                  |
| Schema version (v1 API)        | 2.0.0                                  |
| Parity probe fields compared   | Up to 5 (duration, eccentric, concentric, tempo_ratio, min_visibility) |
| Upload size limit              | 50 MB                                  |

**Qualitative parity observation:** On a well-lit single-camera back squat from
~2 metres, the parity probe returns `within_tolerance` with p90 delta typically
in the 0.05-0.15 range on tempo_ratio_ecc_over_con. Degraded lighting or partial
occlusion reduces `fields_compared` to 1-2, triggering `insufficient_data`.
Quantified cohort statistics are a Milestone 2 deliverable.

---

## KEY RESULT: CALIBRATION OUTPUT EXAMPLE

```json
"calibration": {
  "exercise_id":      "back_squat",
  "evidence_status":  "uncalibrated_v0",
  "evidence_source":  null,
  "comparable_fields": ["tempo_ratio_ecc_over_con", "rep_duration_s"],
  "per_rep": [{
    "rep_index": 0,
    "fields": {
      "tempo_ratio_ecc_over_con": {
        "status":          "no_reference_yet",
        "value":           1.73,
        "range":           null,
        "evidence_status": "uncalibrated_v0",
        "evidence_source": null
      }
    }
  }]
}
```

The system knows 1.73. It does not claim 1.73 is good. That is the correct
scientific posture at this stage. The contract prevents the alternative.

---

## LIMITATIONS AND FUTURE WORK

**Current limitations:**
- Canonical analysis is synchronous -- 8-15 s blocks the HTTP connection.
  ADR 0003 specifies the async job-queue design (submit + poll); not yet implemented.
- No labeled validation of rep segmenter accuracy (IoU / F1 against ground truth).
  Milestone 2 will provide this.
- All 12 exercises ship `uncalibrated_v0` -- no reference ranges yet.
- Parity probe thresholds (p90 <= 0.15) are heuristic, not derived from clinical
  tolerance data.
- Single-camera 2D skeleton: no depth estimation, no multi-joint occlusion recovery
  beyond missingness flagging.

**Milestone 2 (next):**
1. Labeled rep boundary annotations: IoU / F1 harness for segmenter.
2. Biomechanist review of gym clips: populate cited reference ranges.
3. Parity probe threshold calibration from measured lite-vs-heavy agreement distribution.
4. Async job queue (ADR 0003): POST returns job_id, GET polls for result.
5. Basketball v1 schema migration: unify KinematicAnalyzer output into FieldValue schema.

---

## TECH STACK

| Layer         | Technology                                                      |
|---------------|-----------------------------------------------------------------|
| Frontend      | Next.js 14 (App Router), TypeScript, Tailwind CSS               |
| Deployment    | Vercel (frontend), Fly.io shared-cpu-1x 1 GB RAM (backend)      |
| Realtime pose | MediaPipe Tasks Vision JS (lite), WebAssembly                   |
| Capture       | MediaRecorder API (WebM), FormData upload                       |
| Backend       | FastAPI, Python 3.11, Pydantic v2                               |
| Server pose   | MediaPipe Pose Landmarker heavy, OpenCV (frame decode)          |
| Segmenter     | scipy.signal.find_peaks, numpy (NaN-safe moving average)        |
| Schema        | Pydantic models, SHA-256 pinned exercise + calibration manifests |
| Tests         | pytest (205 deterministic, no GPU required)                     |
| Package mgr   | uv                                                              |

---

## REFERENCES

- MediaPipe Tasks Vision JS: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/web_js
- ADR 0004 (this repo): Realtime-preview / canonical-backend dual-path with parity probe
- ADR 0003 (this repo): Observability and async job queue design
- AthletePose3D (CVSports CVPR 2025): cites the offline-vs-realtime gap in athletic pose literature
- scipy.signal.find_peaks: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
