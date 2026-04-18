# ADR 0004: Realtime-preview / canonical-backend dual-path with parity probe

## Status

**Accepted.** Implementation in `app/parity/realtime.py`.  
Live demo surface: `web/` (Day 4-8).

## Context

The Northeastern Research Showcase requires judges to walk up, perform a
movement (basketball jump-shot or gym compound lift), and immediately see
metrics.  Two competing constraints pull in opposite directions:

1. **Immediacy** - Judges need visual feedback within a frame or two. Running
   the full Python pipeline (MediaPipe heavy model + segmenter + calibration)
   round-trips to the server. On a showcase WiFi network that latency is
   unpredictable.

2. **Rigor** - The research contribution is the per-field measurement spine
   with `{value, unit, status, reason_codes}` and the honest calibration
   policy.  Showing numbers that were computed by lightweight JavaScript with
   no uncertainty attribution would contradict that contribution.

Existing art handles these two goals separately: browser pose libraries ship
"use it" without any offline-canonical comparison; offline biomechanics
frameworks ship without realtime. No open-source sports-analysis project
known to us reports the numerical agreement between the two paths.

## Decision

### Two-path architecture

```
Browser
  getUserMedia -> MediaPipe Tasks Vision JS (LIVE_STREAM, lite)
    -> 33-landmark pose @ ~30 FPS
    -> repCounter.ts: lightweight signal (hip_y for squats,
       elbow_angle for bicep, wrist_y for basketball)
    -> ghost metrics with reason_codes=["realtime_preview"]
       shown immediately in UI

  On "End Set" / "Capture Rep":
    -> MediaRecorder: last N seconds of WebM uploaded to /v1/analyze/{sport}

Server (canonical backend)
  -> frames_json or heavy MediaPipe VIDEO decode
  -> full gym/basketball pipeline
  -> returns AnalyzeResponseModel with analysis_mode="canonical_backend"
  -> parity_probe block comparing realtime ghost metrics to canonical values
```

### Parity probe

`app.parity.realtime.compare_feature_vectors` computes per-field absolute
delta between the two vectors. It pools all valid-status pairs, reports:

- `fields_compared` - which fields were present and valid on both sides
- `max_abs_delta` - worst-case absolute difference
- `p90_abs_delta` - 90th-percentile absolute difference (mirrors ADR 0002)
- `status` - `within_tolerance` | `outside_tolerance` | `insufficient_data`

Default thresholds: p90 <= 0.15 (absolute units), max <= 0.50.  These are
conservative for v0 since no calibrated tolerance exists yet.  The thresholds
are keyword-overridable to allow tightening as reference data accumulates.

The parity probe is optional in the response envelope (`parity_probe: null`
when no realtime vector was provided, e.g. pure CLI usage).

### Labeling contract

| `analysis_mode`       | `reason_codes` on fields    | Trust level            |
|-----------------------|-----------------------------|------------------------|
| `realtime_preview`    | `["realtime_preview"]`      | Ghost / indicative     |
| `canonical_backend`   | per taxonomy (existing)     | Authoritative          |

Ghost metrics are never stored in R2 as a result -- only the canonical pass
is persisted.  The realtime vector is ephemeral in the browser session.

### Multi-rep aggregation

`probe_reps` matches reps by `rep_index`, pools all per-field deltas across
all rep pairs, then runs the same p90 / max statistics on the pooled set.
This gives a single aggregate `parity_probe` block per clip rather than one
block per rep, keeping the response schema flat.

## Consequences

**Positive**

- Judges see live feedback immediately (UX goal met).
- The canonical backend result is authoritative and carries full measurement
  spine + calibration + provenance (rigor goal met).
- The parity probe makes numerical agreement explicit and auditable --
  a judge can ask "how close was the realtime estimate to the server's
  answer?" and get a factual number.
- The realtime path is deliberately simple (no physics, no calibration,
  no rep segmenter) so its systematic biases are clearly documented in
  `reason_codes=["realtime_preview"]` rather than hidden.

**Negative / risks**

- Two codebases for the same domain logic: `repCounter.ts` (JS) and the
  Python gym pipeline. They can drift. Mitigation: the parity probe running
  on every capture surfaces drift numerically; if `status="outside_tolerance"`
  the UI flags it.
- The canonical backend pass adds 1-3s latency after the judge stops. This
  is acceptable for the showcase setting (judge is still standing there) but
  would need an async job queue for production (see ADR 0003).
- The realtime-vs-canonical comparison is only as meaningful as the features
  that survive the `status="valid"` filter on both sides.  In a noisy webcam
  environment many fields may be `degraded`, shrinking `fields_compared` to
  below `MIN_COMPARABLE_FIELDS` and triggering `insufficient_data`.

## Reference

- ADR 0002: offline canonical-vs-legacy gate (same p90 statistics, same
  Tukey-fence philosophy, offline context).
- ADR 0003: async job queue (needed if canonical pass exceeds 3s budget).
- `app/parity/realtime.py`: implementation with full docstring.
- `tests/test_parity_realtime.py`: 15 deterministic tests.
- [MediaPipe Tasks Vision JS guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/web_js)
- [AthletePose3D CVSports CVPR 2025](https://github.com/calvinyeungck/AthletePose3D):
  cites similar "offline reference vs real-time" gap in athletic pose literature.
