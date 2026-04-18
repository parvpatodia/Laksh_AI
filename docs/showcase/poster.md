# Laksh.ai - Scalable Sports Biomechanics Analysis
## A Measurement-First AI Pipeline for Gym Movement Quality

**Parv Patodia** | Khoury College of Computer Sciences | Northeastern University

---

## MOTIVATION

Over 1 billion gym memberships worldwide. Coaches cannot watch every rep of
every athlete. Poor form is invisible until injury happens.

Computer vision can extract a skeleton from video in real time. But a skeleton
is not biomechanics. The gap between "we have landmarks" and "we know if the
rep was good" is enormous - and almost no system is honest about that gap.

**Central question:** Can we build a measurement pipeline that is honest about
what it knows, what it doesn't know, and what evidence would change that?

---

## PROBLEM STATEMENT

Existing sports AI systems make one of two mistakes:

1. **Vaporware feedback** - hardcoded "ideal knee angle = 90 deg" with no
   labeled evidence, no uncertainty, no calibration provenance.
2. **Detection only** - outputs landmarks but no biomechanical interpretation.

Neither is trustworthy for coaching decisions.

---

## OUR CONTRIBUTION: THE MEASUREMENT SPINE

A layered gym biomechanics pipeline with explicit uncertainty at every step.

```
Video
  |
  v  MediaPipe Pose (12 joints, normalized image coordinates)
pose_adapter.py  -->  canonical frames  (list[frame_dict | None])
  |
  v  rep_segmenter.py  (scipy.signal.find_peaks, NaN-safe smoothing)
RepSpan[]  with status: VALID | DEGRADED | UNKNOWN
  |
  v  rep_features.py  (7-field per-rep vector)
RepFeatureVector  with per-field FieldValue{value, unit, status, reason_codes}
  |
  v  calibration_v0.py  (versioned reference-range config)
apply_calibration()  -->  no_reference_yet | within | outside | unavailable
```

### Layer 1 - Exercise Taxonomy (exercises_v0.py)
- 12 compound movements: squat, hinge, push, pull, lunge, isometric, carry
- SHA-256 pinned manifest (`evaluation/exercise_v0_manifest.json`)
- Each movement declares: camera view, rep signal type, primary joints
- No ideal angles - structural metadata only (GOALS.md calibration policy)

### Layer 2 - Rep Segmenter (rep_segmenter.py)
- `scipy.signal.find_peaks` for deterministic extrema detection
- Handles 4 signal types: cyclic_vertical, cyclic_angle, duration, gait_cadence
- NaN-safe centered moving average interpolation
- Per-rep status: VALID / DEGRADED (high missingness, truncated) / UNKNOWN

### Layer 3 - Per-Rep Feature Vector (rep_features.py)
7 features per rep, each with `FieldValue{value, unit, status, reason_codes}`:

| Field | Unit | Notes |
|---|---|---|
| rep_duration_s | s | Total rep time |
| eccentric_duration_s | s | Lowering phase |
| concentric_duration_s | s | Lifting phase |
| tempo_ratio_ecc_over_con | ratio | Eccentric control index |
| signal_amplitude | deg or normalized_y | Joint range of motion |
| primary_joints_min_visibility | visibility | Data quality floor |
| primary_joints_missing_frac | frac | Occlusion measure |

### Layer 4 - Calibration Config (calibration_v0.py)
- Versioned JSON at `evaluation/gym_calibration_v0.json`
- Policy enforced in `CalibrationEntry.__post_init__`: an entry with
  `evidence_status = "uncalibrated_v0"` **cannot** carry reference_ranges.
  Prevents silent vaporware ranges from shipping.
- v0 ships all 12 exercises as `uncalibrated_v0` - deliberately honest.
- When labeled data arrives, an entry flips to `"cited"` with `evidence_source`
  pointing to the scorecard + eval run that justified the range.

---

## TECHNICAL IMPLEMENTATION

- **Language / runtime:** Python 3.11+, FastAPI, uv package manager
- **Pose backend:** MediaPipe Pose Landmarker (VIDEO mode, heavy variant)
- **Tolerant accessors:** frame/joint readers handle both in-memory
  (enum keys, JointObservation) and serialized (string keys, plain dicts)
  without a conversion step at every call site
- **Test suite:** 187 tests, all passing; no MediaPipe required for fast
  subset (`make test-pose-core` runs in ~9s)
- **Reproducibility:** SHA-256 pinned exercise manifest; deterministic
  segmenter (no randomness); frozen dataclass results throughout
- **End-to-end script:** `scripts/analyze_gym_clip.py` accepts `--video`
  (full MediaPipe) or `--frames-json` (pre-extracted, no GPU needed for demos)

---

## EVALUATION INFRASTRUCTURE

Following ADR 0002 Phase C/D:
- Parity gate: `LAKSH_USE_CANONICAL_JOINTS` flag enables telemetry; Tukey
  fence p90 outlier detection gates the canonical-vs-legacy flip
- Gym manifest: `evaluation/gym_manifest.csv` tracks labeled clips with
  `exercise_id`, `expect_pose_usable`, `expect_min_detection_rate` columns
- Phase D Milestone 2 target: labeled IoU / F1 harness for rep boundary
  detection; labeled reference ranges from `evaluation/gym_manifest_hard.csv`

---

## KEY INSIGHT: HONESTY AS DESIGN PRINCIPLE

Most AI systems hide uncertainty. We surface it explicitly:

```json
"calibration": {
  "evidence_status": "uncalibrated_v0",
  "per_rep": [{
    "rep_index": 0,
    "fields": {
      "tempo_ratio_ecc_over_con": {
        "status": "no_reference_yet",
        "value": 1.73,
        "range": null,
        "evidence_status": "uncalibrated_v0"
      }
    }
  }]
}
```

A judge can see the system knows its tempo ratio is 1.73 but honestly
says "we have no labeled evidence to know if 1.73 is good or bad for this
movement." That is not a weakness - it is the correct scientific posture
for a system at this stage of development.

---

## FUTURE WORK

1. **Milestone 2:** Labeled rep boundary annotations; F1 / IoU eval harness
2. **Reference ranges:** Cited bands from labeled subset, per-exercise
3. **Phase C promotion:** Canonical joint path as default in KinematicAnalyzer
4. **Multi-sport:** `sport_configs.py` registry wired to backend
5. **Real-time:** Streaming pose -> rep detection -> live coaching cue

---

## REPOSITORY

`github.com/parvpatodia/Laksh_AI`  |  Branch: `feat/pose-p1b-p2-p3-infra`

Run the demo: `make analyze-gym-clip FRAMES=evaluation/fixtures/demo_squat_frames.json EXERCISE=back_squat`
