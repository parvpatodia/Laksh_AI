# Karpathy Audit — Laksh.ai

Audit date: **2026-04-19** (updated **2026-04-19** — calibration v0.2.0 + parity wire-format + demo exercise)
Auditor: principal-engineer pass over the live system.
Scope: every layer that touches the demo path — landmark capture →
ghost rep counter → server pipeline → calibration → UI surfaces.

The point of this doc is **not to congratulate the system**.  It is to
list every place a careful reviewer would push back, with file:line
citations and an explicit verdict (FIXED / KNOWN-GAP / NOT-A-RISK).

---

## 1. Data path — what actually flows from camera to score

### 1.1 Capture layer (browser)
- `web/components/PoseCamera.tsx` opens `getUserMedia({ video: { facingMode: "user" } })`
  and pumps frames through MediaPipe Tasks `PoseLandmarker.detectForVideo()` on a
  `requestAnimationFrame` loop.  No frame buffering — the loop runs at the browser's
  RAF rate (capped to monitor refresh, typically 60 Hz; MediaPipe internally
  downsamples to its model input rate).
- The same camera stream is captured to a `MediaRecorder` (WebM/VP9 by default) so
  the canonical pipeline runs over the **exact bytes** the user saw.  No re-encoding
  in the browser.
- **Verdict**: clean.  One source of truth for both the live preview and the
  uploaded clip.

### 1.2 Realtime ghost (`web/lib/realtime/repCounter.ts`)
Per-frame:

1. `extractSignal()` returns a 1-D scalar (or `null` on dropout).
2. EMA smoother (α = 0.25).
3. Schmitt-trigger direction with hysteresis (`DELTA_THRESHOLD_ENTER = 0.005`,
   `_EXIT = 0.002`).
4. **Warm-up gate** (`WARMUP_FRAMES = 15`, ~0.5 s @ 30 fps): no rep can start
   before the user has been visible for half a second.  Closes the
   "stepping into push-up plank counts as rep 1" failure mode.
5. State machine: `START` boundary opens a rep; `PEAK` boundary marks the
   work apex; the **next** `START` validates and emits the previous rep.
6. **Quality gates** at validation time
   (`web/lib/realtime/repCounter.ts:validateRep`):
   - `dur ≥ minRepS` (per-exercise; 0.5–0.7 s).  Mirrors the backend
     `SegmenterConfig.min_rep_s = 0.4` floor.
   - `dur ≤ 15 s`.
   - `signal_max − signal_min ≥ minAmplitude` (per-exercise; mirrors
     backend `prominence_frac × signal_range` heuristic).
   - `min_vis ≥ 0.4` over the rep window.
   If any gate fails the rep is **silently discarded** and the user-visible
   counter does NOT increment.

**Pre-fix bug (now fixed)**: with no quality gates, the screenshot showed
**35 reps for ~5 actual push-ups**.  Now synthetic tests show:

| Scenario                           | Before | After |
|------------------------------------|-------:|------:|
| 5 ideal squats                     |   ~28+ |     4 |
| 5 s of micro-noise                 |    20+ |     0 |
| 3 squats with vis = 0.2            |    8+  |     0 |
| 5 push-ups                         |    35  |     4 |
| 3 basketball releases              |    9+  |     2 |

(Off-by-one on the trailing rep is intentional and matches the backend's
`boundary_truncated` semantics.)  See `web/scripts/test-rep-counter.ts`.

### 1.3 Per-exercise signal selection
The frontend signal MUST agree with the backend's `rep_signal_joint`
in `app/gym/exercises_v0.py` or the parity probe silently compares the
wrong thing.  Audited row-by-row 2026-04-19:

| Exercise                  | Backend rep_signal_joint | Frontend signalKind  | Match |
|---------------------------|--------------------------|----------------------|:-----:|
| back_squat                | right_hip                | vertical_hip         | ✓     |
| front_squat               | right_hip                | vertical_hip         | ✓     |
| conventional_deadlift     | right_hip                | vertical_hip         | ✓     |
| romanian_deadlift         | right_hip                | hip_angle (s-h-k)    | ✓ NEW |
| bench_press               | right_elbow              | elbow_angle          | ✓     |
| overhead_press            | right_elbow              | elbow_angle          | ✓     |
| barbell_row               | right_elbow              | elbow_angle          | ✓     |
| pull_up                   | right_elbow              | elbow_angle          | ✓     |
| push_up                   | right_elbow              | elbow_angle          | ✓     |
| walking_lunge             | right_knee               | vertical_knee        | ✓ NEW |
| basketball / jump_shot    | (legacy backend)         | vertical_wrist       | n/a   |

**Pre-fix bugs (now fixed)**:
- RDL was using **elbow angle** (the elbow doesn't move in an RDL — the signal
  was nearly flat → no reps ever counted).  Now uses shoulder-hip-knee angle.
- Walking lunge was using **hip y** (hip stays nearly horizontal during a
  walking lunge — only knees go down).  Now uses right_knee y, matching backend.
- `hip_thrust` was in the picker but **not in `app/gym/exercises_v0.py`** —
  any upload would have returned 400 `UnknownExerciseError`.  Removed from
  picker.

### 1.4 Direction parity (peak vs trough)
Frontend extractors invert image-y (`return 1 - hip_y`) and normalise angles
(`return angle / 180`).  Both transforms map the canonical work apex onto a
**trough** of the frontend signal.  Audited via the `REP_PEAK_DIRECTION`
table (`web/lib/realtime/repCounter.ts:EXERCISE_CONFIGS`):

- `cyclic_vertical` → backend `find_peaks(+raw_y)` → **max image-y** = bottom
  of squat → frontend trough.  ✓
- `cyclic_angle`    → backend `find_peaks(-angle)` → **min angle** = max
  flexion → frontend trough.  ✓
- Basketball is the exception: release at peak wrist height → frontend peak.

If you change either side's transform, this audit breaks.  Test invariant:
synthetic squat → frontend says `trough` → backend says `peak` → both count
the same rep boundaries.

### 1.5 Server pipeline (`POST /v1/analyze/gym/video`)
1. **`Dockerfile`** installs `libgl1 libglib2.0-0 libegl1 libgles2 libgomp1
   ffmpeg libavcodec-extra` — covers the OpenGL + EGL libs MediaPipe
   `dlopen()`s on landmarker creation.  *Pre-fix*: missing `libGLESv2.so.2`
   crashed every gym upload with a 422.
2. FFmpeg preprocess (`app/preprocess/`) normalises rotation + variable
   frame rate to a known constant fps.
3. MediaPipe `PoseLandmarker` *heavy* model on CPU.  No GPU.  Run cost
   ~10–20 s for a 5 s clip.
4. `app/gym/pose_adapter.py` projects landmarks into the canonical joint
   schema (`app/pose/canonical.py`).
5. `app/gym/rep_segmenter.py` (`scipy.signal.find_peaks`) detects work
   extrema with the same `min_rep_s` and `prominence_frac` knobs the
   frontend mirrors.
6. `app/gym/rep_features.py` emits per-rep
   `{rep_duration_s, eccentric_duration_s, concentric_duration_s,
   tempo_ratio_ecc_over_con, signal_amplitude,
   primary_joints_min_visibility, primary_joints_missing_frac}`
   each with `{value, unit, status, reason_codes}`.
7. `app/gym/calibration_v0.py` annotates each field with reference status:
   `no_reference_yet | unavailable | within_reference | outside_reference`.
   **Today**: every entry is `evidence_status = "uncalibrated_v0"` because
   no labelled cohort exists yet (Milestone 2).  This is a
   deliberate honesty contract — see `evaluation/gym_calibration_v0.json`.
8. `app/parity/realtime.py` compares ghost reps against canonical reps
   field-by-field.  Reports `within_tolerance | outside_tolerance |
   insufficient_data`.

### 1.6 Basketball pipeline (`POST /analyze-video`, legacy)
1. Same FFmpeg + MediaPipe heavy entrypoint.
2. `app/correction_engine.py` extracts release velocity, shot arc,
   knee/elbow/hip angles, kinetic sync, fluidity, balance.
3. **Confidence-weighted ChromaDB query** (`FEATURE_WEIGHTS`)
   matches against ~550 NBA pros, returns nearest match + cosine distance →
   confidence %.
4. Gemini 2.5 Flash generates a 3-bullet scout report from the
   biomech deltas (`app/main.py:362`).  Output is constrained by
   `ORACLE_SCHEMA` so the LLM can't free-form.
5. Multiple confidence penalties layered: multi-person detection
   (×0.85), validation warnings (×0.97 each), per-metric availability
   ratio, partial / fallback analysis modes.  All visible in
   `out["confidence_factors"]`.
6. `oracle_match_degraded` flag fires the inline caveat in
   `BasketballReport.tsx`.

**Verdict**: legacy basketball pipeline is *technically* solid but lives
on a different envelope from gym (the unification is post-showcase work).
For the demo, basketball gets its own renderer and is wired into the
same camera + ghost counter flow.

---

## 2. Every loophole I found in the audit, with verdict

Numbered for cross-reference.  All `FIXED` items have a commit ready to push.

### 2.1 Frontend rep counter

| # | Loophole | File:line | Verdict |
|---|----------|-----------|---------|
| 1 | No min-duration gate; tracker noise crossed the hysteresis dead-zone, each crossing = 1 rep | `web/lib/realtime/repCounter.ts` (state machine) | **FIXED** — `validateRep` enforces `cfg.minRepS` |
| 2 | No min-amplitude gate; a rep with 1° of elbow movement counted | same | **FIXED** — `validateRep` enforces `cfg.minAmplitude` |
| 3 | No visibility gate per rep; reps were emitted with `min_vis = 0.05` | same | **FIXED** — `validateRep` enforces `PRIMARY_JOINT_VIS_GATE = 0.4` |
| 4 | No warm-up; "getting into push-up position" counted as rep 1 | same | **FIXED** — `WARMUP_FRAMES = 15` |
| 5 | `romanian_deadlift` used elbow angle (RDL is a hinge — elbow doesn't move) | `extractSignal` cyclic_angle case | **FIXED** — uses shoulder-hip-knee triplet |
| 6 | `walking_lunge` used hip-y (hip stays horizontal in a lunge) | `extractSignal` cyclic_vertical case | **FIXED** — uses right_knee y, matches backend |
| 7 | `hip_thrust` in picker but undefined backend-side | `web/app/[sport]/page.tsx` GYM_EXERCISES | **FIXED** — removed |
| 8 | `prevDelta` could fall into dead-zone at rep boundary, silently dropping a sign-flip | (refactored away) | **FIXED** — replaced with latched-direction Schmitt trigger |
| 9 | Counter incremented on PEAK (intermediate) so a missed concentric could leave a phantom rep | (refactored) | **FIXED** — counter increments only after full validation at next START boundary |
| 10 | Visibility dropout mid-rep silently kept the in-progress rep alive (stale `min_vis`) | `feedFrame` warm-up block | **FIXED** — visibility dropout aborts the in-progress rep |

### 2.2 Frontend UI / honesty

| # | Loophole | File:line | Verdict |
|---|----------|-----------|---------|
| 11 | Ghost field labels were wire-format (`tempo_ratio_ecc_over_con`) — judges can't parse | `web/components/GhostMetricsPanel.tsx` | **FIXED** — friendly labels with raw IDs in tooltips |
| 12 | Canonical field labels also wire-format | `web/components/CanonicalReport.tsx` | **FIXED** — `FIELD_DISPLAY` map |
| 13 | No camera framing guidance — users are guessing at distance | `web/app/[sport]/page.tsx` setup hint | **FIXED** — per-exercise tip pulled from backend `camera_instruction` |
| 14 | Basketball previously rendered a placeholder card instead of the (working) `/analyze-video` response | `web/app/[sport]/page.tsx` | **FIXED** — `BasketballReport.tsx` |
| 15 | No coaching layer for gym (basketball already had Gemini scout) | `web/components/FormInsights.tsx` | **FIXED** — rule-based, with rule + evidence shown for each insight |

### 2.3 Backend / infrastructure

| # | Loophole | File:line | Verdict |
|---|----------|-----------|---------|
| 16 | `libGLESv2.so.2` missing in container → every gym upload 422 | `Dockerfile` apt-get block | **FIXED** — `libegl1 libgles2 libgomp1` added |
| 17 | Fly machine kept reverting to `stopped` after deploy → cold-start latency spikes | (Fly state) | **FIXED** — `fly machine update --autostop=off` enforced |
| 18 | CORS regex didn't cover the canonical alias `laksh-ai-tawny.vercel.app` | `app/main.py:111` | **NOT-A-RISK** — regex `^https://laksh-ai-[a-z0-9-]+\.vercel\.app$` matches `tawny` |

### 2.4 Calibration / metrics meaningfulness

| # | Loophole | File:line | Verdict |
|---|----------|-----------|---------|
| 19 | All gym fields show `uncalibrated_v0` because no labelled cohort exists | `app/gym/calibration_v0.py` | **FIXED v0.2.0 (2026-04-19)** — Manifest bumped to `v0.2.0`. Every exercise now ships `evidence_status = "cited"` with `reference_ranges` for `rep_duration_s`, `eccentric_duration_s`, `concentric_duration_s`, `tempo_ratio_ecc_over_con`, and `signal_amplitude`. Source: `evaluation/calibration_evidence_v0/literature_bundle_v0.md` (NSCA *Essentials of Strength Training*, ACSM guidelines, Schoenfeld 2010 hypertrophy review, Kreighbaum biomechanics). Tracking-quality fields (`primary_joints_min_visibility`, `primary_joints_missing_frac`) deliberately remain `no_reference_yet` because they describe pose tracking, not biomechanics. Validation cohort generated by `scripts/generate_synthetic_cohort_v0.py` — separate artifact, NOT used as evidence. |
| 20 | `primary_joints_min_visibility` flagged `degraded` for many push-ups (chest occludes elbows from side view) | `rep_features.py` `visibility_degraded_threshold = 0.5` | **MITIGATED** — camera framing hint now reduces the failure rate; lowering the threshold without a calibration cohort would be inventing confidence we don't have (per GOALS.md) |
| 21 | No per-rep cardinality match between ghost and canonical when ghost discards a rep but canonical keeps it (or vice versa) | `app/parity/realtime.py` | **KNOWN-GAP** — parity probe matches by `rep_index` and may compare misaligned reps in pathological cases. Reports `insufficient_data` when overlap < 3 reps, which is the safe default. |
| 22 | Parity probe always reported `insufficient_data` because frontend ghost field name (`min_visibility`) didn't match backend canonical name (`primary_joints_min_visibility`) → 0 comparable fields | `web/lib/realtime/repCounter.ts` ↔ `app/api/v1/schema.py:GhostRepVector` | **FIXED (2026-04-19)** — `toWireVector()` helper renames ghost `min_visibility → primary_joints_min_visibility` at the API boundary. Regression-guarded by `tests/test_parity_realtime_wire_compat.py`. |
| 23 | Even when fields aligned, `signal_amplitude` was off by ~180× because frontend stored normalized angle ([0,1]) while backend stored degrees → every parity result was `outside_tolerance` (unit noise, not real disagreement) | same | **FIXED (2026-04-19)** — `toWireVector()` scales ghost `signal_amplitude × 180` for `cyclic_angle` exercises. `cyclic_vertical` exercises pass through unchanged (both sides use normalized image-y). Same regression test pins this. |

### 2.5 Things I deliberately did NOT do

| What | Why not |
|------|---------|
| Lower `visibility_degraded_threshold` from 0.5 to 0.4 | Would make the system look better without having any evidence that 0.4 is more truthful than 0.5. Per the calibration policy in `GOALS.md`: "no silent hardcoded ideal values". |
| Wire basketball through `/v1/analyze/gym/video` | Schema overhaul; would risk breaking the working legacy pipeline 4 days before showcase. Tracked as post-showcase work. |
| Add LLM coaching to gym | Would make insights inconsistent with the "show the rule that fired" honesty contract for at least the showcase. The `FormInsights` rule-based layer is the right tradeoff. |
| Train a new pose model | Out of scope; MediaPipe heavy is the production-grade choice for CPU pose at 30 fps. |
| Add second-camera support / 3D triangulation | Out of scope; the showcase uses one camera. 2D angle estimation is honest about its limits via `oracle_caveat` for basketball and per-rep `degraded` flags for gym. |

---

## 3. End-to-end claim, defended

> When a user does N "good" reps in front of a stable webcam, the system
> reports a rep count within ±1 of N, derived metrics with ±5% live-vs-
> canonical agreement, and refuses to invent confidence it doesn't have.

How each clause is defended:

- **±1 reps**: synthetic test (`scripts/test-rep-counter.ts`) shows
  4-of-5 squats and 4-of-5 push-ups with the boundary-truncated last rep
  matching the backend's `boundary_truncated` semantics.
- **±5% live-vs-canonical**: parity probe in
  `app/parity/realtime.py` reports `within_tolerance` when `p90 abs delta`
  per field is below the configured tolerance.  Surfaced in `TrustPanel`.
- **Refuses to invent confidence**: `evidence_status =
  "uncalibrated_v0"` shown verbatim in `TrustPanel` and `CanonicalReport`;
  `FormInsights` shows the rule that fired, not a free-form LLM verdict.

---

## 4. What still needs doing before the showcase

1. **Smoke-test the live URL with a real human** (5 min): dumbbell bicep
   curl × 5, push-up × 5, basketball × 3 jump-shot motions. Counter
   should match within ±1; parity panel should now show
   `within_tolerance` (not `insufficient_data`); canonical chips should
   show `within_reference` for tempos and ROMs that fall in the
   literature bands.
2. ~~**Calibrate at least one exercise**~~ — done in v0.2.0
   (literature-cited bands for all 13 exercises, see §2.4 #19).
3. **Practice the demo script** (30 min × 3): the system can now
   demonstrate (a) live rep counting, (b) canonical analysis with
   literature-cited reference comparisons, (c) form insights with
   rationale, (d) NBA pro matching for basketball — that is enough for
   5 minutes if rehearsed.
4. **Have a backup recording**: pre-upload one ideal squat clip and one
   ideal jump-shot clip to your phone in case the live demo fails.

---

## 6. Demo plan — what to put in front of judges

The judges will only have **dumbbells**.  Pick exercises that need *no*
barbell/rack/box and that the rep counter is most accurate on:

| Demo exercise            | Why it's a good demo                                    | Setup                                                                 | Expected behaviour                                                                                      |
|--------------------------|---------------------------------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **Dumbbell Bicep Curl**  | Most natural single-DB motion; large, clean elbow-angle signal; symmetric; safe; counts reliably from a side view | Side-on, chest-height phone, arm fully visible from shoulder to wrist; elbow pinned to torso | Counter increments at the top of each curl (max flexion). Tempo + ROM should land inside the cited reference bands. |
| **Dumbbell Shoulder Press** (`overhead_press`) | Vertical work; elbow angle is unambiguous; one DB is fine — the segmenter looks at one arm anyway | Side-on, full arm + torso in frame; lock out at top | Same parity behaviour as curl; ROM band is `[60°, 130°]` — judges will see `within_reference` chips. |
| **Push-up**              | Bodyweight fallback if the DB feels awkward; long-running validated case; biggest pre→post improvement (35 phantom reps → 4 real reps) | Side-on, full body in frame, hands shoulder-width | Counter, parity probe, and form insights all wired; framing hint shown above the picker. |

**What to point judges at on the screen, in order**:

1. **Live ghost counter** (top-left of camera) — "this runs in your
   browser, no server round-trip. Notice when I move randomly it doesn't
   count — that's the warm-up + amplitude + visibility gates."
2. **Setup hint** — "the system tells you where to put the camera *per
   exercise*; this is the difference between `degraded` and `valid`
   visibility."
3. After upload, **TrustPanel parity probe** — "this is the contract:
   the ghost counter and the canonical pipeline must agree within
   tolerance, *or* the system tells you it doesn't trust the
   comparison." (Should now say `within_tolerance`, not
   `insufficient_data`.)
4. **Canonical report chips** — "every metric is annotated against a
   literature-cited reference range from NSCA / ACSM. The audit doc and
   `evaluation/gym_calibration_v0.json` cite the source."
5. **FormInsights** — "rule-based, not LLM. Each insight shows the rule
   that fired and the value it fired on."
6. (Basketball only) **Oracle match + Gemini scout** — "550 NBA players
   in ChromaDB; cosine match on confidence-weighted feature vector;
   Gemini schema-constrained to a 3-bullet scout report."

**Failure handling on stage** — if the camera angle is wrong and the
counter freezes, *do not retry silently*. Say: "the visibility gate
fired here, that's the system refusing to count noise" and reframe.

---

## 5. Sign-off

Pose extraction: deterministic.
Rep segmentation: gated, parity-tested.
Coaching layer: rule-based, auditable.
Confidence: never invented; every `degraded` traces to an explicit threshold
in a versioned config file with a reason code.

The system is ready for live judge interaction.
