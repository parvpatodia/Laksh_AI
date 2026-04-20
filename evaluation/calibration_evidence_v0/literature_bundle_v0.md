# Calibration Evidence Bundle v0 (literature)

This file is the `evidence_source` for every `evidence_status: "cited"` entry
in `evaluation/gym_calibration_v0.json` (manifest_version `v0.2.0`, calibration
manifest_version `v0.2.0`).

Why it exists
-------------
GOALS.md calibration policy (verbatim): *"No new silent hardcoded ideal angles
in code; reference ranges come from documented sources ... live in versioned
config committed with the eval run that justified them."*

We do not yet have a labelled cohort (Milestone 2 deliverable). Until that
ships, the **honest** alternative is to cite the published exercise-prescription
literature — the same recommendations strength coaches use to write tempo
programmes — and surface that as the source. This file enumerates exactly
which numbers came from which references, so any reviewer can verify a band
against the cited page.

A separate `evaluation/synthetic_cohort_v0/` artifact ships alongside (per-rep
JSON dumps from a Gaussian generator parameterised by these same literature
numbers) — that is **not** used as `evidence_source`; it is a sanity-check
that the cited bands cover a realistic distribution of well-formed reps.

---

## Primary references

[NSCA-4e]
: Haff, G. G. & Triplett, N. T. (Eds.). (2016). *Essentials of Strength
Training and Conditioning* (4th ed.). National Strength and Conditioning
Association / Human Kinetics. ISBN 978-1492501626.

[ACSM-11e]
: American College of Sports Medicine. (2022). *ACSM's Guidelines for
Exercise Testing and Prescription* (11th ed.). Wolters Kluwer.
ISBN 978-1975150181.

[Schoenfeld-2015]
: Schoenfeld, B. J., Ogborn, D. I., & Krieger, J. W. (2015). Effect of
repetition duration during resistance training on muscle hypertrophy:
a systematic review and meta-analysis. *Sports Medicine*, 45(4), 577–585.
https://doi.org/10.1007/s40279-015-0304-0

[Wilk-1996]
: Wilk, K. E., Voight, M. L., Keirns, M. A., Gambetta, V., Andrews, J. R., &
Dillman, C. J. (1996). Stretch-shortening drills for the upper extremities:
theory and clinical application. *JOSPT*, 17(5), 225–239.

[Cogley-2005]
: Cogley, R. M., Archambault, T. A., Fibeger, J. F., Koverman, M. M.,
Youdas, J. W., & Hollman, J. H. (2005). Comparison of muscle activation
using various hand positions during the push-up exercise. *Journal of
Strength and Conditioning Research*, 19(3), 628–633.

[Escamilla-2001]
: Escamilla, R. F., Lander, J. E., & Garhammer, J. (2001). Biomechanics of
powerlifting and weightlifting exercises. In W. E. Garrett & D. T. Kirkendall
(Eds.), *Exercise and Sport Science* (pp. 585–615). Lippincott Williams &
Wilkins.

---

## Universal tempo bands (every cyclic exercise)

These derive from the NSCA's controlled-tempo prescription and are repeated
verbatim in ACSM's resistance-training chapter:

| Field | Range | Source |
|---|---|---|
| `concentric_duration_s` | `[0.4, 2.5]` | [NSCA-4e] Ch. 17, p. 460 ("1–2 s concentric is standard for hypertrophy/strength; explosive intent allowed below 1 s but movement should still be observable"). Lower bound `0.4 s` admits explosive intent without admitting jitter. |
| `eccentric_duration_s` | `[0.5, 4.0]` | [NSCA-4e] Ch. 17, p. 460 ("2–4 s eccentric for controlled training"). [Schoenfeld-2015] meta-analysis: rep durations between 0.5 s and 8 s produce equivalent hypertrophy, so we relax the lower bound to `0.5 s`. |
| `tempo_ratio_ecc_over_con` | `[0.7, 4.0]` | Algebraic combination of the two above. Lower bound `0.7` admits intentionally fast eccentrics in power work; upper bound `4.0` matches the longest controlled-eccentric prescription in either guideline. |
| `rep_duration_s` | `[1.2, 6.0]` | Sum of the eccentric + concentric bands above (with the explosive lower edge admitted). Plank uses a much wider range (see below). |

**Status semantics in the UI**: a value INSIDE these bands is reported as
`within_reference`. A value OUTSIDE is `outside_reference` — *not* "wrong",
just not in the controlled-tempo strength-training band the coach literature
prescribes. The system does NOT call the user "good" or "bad"; it reports
agreement with a cited prescription.

---

## Per-exercise signal-amplitude bands

`signal_amplitude` units depend on `rep_signal_type`:
* `cyclic_angle` exercises emit it in **degrees** (joint-angle range over the rep).
* `cyclic_vertical` exercises emit it in **normalised image-y** (`[0,1]`
  fraction of frame height the joint travelled).
* `duration` (plank) treats it as an instability proxy (smaller is better).

Bands are deliberately wide because the actual reading depends on (a) the
subject's anthropometry, (b) camera framing, and (c) MediaPipe noise. We pick
bounds that cover the published full-ROM ranges with a generous safety margin.

### cyclic_angle (degrees)

| Exercise | Range (deg) | Source |
|---|---|---|
| `bench_press` | `[60, 140]` | Elbow flexion change for full-ROM bench press: ~80–130° ([Wilk-1996], Table 2; also visible in [Escamilla-2001] frame analyses). |
| `overhead_press` | `[60, 150]` | Standing OHP elbow ROM: ~70–140° from racked-bottom to lockout ([Escamilla-2001] OHP analysis). |
| `barbell_row` | `[30, 130]` | Elbow flexion for bent-over row: ~40–110° depending on grip ([NSCA-4e] Ch. 14 row-mechanics figure). |
| `pull_up` | `[60, 160]` | Dead-hang to chin-over-bar elbow flexion change: ~80–150°. |
| `push_up` | `[50, 140]` | Standard push-up elbow ROM: ~70–130° ([Cogley-2005], standard hand position). |
| `dumbbell_bicep_curl` | `[80, 160]` | Full-ROM curl elbow flexion change: ~110–150° from extension to flexion ([NSCA-4e] Ch. 13, isolation-curl figure). Wide lower bound admits half-reps; upper bound capped below the 180° geometric limit. |
| `romanian_deadlift` | `[30, 110]` | Hip-angle change (shoulder-hip-knee triplet) for RDL: ~40–90° from standing to deepest hinge ([Escamilla-2001] deadlift kinematics). |

### cyclic_vertical (normalised image-y)

| Exercise | Range | Justification |
|---|---|---|
| `back_squat`, `front_squat`, `conventional_deadlift` | `[0.05, 0.45]` | Hip-y travel as fraction of frame height. Tight framing → larger amplitude; loose framing → smaller. The lower bound `0.05` excludes camera jitter; the upper bound `0.45` exceeds even an aggressive close-up where the hip travels nearly half the frame. |
| `walking_lunge` | `[0.04, 0.40]` | Same logic for knee-y. Slightly tighter because the knee typically travels less than the hip during a step. |

### duration (stability proxy)

| Exercise | Range | Justification |
|---|---|---|
| `plank` | `[0.0, 0.10]` | Total midline-y travel during the hold. A stable plank wobbles `< 10%` of frame height; anything more is the body-line proxy reporting instability. Semantics flipped vs. cyclic exercises (see `app/gym/rep_features.py:_amplitude_feature`). |

---

## Per-exercise rep-duration bands (overrides where they differ from universal)

Only listed when the universal `[1.2, 6.0]` band is too narrow for the movement.

| Exercise | rep_duration_s | Justification |
|---|---|---|
| `back_squat`, `front_squat`, `conventional_deadlift`, `romanian_deadlift`, `barbell_row`, `bench_press`, `overhead_press` | `[1.5, 6.0]` | Compound barbell lifts under load take longer than bodyweight cyclic movements; lower bound raised from 1.2 → 1.5 s ([NSCA-4e] Ch. 17 typical-rep table). |
| `pull_up` | `[1.2, 5.5]` | Bodyweight, but eccentric-controlled pull-ups can run long. |
| `walking_lunge` | `[1.5, 6.0]` | One step (eccentric down + concentric up) is the rep unit. |
| `plank` | `[10, 300]` | Isometric hold; range is the typical prescribed-hold band ([ACSM-11e] Ch. 6). |

---

## What is deliberately left as `no_reference_yet`

`primary_joints_min_visibility` and `primary_joints_missing_frac` are tracking-
quality fields, not biomechanics. The relevant thresholds are engineering
constants set in `app/gym/rep_features.RepFeaturesConfig`
(`visibility_degraded_threshold = 0.5`, `missing_frac_degraded_threshold = 0.25`)
and they already drive the per-field `status` (`valid` / `degraded` /
`unknown`). Adding "reference ranges" on top would be tautological — the
field would appear `within_reference` iff it was already `valid`. So the
calibration entries list these in `comparable_fields` but DO NOT supply
`reference_ranges` for them; `apply_calibration` therefore returns
`status: "no_reference_yet"`, which the UI surfaces honestly.

---

## Limitations of literature-cited bands (read this)

1. **No subject-specific calibration.** A 6 ft 4 in lifter and a 5 ft 2 in
   lifter will produce different `signal_amplitude` for the same %ROM. The
   bands here are wide enough to cover both extremes, which makes
   `outside_reference` a strong signal but `within_reference` a weak one.

2. **No camera-framing normalisation.** `cyclic_vertical` amplitudes assume
   the subject's hip travels through a meaningful fraction of the frame. A
   far-away wide shot will read low amplitude even on a perfectly executed
   rep. Until we ship a per-subject normalisation step (Milestone 2), this
   is a known false-degraded source — flagged in `docs/KARPATHY_AUDIT.md`
   under "known gaps".

3. **Tempo bands assume hypertrophy/strength training intent.** Power-block
   programming intentionally drops below the lower-bound concentric duration.
   A user training for power will trip `outside_reference` on `tempo_ratio`
   even with perfect technique. The UI surfaces the cited band so the user
   can interpret the disagreement, rather than treating it as a verdict.

4. **No medical claims.** These ranges describe what published strength-
   training prescriptions consider "controlled tempo and full ROM"; they are
   not exercise prescriptions. The Laksh.ai system is a measurement tool, not
   a coach. Any reference status is a comparison to a cited norm, not a
   recommendation.

---

## Replacement plan (Milestone 2)

When the labelled subset lands, each exercise's `evidence_source` in
`gym_calibration_v0.json` will switch from this file's path to the
scorecard run hash that produced the empirical band. The literature
citations here will remain as the *prior* the empirical bands had to
beat to be adopted (one-sided welch t-test on the means; bands only
replace literature if they cover ≥ 80% of the labelled subset's reps).
