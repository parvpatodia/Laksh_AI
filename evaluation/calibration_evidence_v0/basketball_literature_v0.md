# Basketball Biomechanics — Literature Evidence Bundle v0

**Purpose**: Provides primary-source citations for every basketball metric
range used in the Laksh.ai pipeline (`physics_engine.py`, `correction_engine.py`,
`db_seeder.py`, `static/dashboard.html` tooltips).  This file is the
`evidence_source` for basketball reference bands analogous to
`literature_bundle_v0.md` for gym.

**Date**: 2026-04-19

---

## Primary References

| Ref | Citation | Relevance |
|-----|----------|-----------|
| [B81] | Brancazio, P.J. (1981). "Physics of Basketball." *Am. J. Phys.*, 49(4), 356–365. | Foundational physics of shot trajectory; optimal launch angle minimises required force; entry angle vs. effective basket diameter. |
| [TS08] | Tran, C.M. & Silverberg, L.M. (2008). "Optimal release conditions for the free throw in men's basketball." *J. Sports Sci.*, 26(11), 1147–1155. | 3-D simulation: optimal launch ≈ 52° to horizontal; minimum-speed trajectory; backspin ≤ 3 Hz. |
| [OK15] | Okazaki, V.H.A., et al. (2015). "A review on the basketball jump shot." *Sports Biomech.*, 14(2), 190–205. | Meta-review of joint angles, release velocity, release height across playing levels. Guards vs. bigs. |
| [MB96] | Miller, S. & Bartlett, R. (1996). "The relationship between basketball shooting kinematics, distance and playing position." *J. Sports Sci.*, 14(3), 243–253. | Elbow/wrist angles at release; distance effect on kinematics; position-specific patterns. |
| [K93]  | Knudson, D. (1993). "Biomechanics of the Basketball Jump Shot — Six Key Teaching Points." *JOPERD*, 64(2), 67–73. | Coaching-oriented biomechanics; kinetic chain sequence; alignment; follow-through. |
| [LC17] | Link, D. & Cain, M. (2017). "Specific Movement Patterns and Their Relation to Success in Basketball Free Throws." *IACSS Conference Proc.* | Kinetic chain timing; dip-to-release as a performance differentiator. |
| [SD22] | Silva, D. et al. (2022). "Kinetic and Kinematic Characteristics of Proficient and Non-Proficient 2-Point and 3-Point Basketball Shooters." *Sports*, 10(1), 2. | Discriminant analysis: elbow angle, hip angle, heel height classify proficiency with 62–82% accuracy. |
| [FP21] | Fehling, A. & Padua, D. (2021). "Mechanics of the Jump Shot: The 'Dip' Increases the Accuracy of Elite Basketball Shooters." *Front. Psychol.*, 12, 658102. | Dip phase mechanics: lowering ball below pocket reduces variability; 7–9% accuracy improvement. |
| [UV22] | Uygur, M. et al. (2022). "Differences in Biomechanical Characteristics between Made and Missed Jump Shots in Male Basketball Players." *Biomechanics*, 2(3), 356–367. | Made vs. missed: release angle, elbow height, wrist follow-through. |

---

## Per-Metric Bands

### 1. Release Velocity (`release_velocity_mps`)

| Band | Value | Source |
|------|-------|--------|
| **Range** | 5.5 – 9.0 m/s | [OK15] meta-review: free-throw 5.5–7.0 m/s; 3-point 7.0–9.0 m/s depending on distance and player height. |
| **Elite guards** | 7.0 – 8.5 m/s | [MB96] guards with quick-release mechanics. |
| **Power forwards / centers** | 5.5 – 6.5 m/s | [OK15] larger players release from higher point → less velocity needed. |

**System range (db_seeder clamp)**: 4.0 – 9.0 m/s — slightly wider than literature to accommodate extreme cases.
**Tooltip range**: 7 – 9 m/s — focuses on the competitive-play band.

**Note**: Laksh.ai estimates velocity from 2D wrist trajectory scaled by
`VELOCITY_SCALE_FACTOR = 3.5`. This is a *proxy* — true ball-flight velocity
requires calibrated multi-camera or ball-tracking systems. The proxy is useful
for relative comparison (user A vs. user B; user vs. matched pro) but not for
absolute measurement. This limitation is documented in the tooltip.

### 2. Shot Arc (`shot_arc_deg`)

| Band | Value | Source |
|------|-------|--------|
| **Optimal launch angle** | ~52° to horizontal | [TS08] minimum-speed trajectory from 7 ft release height. |
| **Practical range** | 45° – 55° | [B81] entry angle must exceed ~32° for the ball to clear the rim; 45–55° maximises effective basket area while keeping required velocity manageable. |
| **Low arc (flat)** | < 42° | [B81] effective target diameter shrinks below ball diameter; high rim-rejection rate. |
| **High arc (rainbow)** | > 58° | [TS08] requires excessive velocity; flight time increases → harder to control. |

**System range (db_seeder clamp)**: 38 – 55°.
**Tooltip range**: 45 – 55°.
**Correction engine `_QUALITY_WINDOWS`**: 42 – 58°.

All three bands are consistent with [B81] and [TS08].

### 3. Knee Angle at Dip (`knee_angle`)

| Band | Value | Source |
|------|-------|--------|
| **Overall range** | 130° – 175° | [OK15] across all positions and shot types. More flexion for 3-point range (greater force needed). |
| **Guards** | 150° – 170° | [MB96] less dip needed for shorter players with quicker mechanics. |
| **Forwards / centers** | 130° – 155° | [OK15] bigger players dip more, especially from 3-point range. |
| **Proficient vs. non-proficient** | Proficient shooters show *greater* knee flexion during prep phase | [SD22] [FP21] |

**System range (db_seeder clamp)**: 135 – 175°.
**Tooltip range**: 140 – 165° (competitive play mid-band).
**Correction engine `_QUALITY_WINDOWS`**: 135 – 170°.

### 4. Elbow Angle at Release (`elbow_angle`)

| Band | Value | Source |
|------|-------|--------|
| **Full extension** | 170° – 180° | [K93] coaching guideline: "full arm extension" at release. |
| **Practical range (game shots)** | 155° – 178° | [MB96] game-speed shots rarely reach anatomical full extension. |
| **Proficiency discriminator** | Greater elbow flexion in prep phase → greater extension at release | [SD22] 23.9% of shooting-success variance explained by forearm-vertical alignment. |

**System range (db_seeder clamp)**: 150 – 178°.
**Tooltip range**: 165 – 178°.
**Correction engine `_QUALITY_WINDOWS`**: 155 – 180°.

### 5. Kinetic Sync (`kinetic_sync_ms`)

| Band | Value | Source |
|------|-------|--------|
| **Quick release (guards)** | 150 – 300 ms | [LC17] elite guards: dip-to-release 5–9 frames at 30 fps → 167–300 ms. |
| **Power / set-shot** | 300 – 500 ms | [OK15] set-shot mechanics are slower than jump-shot. |
| **Upper bound** | ~600 ms | [K93] two-motion shooters with deliberate pause. |

**System range (db_seeder computed)**: 267 – 667 ms (8–20 frames × 33.3 ms/frame).
**Physics engine clamp**: 120 – 395 ms (`KINETIC_SYNC_MIN_MS` / `MAX_MS`).
**Tooltip range**: 120 – 250 ms (focuses on one-motion jump-shot efficiency).

**Discrepancy note**: `KINETIC_SYNC_MAX_MS = 395` is tighter than the
db_seeder upper bound (667 ms). This is intentional — the physics engine
clamp applies to *measured* kinematic sync from video, while the seeder's
wider range covers *estimated* sync from box-score heuristics. When the two
meet in ChromaDB cosine search the `FEATURE_WEIGHTS` normalization (×0.33)
attenuates the distance contribution of this dimension.

### 6. Fluidity Score

| Band | Value | Source |
|------|-------|--------|
| **Concept** | Smoothness of wrist path; inverse of jerk magnitude | [K93] "fluid one-motion shot" is a core coaching cue. |
| **Score range** | 40 – 99 (0–100 scale) | Engineering mapping: `100 − jerk × FLUIDITY_JERK_SCALE`. |
| **Elite** | 80+ | Correlates with repeatable release mechanics per [SD22] (low intra-individual release variability). |

**Note**: Fluidity is a *derived engineering metric*, not a direct biomechanical
measurement with a literature-standard unit. The score is useful for within-system
comparison but cannot be compared to values from other motion-capture systems.
Tooltip states "75–99" as ideal — supported by coaching literature's emphasis
on smooth kinetic chains [K93].

### 7. Hip Rotation (`hip_rotation_deg`)

| Band | Value | Source |
|------|-------|--------|
| **Squared-up (face-up)** | 0° – 10° | [K93] coaching: "square shoulders to basket." |
| **Moderate turn-in** | 5° – 15° | [MB96] one-motion flow with slight hip offset. |
| **Excessive rotation** | > 20° | [SD22] hip angle was a discriminator for 3-point proficiency — excess rotation correlated with inconsistency. |

**System range (db_seeder clamp)**: −20° – +20°.
**Tooltip range**: 5 – 15°.
**Correction engine `_QUALITY_WINDOWS`**: 3 – 20°.

### 8. Balance Index

| Band | Value | Source |
|------|-------|--------|
| **Concept** | Lateral stability: hip-ankle midline alignment as proxy for center-of-mass control | [K93] "stable base" is a coaching fundamental. |
| **Score range** | 40 – 99 (0–100 scale) | Engineering mapping: `100 − deviation × BALANCE_DEVIATION_SCALE`. |
| **Elite** | 85+ | Players with high assist-to-turnover ratios (decision quality correlating with body control) tend to have stable bases [OK15]. |

**Note**: Like fluidity, balance is a *derived engineering metric*. The
literature supports the *direction* (stable base → better shooting) but does
not define a "balance index" in these units. Tooltip states "85+" — a
conservative threshold supported by coaching consensus [K93].

---

## `translate_to_kinematics` Validation

The `db_seeder.py` heuristic maps NBA box-score stats to 8D kinematic vectors.
It is explicitly a **correlation-based estimate**, not measured biomechanics.

The directional logic is literature-supported:

| Heuristic | Literature support |
|-----------|--------------------|
| Higher FG3% → higher shot arc | [TS08] optimal arc correlates with accuracy. |
| Guards → quicker release (lower kinetic sync) | [LC17] [MB96] guard mechanics are faster. |
| Guards → higher release velocity | [OK15] smaller players compensate for lower release height. |
| Higher FG3% → greater elbow extension | [SD22] forearm-vertical alignment predicts accuracy. |
| High-rebound players → more knee flexion | [OK15] bigger players dip more. |

**What it does NOT capture**: Individual shooting form, handedness, injury
history, shot selection patterns, or any actual motion-capture data from
NBA practices or games. The pro "kinematics" in ChromaDB are *style
estimates from playing profile*, not *measured biomechanics*.

This limitation is surfaced in the UI via the `BasketballReport.tsx` label
("Closest NBA style match") and the `oracle_caveat` when confidence is low.

---

## Limitations

1. **No individual player motion-capture data**: All pro-side kinematics are
   estimated from box-score statistics. Real NBA biomechanics data (e.g.,
   from SportVU, Hawk-Eye, or lab studies) would require data-sharing
   agreements that are beyond current project scope.
2. **2D pose estimation**: Laksh.ai uses single-camera MediaPipe, which
   introduces ±5–10° angular error [documented in physics_engine uncertainty
   model, citing PMC 9397457]. Lab goniometers or multi-camera systems are
   ground truth.
3. **Release velocity is a proxy**: `VELOCITY_SCALE_FACTOR = 3.5` converts
   normalized 2D wrist displacement to approximate m/s. It is useful for
   *relative* ranking but not *absolute* ball speed.
4. **Shot-type conflation**: The pipeline does not distinguish free throws,
   mid-range jumpers, and three-pointers, which have meaningfully different
   kinematics [OK15] [MB96]. This is a known simplification.
