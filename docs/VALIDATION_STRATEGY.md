# Laksh.ai Validation Strategy & Technical Roadmap

> **Purpose**: How to validate that our metrics are correct and true — for investors, users, and technical due diligence. Long-term thinking for platform credibility.

---

## Part 1: How Do You Validate the System Works?

### The Investor Question: "How do you know your numbers are right?"

**Short answer**: Multi-layer validation: ground-truth comparison (where possible), biological plausibility checks, reproducibility, and transparent limitations.

---

### A. Ground-Truth Validation (Gold Standard)

| Method | What It Does | Feasibility | Output |
|--------|--------------|-------------|--------|
| **Motion-capture lab comparison** | Record the same shot in a Vicon/Qualisys lab with reflective markers. Compute true 3D joint angles. Compare our MediaPipe-derived angles to lab ground truth. | Medium (requires lab partnership) | Mean Absolute Error (MAE) per joint, e.g. "Knee angle: ±6° of lab measurement" |
| **Goniometer / inclinometer** | Physical angle sensors on athlete during shot. Compare sensor readings to our computed angles. | Low–medium | Per-joint validation |
| **Calibrated multi-view** | Two+ cameras with known positions. Triangulate 3D from 2D. Compare to our dimensionless estimates. | Medium | 3D accuracy bounds |

**Action**: Partner with a sports science lab (university, NBA G League performance center) for a 1-day validation shoot. 10–20 shots with lab + our pipeline. Publish results: "Knee flexion within ±8° of lab for single-subject, well-lit side view."

---

### B. Biological Plausibility (Always-On Sanity Checks)

| Check | Rationale | Implementation |
|-------|-----------|-----------------|
| **Range bounds** | Human joints have physical limits (knee 0–180°, elbow 0–180°). | Already implicit in physics_engine; add explicit validation layer that flags out-of-range. |
| **Temporal continuity** | Joint angles don't jump 40° between adjacent frames. | Smoothness metric; flag high jerk. |
| **Correlations** | e.g. higher release velocity → longer shot distance; arc vs. distance. | Cross-validate with physics models. |

**Action**: Add a `validation_flags` field to the API response when any check fails. Display a subtle indicator: "Video conditions may affect accuracy."

---

### C. Reproducibility (Determinism)

| Test | How | Expectation |
|------|-----|-------------|
| **Same video, N runs** | Run pipeline 10× on identical video. | Std dev of outputs ≈ 0 (deterministic). |
| **Model versioning** | Hash of `pose_landmarker_heavy.task` in logs/config. | Reproducible across deploys. |
| **Seed control** | Any randomness (if any) must be seeded. | Same input → same output. |

**Action**: Add a `/health` or `/validate` endpoint that runs a known test video and returns expected ranges. CI runs this on every deploy.

---

### D. Expert Correlation (Human-in-the-Loop)

| Approach | How | Value |
|----------|-----|-------|
| **Expert raters** | 3–5 biomechanists/coaches rate same videos (1–10 "shot quality"). Correlate with our metrics. | Validates that our numbers align with expert judgment. |
| **Pro self-check** | If we analyze a known pro's public video, does the system place them near their own vector in our DB? | Validates pro-matching logic. |

**Action**: Build a small "expert benchmark" set (20–50 videos with expert ratings). Report Pearson correlation between our composite score and expert ratings.

Operational clip taxonomy, pass/fail rules, and `scripts/benchmark_pipeline.py` are in [evaluation_set_spec.md](./evaluation_set_spec.md).

---

### E. Transparency & Documentation

| Artifact | Purpose |
|----------|---------|
| **Method cards** | Per-metric: what it measures, formula, assumptions, limitations, expected error. |
| **Limitations doc** | When the system fails: occlusion, multiple people, side-view distortion, low light. |
| **Uncertainty display** | Show ± ranges where estimable (e.g. "147° ± 5°") instead of bare point estimates. |

**Action**: Add `METRIC_METHODOLOGY.md` and link from tooltips. Display confidence intervals when we have them.

---

### F. Third-Party Trust

| Option | Effort | Impact |
|--------|--------|--------|
| **University validation study** | Medium | High — "Validated by [Lab Name]" |
| **White paper / technical report** | Low–medium | High — investors read these |
| **Published benchmark** | Medium | Medium — community reproducibility |
| **Independent audit** | High | Very high — for enterprise/serious investors |

---

## Part 2: Prioritized Roadmap — Plausible Next Steps

### Tier 1: Quick Wins (1–2 weeks)

1. **Metric tooltips** — What/Why/Ideal per metric. Builds trust, educates users. ✅ Implemented
2. **Method cards** — One-page doc per metric; link from tooltips.
3. **Validation flags** — Return `validation_warnings: []` when out-of-range or high jerk.
4. **Regression test** — One known video, expected output ranges, run in CI.

### Tier 2: Validation Infrastructure (2–4 weeks)

5. **Benchmark dataset** — 20–50 labeled shots (coarse labels: "good form" / "needs work"). Report accuracy.
6. **Reproducibility check** — Same-video multi-run; log std dev.
7. **Uncertainty quantification** — Per-frame variance → ±range on final metrics (where feasible).

### Tier 3: Scientific Rigor (1–2 months)

8. **Lab partnership** — One validation shoot with mocap.
9. **Technical white paper** — Methodology, validation experiments, limitations.
10. **Expert benchmark** — Correlation with human raters.

### Tier 4: Platform Evolution

11. **Video quality score** — Single 0–100: lighting, occlusion, framing, single-subject.
12. **Multi-angle fusion** — If user uploads 2+ views, fuse for better 3D.
13. **Longitudinal tracking** — Same athlete over time, improvement trends.

---

## Part 3: Investor-Ready Claims (What You Can Say Today)

**Today (with current state)**:
- "We use MediaPipe Pose (Google's state-of-the-art model) for joint detection."
- "Our physics engine derives 8 biomechanical metrics from 3D joint angles."
- "We apply Savitzky-Golay smoothing and interpolate missing data."
- "When multiple people are detected, we reduce match confidence and advise single-subject recording."

**After Tier 1–2**:
- "Our metrics are reproducible (deterministic pipeline)."
- "We validate outputs against biological plausibility ranges."
- "We maintain a benchmark set and report accuracy on it."

**After Tier 3**:
- "Knee and elbow angles validated within ±X° of laboratory motion capture for [conditions]."
- "Our composite score correlates R=X with expert biomechanics ratings."

---

## Summary

**Validation is not a single test** — it's a layered approach:
1. **Ground truth** (when available)
2. **Plausibility** (always)
3. **Reproducibility** (always)
4. **Expert correlation** (when available)
5. **Transparency** (always)

**For investors**: Lead with transparency. Document methodology, run validation where feasible, and state limitations. That builds more trust than overclaiming.
