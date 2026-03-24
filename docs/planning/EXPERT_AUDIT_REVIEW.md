# Expert Audit: Metric Tooltips & Validation Strategy

**Reviewer lens**: AI platform builder, sports analytics expert, deployment-focused  
**Date**: March 2025  
**Scope**: Metric tooltips, VALIDATION_STRATEGY.md, technical alignment

---

## Executive Summary

| Aspect | Grade | Notes |
|--------|-------|-------|
| **Metric tooltips** | A | Scientifically accurate, consistent, accessible |
| **Physics alignment** | A | All 8 metrics match physics_engine output |
| **Validation strategy** | A | Investor-ready, layered, realistic roadmap |
| **UX & discoverability** | B+ | Hover/click, role="tooltip"; minor mobile edge cases |

**Verdict**: Implementation is production-ready. Recommendations below are refinements, not blockers.

---

## 1. Metric Tooltips — Verification

### Scientific accuracy (vs physics_engine)

| Metric | Tooltip claim | Physics engine implementation | ✓ |
|--------|---------------|-------------------------------|---|
| release_velocity_mps | "Wrist trajectory and limb length scaling" | `power_ratio * 3.5` from wrist travel / torso length | ✓ |
| shot_arc_deg | "Shoulder→wrist at release, or post-release flight path" | Lever arc (s→w) + polynomial fit when available | ✓ |
| knee_angle | "Hip–knee–ankle at dip" | `_calculate_3d_angle(h3d, k3d, a3d)` at dip_frame | ✓ |
| elbow_angle | "Shoulder–elbow–wrist at release" | `_calculate_3d_angle(s3d, e3d, w3d)` at release_frame | ✓ |
| hip_rotation_deg | "Shoulder line vs hip line at dip" | `atan2` yaw of shoulder vs hip vectors | ✓ |
| kinetic_sync_ms | "Dip (lowest wrist) to release (highest wrist)" | Frame delta `dip_frame → release_frame` | ✓ |
| balance_index | "Hip–ankle alignment, centered over base" | `hip_mid_x` vs `ankle_mid_x` deviation | ✓ |
| fluidity_score | "Smoothness of wrist path (low jerk)" | `std(diff(diff(wrist_y)))` → jerk metric | ✓ |

### Consistency fix applied

- **knee_angle**: "Why" previously said "Deeper dip (120–150°)" while ideal was "140–160°" — conflated "deeper" with angle ranges. **Corrected** to "Moderate dip (140–165°) loads lower body for power transfer without excessive crouch." Ideal set to "140–165°" for consistency with biomechanics literature.

### Ideal ranges vs literature

| Metric | Our ideal | Literature / practice |
|--------|-----------|------------------------|
| shot_arc_deg | 45–55° | 45–55° (NBA analytics) ✓ |
| elbow at release | 165–178° | ~165–180° ✓ |
| knee dip | 140–165° | 135–165° ✓ |
| kinetic_sync | 120–250 ms | 100–300 ms ✓ |
| release velocity | 7–9 m/s | NBA guards 7–9 m/s ✓ |

---

## 2. Component & UX Audit

### MetricCardWithTooltip (Biomechanics page)

- **What works**: Hover/click, full content (What/Why/Ideal/Limitation), `role="tooltip"`, `aria-label`, `aria-expanded`, `aria-haspopup`.
- **Hover discoverability**: `.metric-info-btn:hover` changes icon color to cyan.
- **Redundancy removed**: `key={m.key}` was on inner div — unnecessary when parent map provides key; removed.

### InlineMetricTooltip (Oracle COMPARE)

- **What works**: Same content, `role="tooltip"`, icon color change when open.
- **Placement**: `top:'100%'` — tooltip appears below. For lower rows in a long list, may go off-viewport. Acceptable for v1; future: `position:fixed` with `getBoundingClientRect` for viewport-aware placement.

### Potential edge cases

1. **Tooltip overflow on mobile**: Bottom-row cards may push tooltip off-screen. Mitigation: scroll or flip tooltip above when near bottom (later improvement).
2. **IcoInfo `className`**: I() wrapper uses `cl` for className; `className` prop is ignored. If a class is needed, use `cl="metric-info-btn"` on the wrapper, not on IcoInfo. Current setup: button has class, icon uses `currentColor` — correct.

---

## 3. VALIDATION_STRATEGY.md — Assessment

### Strengths

1. **Layered approach**: Ground truth → plausibility → reproducibility → expert correlation → transparency. No single point of failure.
2. **Actionable items**: Each section has clear actions (e.g., lab partnership, regression test).
3. **Investor positioning**: Separates "what you can say today" vs "after Tier 1–2" vs "after Tier 3."
4. **Realistic**: Acknowledges limitations (2D, occlusion, multi-person) instead of overclaiming.

### Minor refinements (optional)

1. **Model precision**: "MediaPipe Pose" is fine; for technical docs, "MediaPipe Pose Landmarker (Heavy)" is more precise.
2. **Benchmarking tools**: Consider citing OpenPose/MediaPipe benchmarks (e.g., COCO validation) for context, even if not basketball-specific.

---

## 4. Technical Stack Notes (2025)

- **MediaPipe Pose Landmarker Heavy** is a solid choice for real-time 3D pose.
- **Savitzky–Golay** smoothing is appropriate for kinematic signals.
- **2D→3D limitations** are clearly stated in tooltips — important for credibility.
- **Future**: Lab validation (Vicon/Qualisys) remains the gold standard for investor-facing claims.

---

## 5. Recommendations (Prioritized)

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | None — ship as-is | — | — |
| P1 | Add `validation_flags` to API when out-of-range | Low | High |
| P1 | Regression test: one reference video + expected ranges in CI | Low | High |
| P2 | Method cards (one-pager per metric) linked from tooltips | Medium | Medium |
| P2 | Viewport-aware tooltip placement for long lists | Low | Low |
| P3 | Uncertainty display (e.g., "147° ± 5°") when estimable | Medium | Medium |

---

## 6. Final Checklist

- [x] All 8 metrics have tooltip content
- [x] Tooltip content matches physics_engine
- [x] Knee angle why/ideal made consistent
- [x] `role="tooltip"` and aria attributes for accessibility
- [x] Hover feedback on info icons
- [x] VALIDATION_STRATEGY.md created and coherent
- [x] No duplicate InlineMetricTooltip components
- [x] Biomechanics and Oracle pages both use tooltips

---

**Conclusion**: The implementation is technically sound, scientifically aligned, and ready for users and investors. Transparency about limitations strengthens trust.
