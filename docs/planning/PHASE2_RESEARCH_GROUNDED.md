# Phase 2: Research-Grounded Implementation Plan

> **Purpose**: Upgrade Phase 2 with mathematical rigor, published methods, and citations from leading labs (Google Research, MPI, Netflix, Frontiers, MDPI, PMC). Be honest about what we can implement vs. what requires infrastructure we don't have.

---

## Research Summary

| Topic | Key Sources | What We Adopt |
|-------|-------------|---------------|
| Video quality | Netflix VMAF, perceptual quality research | Weighted factors; log-scale for resolution/FPS |
| Uncertainty | PMC 2022 (knee joint propagation), POCO 2024, MediaPipe visibility | Per-landmark visibility; frame-window std dev |
| Biomechanics validation | Frontiers 2026, PMC (free-throw, 3-point) | Metric definitions aligned with literature |
| Transparency | Google Model Cards (Mitchell et al. 2018) | Structured method cards per metric |
| Pose accuracy | MDPI 2025 (RTMPose basketball) | Note MediaPipe limits; cite RTMPose as future upgrade |

---

## 1. Video Quality Score — Research-Backed Formula

### Literature

- **Netflix VMAF**: Multi-method fusion for perceptual quality; too heavy for our use case.
- **Perceptual quality formula** (sph.mn / research): Resolution and FPS contribute logarithmically; weighted additive factors.
- **Pose-specific**: Resolution, FPS, and visibility drive pose accuracy (OpenPose: large movements >10 cm accurate; small movements weak — PMC 11695451).

### Proposed Formula

For **pose analysis suitability** (not general perceptual quality), we define:

```
Q_resolution = 20 * log10(max(w,h) / 320)   # 320p baseline; cap at 100
Q_fps        = 15 * min(fps/30, 1.5)        # 30 fps reference; 45+ gets bonus
Q_aspect     = 10 if 0.6 ≤ ar ≤ 2.2 else 5  # Penalize extreme portrait/ultrawide
Q_visibility = 30 * mean(visibility[j] for j in shooting_joints)
Q_people     = 10 if people ≤ 1 else max(0, 10 - 5*(people-1))
Q = min(100, Q_resolution + Q_fps + Q_aspect + Q_visibility + Q_people)
```

**Rationale**:
- Resolution: 720p → ~27, 1080p → ~30; aligns with "720p+ recommended."
- FPS: 30 fps = 15 pts; matches literature on motion blur.
- Visibility: MediaPipe provides per-landmark visibility; average over wrist, elbow, shoulder (shooting chain).
- People: Single subject = full 10 pts; multi-person penalized.

**Citation**: OpenPose accuracy vs. movement amplitude — [PMC 11695451](https://pmc.ncbi.nlm.nih.gov/articles/PMC11695451/).

---

## 2. Uncertainty Quantification — Literature-Aligned

### Literature

- **PMC 2022** (Fonseca et al.): Analytical propagation of uncertainty in knee joint angle computation; ~5° input uncertainty in attitude vector matched experimental std dev in kinematic curves.
- **POCO 2024** (MPI): Per-sample variance from pose regressors; confidence for downstream tasks.
- **MediaPipe**: `visibility` per landmark (0–1); we already have this in 2D data.

### What We Can Do Without Retraining

1. **Frame-window std dev**: For knee and elbow, compute angle over dip±3 frames; output `std(angles)` as empirical uncertainty. Simple, interpretable.
2. **Visibility-weighted uncertainty**: When mean visibility of shooting joints < 0.6, inflate reported ± by factor 1.5.
3. **Report format**: `knee_angle: 147°, ± 6° (from frame variance)` — honest, no false precision.

### Formula (Frame-Window)

```python
window = angles[dip_frame-3 : dip_frame+4]  # 7 frames
std_deg = np.nanstd(window)
uncertainty = max(3, min(12, std_deg * 1.2))  # clamp 3–12°
```

**Citation**: Uncertainty propagation in joint angles — [PMC 9397457](https://pmc.ncbi.nlm.nih.gov/articles/PMC9397457/).

### What We Cannot Do (Yet)

- Per-joint uncertainty from a single forward pass (requires POCO-style model change).
- Full analytical propagation through our custom formulas (would need symbolic/autodiff; out of scope for Phase 2).

---

## 3. Method Cards — Model Card Structure

### Literature

- **Model Cards for Model Reporting** (Mitchell et al., Google, 2018): Intended use, factors, metrics, caveats, recommendations.
- Hugging Face / Stanford extensions: Quantitative analyses, ethical considerations.

### Per-Metric Card Template (Abbreviated)

For each of our 8 metrics, document:

| Section | Content |
|--------|---------|
| **Definition** | What is measured, in one sentence |
| **Formula** | Mathematical form (e.g. arccos for angle) |
| **Data source** | MediaPipe landmarks used |
| **Assumptions** | Single camera, 2.5D depth, no ball tracking |
| **Typical error** | From literature or our empirical ± |
| **Limitations** | When it fails (occlusion, side-view, etc.) |
| **Validation status** | Not validated / plausibility-checked / lab-validated |

### Metric-Specific Literature

- **Knee / Elbow**: 3D angle from hip–knee–ankle, shoulder–elbow–wrist. Standard biomechanics.
- **Release velocity**: Wrist proxy; no ball. Literature: true velocity from ball trajectory.
- **Shot arc**: Lever arm (shoulder–wrist) at release; 2D projection compresses. [Frontiers 2026](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2026.1732293/full): proficient 3-point shooters show distinct prep-phase mechanics.
- **Kinetic sync**: Frame count dip→release × 1000/fps. Frame-rate dependent; no literature-specific formula.

---

## 4. Confidence Factors — Transparent Decomposition

### Literature

- **Calibrated confidence**: Not from one paper; standard practice in ML transparency.
- **Attribution**: User can act on each factor (e.g. "re-record with single person").

### Implementation

Return `confidence_factors` as array:

```json
{
  "confidence": 72,
  "confidence_factors": [
    { "factor": "video_quality", "impact": -8, "message": "Quality score 58/100" },
    { "factor": "multi_person", "impact": -15, "message": "2 people detected" },
    { "factor": "pose_visibility", "impact": -5, "message": "Low joint visibility" }
  ]
}
```

Sum of impacts ≈ (100 - confidence). User sees exactly why.

---

## 5. Honest Gaps and Future Work

| Gap | Honest Statement |
|-----|------------------|
| **Ball tracking** | Release velocity and arc from wrist proxy; systematic error. Ball tracking (YOLO) would improve. |
| **Pose model** | MediaPipe Pose; RTMPose validated for basketball (MDPI 2025) with higher accuracy — future upgrade. |
| **Lab validation** | No mocap comparison yet; metrics are plausibility-checked, not ground-truth validated. |
| **3D** | Single-camera 2.5D; true 3D requires multi-view or calibrated cameras. |

---

## 6. Implementation Order (Revised)

| Week | Feature | Research Basis |
|------|---------|----------------|
| 1 | Video quality score | Log-scale resolution/FPS; visibility-weighted; pose-specific |
| 2 | Uncertainty ± (knee, elbow) | Frame-window std dev; visibility inflation |
| 3 | Method cards | Model card structure; per-metric sections |
| 4 | Confidence factors | Transparent attribution; actionable messages |

---

## 7. References

1. Mitchell et al., "Model Cards for Model Reporting," 2018. https://research.google/pubs/model-cards-for-model-reporting/
2. Fonseca et al., "An analytical model to quantify the impact of the propagation of uncertainty in knee joint angle computation," PMC 9397457, 2022.
3. "How accurately can we estimate spontaneous body kinematics from video recordings?" PMC 11695451.
4. Cabarkapa et al., "Biomechanical determinants of proficient 3-point shooters: markerless motion capture analysis," Frontiers in Sports and Active Living, 2026.
5. "Feasibility and Accuracy of an RTMPose-Based Markerless Motion Capture System for Single-Player Tasks in 3x3 Basketball," MDPI Sensors, 2025.
6. Netflix VMAF: https://github.com/Netflix/vmaf
7. MediaPipe Pose Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

---

*Document version: 1.0 | Research-grounded Phase 2*
