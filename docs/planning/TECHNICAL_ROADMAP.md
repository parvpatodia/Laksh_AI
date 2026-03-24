# Laksh.ai Technical Roadmap — Accuracy, Reliability & Trust (5-Year Vision)

> **Purpose**: A PhD-level technical roadmap for making Laksh.ai the most accurate, reliable, and trustworthy basketball shot analysis system in the industry. Covers the full pipeline, math/physics foundations, and phased improvements.

---

## 1. Current Architecture & Limitations

### Pipeline Today

```
Video Input → MediaPipe Pose (2.5D landmarks) → Physics Engine → 8 Metrics → UI
```

| Stage | Technology | Limitation |
|-------|------------|------------|
| Pose | MediaPipe Pose Landmarker (BlazePose) | Single-camera 2.5D; depth from monocular heuristics |
| Events | Frame-by-frame thresholding | No temporal model (HMM/RNN); single-shot assumption |
| Metrics | Joint-angle formulas, Savitzky-Golay smoothing | No uncertainty quantification; point estimates only |
| Pro Match | Euclidean distance in 8D | No per-metric weighting; no temporal alignment |

### Root Causes of Inaccuracy

1. **Monocular depth**: True 3D requires stereo or calibrated multi-view. MediaPipe infers depth from scale/pose priors → hip/knee depth is approximate.
2. **Single-shot assumption**: Multi-shot videos use "first event" or heuristics → wrong subject or blended metrics.
3. **No uncertainty**: We output "147°" not "147° ± 6°" → overconfident.
4. **Event detection brittleness**: Threshold-based release/gather can fail on low-quality or atypical motions.

---

## 2. Physics & Math Foundations (What We're Actually Computing)

### Release Velocity

- **Formula**: \( v = \| \mathbf{p}_{ball,t+1} - \mathbf{p}_{ball,t} \| \cdot f \) (pixel displacement × scale factor × FPS).
- **Assumption**: Ball tracking or proxy (e.g., wrist velocity). If no ball: wrist/elbow velocity proxy.
- **Limitation**: Scale from camera distance; no calibration → m/s is approximate.

### Shot Arc (Launch Angle)

- **Formula**: \( \theta = \arctan2(v_y, v_x) \) in release frame.
- **Limitation**: Requires clean release vector; side-view compresses horizontal component.

### Joint Angles (Knee, Elbow)

- **Formula**: \( \theta = \arccos\left( \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|} \right) \) between limb vectors.
- **Limitation**: 2D projection → true 3D angle error; occlusion flattens angles.

### Hip Rotation

- **Formula**: Pelvis orientation relative to shooting direction.
- **Limitation**: Strongly view-dependent; "requires calibrated cameras" is correct.

### Kinetic Sync

- **Formula**: Time between knee peak flexion and release (ms).
- **Limitation**: Depends on accurate release-frame detection.

### Balance Index & Fluidity Score

- **Formula**: Heuristics from CoM variance, jerk, smoothness.
- **Limitation**: No lab-validated thresholds; relative more than absolute.

---

## 3. Phased Improvements (Priority Order)

### Phase 1: Robustness (0–3 months)

| Improvement | Description | Impact |
|-------------|-------------|--------|
| **Multi-shot detection** | Segment video by release events; analyze each shot separately; let user choose which to display. | Correct subject per shot; no blended metrics. |
| **Primary subject selection** | If N people: choose by size, centrality, or user tap. | Avoid "wrong person" analysis. |
| **Validation flags** | Return `validation_warnings` when out-of-range, high jerk, low visibility. | ✅ Implemented. |
| **Video quality score** | Single 0–100: resolution, FPS, aspect, occlusion, lighting. Gate confidence. | User knows when to re-record. |

### Phase 2: Uncertainty & Transparency (3–6 months)

| Improvement | Description | Impact |
|-------------|-------------|--------|
| **Per-metric ± ranges** | Bootstrap or analytic propagation: "147° ± 6°". | Honest reporting; builds trust. |
| **Confidence tiers** | Map quality score → "High / Medium / Low confidence" with clear UI. | User understands reliability. |
| **Method cards** | One-page doc per metric: formula, assumptions, limitations, typical error. | Investor & expert credibility. |
| **Regression suite** | 10–20 golden videos; CI runs pipeline; assert output within bounds. | Prevents regressions. |

### Phase 3: Accuracy (6–12 months)

| Improvement | Description | Impact |
|-------------|-------------|--------|
| **Lab validation** | Mocap partnership; publish MAE per joint (e.g. knee ±8°). | Gold-standard credibility. |
| **Temporal models** | HMM or small RNN for release/gather detection. | More robust event detection. |
| **View-angle correction** | Heuristic correction for pure side-view (e.g. arc, depth). | Better metrics for common setup. |
| **Benchmark dataset** | 50+ labeled shots; report accuracy, precision, recall. | Reproducible evaluation. |

### Phase 4: Multi-View & 3D (12–24 months)

| Improvement | Description | Impact |
|-------------|-------------|--------|
| **Dual-camera fusion** | Two calibrated views → triangulate 3D. | True joint angles; hip rotation reliable. |
| **Camera calibration flow** | User records checkerboard or known object → intrinsics/extrinsics. | Scale and depth correct. |
| **Ball tracking** | Dedicated ball detector + trajectory model. | True release velocity and arc. |

### Phase 5: Platform Evolution (2–5 years)

| Improvement | Description | Impact |
|-------------|-------------|--------|
| **Longitudinal analytics** | Same athlete over time; trends, improvement. | Coaching value. |
| **Comparative norms** | Age/level/gender percentiles. | Contextualized feedback. |
| **Real-time analysis** | Edge deployment; sub-second feedback. | In-practice use. |
| **API for third parties** | SDK for apps, wearables, broadcast. | Ecosystem growth. |

---

## 4. Implementation Checklist (Phase 1)

- [ ] **Multi-shot segmentation**: Detect release peaks; split into N segments; analyze each.
- [ ] **Subject selection**: If multi-person, rank by bbox size + centrality; expose "Analyze this person" in UI.
- [ ] **Quality score**: Combine resolution, FPS, visibility, aspect → 0–100; display in UI.
- [ ] **Regression test**: One golden video in `tests/`; assert key metrics within ±X%.

---

## 5. Investor & User Messaging (Aligned with Roadmap)

**Today**:

- "We use state-of-the-art pose estimation (MediaPipe) and a physics-based kinematic pipeline."
- "We validate outputs against biological plausibility and surface warnings when conditions are suboptimal."
- "Best results: single subject, 45° front-offset, good lighting, 30+ FPS."

**After Phase 1–2**:

- "Multi-shot support and subject selection eliminate wrong-person analysis."
- "We report confidence tiers and uncertainty ranges where available."

**After Phase 3**:

- "Joint angles validated within ±X° of laboratory motion capture for standard recording conditions."

---

## 6. Summary

**Accuracy comes from**:

1. **Correct events** — release, gather, landing.
2. **Correct subject** — single person or user-selected.
3. **Correct math** — joint angles, velocities, timing.
4. **Correct representation** — uncertainty, confidence, limitations.

**Trust comes from**:

1. **Transparency** — method cards, limitations, validation warnings.
2. **Validation** — lab comparison, reproducibility, expert correlation.
3. **User control** — re-record prompts, shot selection, confidence display.

This roadmap aligns the technical pipeline with the validation strategy and positions Laksh.ai for industry-leading accuracy and trust over the next five years.
