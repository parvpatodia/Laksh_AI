# Laksh.ai Phased Execution Plan — Phase 2 & 3

> **Purpose**: Concrete, step-by-step procedures for Phase 2 (Trust & Transparency) and Phase 3 (Accuracy Breakthroughs). Designed for hackathon wins, investor pitches, and sports academy trials. Long-term thinking baked into every step.

---

## Prerequisites: Phase 1 Complete

Before starting Phase 2, ensure Phase 1 is done:

- [ ] `GET /health` returns 200 when ChromaDB is ready
- [ ] CORS configured for `https://lakshai-production.up.railway.app` (no wildcard override)
- [ ] Regression test added (`tests/test_physics_regression.py`)
- [ ] Golden video guide documented (`docs/GOLDEN_VIDEO_GUIDE.md`)
- [ ] Repo clean (no LiteLLM artifacts that cause confusion)

---

## Phase 2: Trust & Transparency (2–4 weeks)

**Goal**: Users and investors can trust the numbers. Every metric has context; limitations are visible; quality is quantified.

### 2.1 Video Quality Score (0–100) — Week 1

**Why**: Single number that answers "How good is this video for analysis?" Enables re-record prompts and confidence gating.

**Procedure**:

1. **Backend** (`physics_engine.py`):
   - Extend `_assess_video_quality()` to return a numeric score.
   - Factors: resolution (weight ~25), FPS (~20), aspect ratio (~15), pose visibility (~30), people count (~10).
   - Formula: `score = max(0, min(100, weighted_sum))`.
   - Add `video_quality_score` to `telemetry`.

2. **API** (`main.py`):
   - Pass `video_quality_score` in response (top-level or inside `telemetry`).
   - Reduce `confidence_score` when quality < 60 (e.g. `confidence *= 0.9`).

3. **Frontend** (`dashboard.html`):
   - Show badge: "Video Quality: 78/100 — Good" (green) or "62/100 — Fair, consider re-recording" (amber).
   - On ingestion page: after analysis, display score next to confidence.

**Acceptance**: Upload a low-res clip; score < 70; confidence reduced.

---

### 2.2 Per-Metric Uncertainty Ranges — Week 2

**Why**: "147° ± 5°" builds more trust than "147°". Investors ask "how confident are you?"

**Procedure**:

1. **Physics engine**:
   - Compute per-frame variance for knee and elbow angles in a small window (dip ± 3 frames).
   - Output: `knee_angle_std`, `elbow_angle_std` or ± ranges.
   - Start with 2–3 metrics; expand later.

2. **API**:
   - Add `knee_angle_uncertainty`, `elbow_angle_uncertainty` (or ± in the stats block).
   - Frontend receives e.g. `{ "knee_angle": 147, "knee_angle_uncertainty": 5 }`.

3. **Frontend**:
   - Display "147° ± 5°" where uncertainty is available.
   - Tooltip: "Uncertainty from pose confidence and frame variance."

**Acceptance**: Metric cards show ± when available; others show point estimate.

---

### 2.3 Method Cards — Week 3

**Why**: One-page docs per metric for investor due diligence and user education.

**Procedure**:

1. Create `docs/METRIC_METHODOLOGY.md` with sections per metric:
   - **Release Velocity**: Formula (wrist proxy), assumption (scale from torso), limitation (no ball).
   - **Shot Arc**: Lever arm vs. polynomial, 2D projection limits.
   - **Knee / Elbow**: 3D angle formula, MediaPipe landmarks, typical error.
   - **Kinetic Sync**: Frame count, FPS dependence.
   - **Balance / Fluidity**: Heuristics, no lab validation yet.

2. Link from tooltips: "See methodology →" opens or scrolls to doc.

3. Keep each section under 150 words; cite sources where applicable.

**Acceptance**: Every metric tooltip has a "Methodology" link; doc is complete.

---

### 2.4 Confidence Explanation — Week 4

**Why**: "Why is my confidence 72%?" — actionable, transparent.

**Procedure**:

1. **Backend**: Return `confidence_factors` array:
   ```json
   {
     "confidence": 72,
     "confidence_factors": [
       { "factor": "video_quality", "impact": -5, "message": "Low resolution" },
       { "factor": "multi_person", "impact": -15, "message": "2 people detected" }
     ]
   }
   ```

2. **Frontend**: On Oracle page, add "Why this score?" expandable section listing factors.

**Acceptance**: User sees breakdown; can act on each factor (e.g. re-record with single person).

---

## Phase 3: Accuracy Breakthroughs (1–3 months)

**Goal**: Best-in-class accuracy for shot biomechanics. Ball tracking, multi-shot, optional RTMPose.

### 3.1 Multi-Shot Detection — Month 1

**Why**: Videos with multiple shots use "first dip/release" → wrong subject or blended metrics. Auto-segment = correct analysis per shot.

**Procedure**:

1. **Physics engine**:
   - Detect multiple release peaks (wrist Y minima) across the full video.
   - Return `segments: [{ start_sec, end_sec, release_frame }]`.
   - User picks segment (or we analyze first by default).

2. **Frontend**:
   - Timeline shows detected segments as markers.
   - "Analyze shot 1 / 2 / 3" buttons.
   - Clip selection UI pre-populates with segment ranges.

3. **API**:
   - Accept `segment_index` or keep `start_sec`/`end_sec`; backend uses segments when provided.

**Acceptance**: 3-shot video → 3 segments; user selects one; analysis correct for that shot.

---

### 3.2 Primary Subject Selection — Month 1 (parallel)

**Why**: When N people in frame, we use `num_poses=1` → first detected. User may want the other person.

**Procedure**:

1. **Physics engine**:
   - When `people_detected_max > 1`, run pose with `num_poses=N` on key frames.
   - Return bboxes or centroids per person.
   - Rank by size + centrality (largest, most centered = likely shooter).

2. **Frontend**:
   - Show "2 people detected. Analyzing: Person 1 (largest). [Switch to Person 2]"
   - "Switch" sends `subject_index` or re-runs with different pose ordering (if MediaPipe allows).

3. **Fallback**: If MediaPipe doesn't support subject selection, document limitation; multi-shot + clip selection still help.

**Acceptance**: Two people in frame; user can switch subject; metrics update.

---

### 3.3 Ball Tracking (YOLO) — Month 2

**Why**: Biggest accuracy gain. Release velocity and shot arc from wrist proxy → approximate. Ball trajectory = ground truth.

**Procedure**:

1. **Research**:
   - YOLOv8/v11 with "sports ball" class or fine-tune on basketball.
   - Alternative: ByteTrack or SORT for trajectory association.

2. **Backend**:
   - New module `ball_detector.py`: runs on frames in release window.
   - Output: ball 2D positions, release frame (ball leaves hand), trajectory.
   - If ball found: override `release_velocity_mps` and `shot_arc_deg` from ball.
   - If not: fall back to wrist proxy (current behavior).

3. **Dependencies**:
   - Add `ultralytics` (YOLO) to requirements; Docker build includes model download.

4. **Integration**:
   - Physics engine calls ball detector when available; merges results.
   - Telemetry: `ball_detected: true/false`, `release_from_ball: true/false`.

**Acceptance**: Ball visible in release window → velocity/arc from ball; otherwise wrist proxy.

---

### 3.4 RTMPose Option — Month 3

**Why**: Higher joint accuracy; validated in basketball studies. Use when GPU available.

**Procedure**:

1. **Abstraction**:
   - Define `PoseExtractor` interface: `extract_frames(video_path, start_sec, end_sec) -> (fps, data_3d, data_2d)`.
   - `MediaPipeExtractor` and `RTMPoseExtractor` implement it.
   - Physics engine uses extractor; rest unchanged.

2. **RTMPose**:
   - Install `mmpose`, `mmcv`, `mmengine` (or `rtmpose` package if simpler).
   - Map RTMPose keypoints to canonical schema (wrist, elbow, etc.).
   - Env var: `POSE_MODEL=mediapipe|rtmpose`; default mediapipe.

3. **Railway**:
   - CPU-only by default; RTMPose needs GPU. Option: Railway GPU add-on or separate GPU service.
   - Document: "RTMPose available when POSE_MODEL=rtmpose and GPU present."

**Acceptance**: With GPU + env, RTMPose runs; output schema identical; metrics comparable.

---

## Library & Model Recommendations (Consolidated)

| Need | Recommendation | Rationale |
|------|-----------------|-----------|
| **Pose** | MediaPipe (now), RTMPose (later) | MediaPipe: fast, CPU, good enough. RTMPose: basketball-validated, GPU. |
| **Ball** | Ultralytics YOLOv8 | Pre-trained "sports ball"; fine-tune on basketball if needed. |
| **Trajectory** | OpenCV + numpy | Custom; lightweight. SORT if multi-ball. |
| **Validation** | pytest + golden video | Determinism; CI blocks regressions. |
| **Uncertainty** | Bootstrap or analytic | Per-frame variance → ±range. |
| **Datasets** | Internal golden set; public basketball videos (CC) | Start with 10–20 clips; expand for benchmarks. |

---

## Long-Term Sequencing

```
Phase 1 (Done)     → Phase 2 (Trust)    → Phase 3 (Accuracy)
Foundations           Transparency         Breakthroughs
Health, CORS,          Quality score,      Multi-shot,
regression test        Uncertainty ±,      Ball tracking,
                       Method cards,       RTMPose option
                       Confidence factors
```

**After Phase 3**:
- Lab validation (mocap comparison)
- Benchmark dataset (50+ shots, reported accuracy)
- Export annotated video
- Longitudinal analytics (same athlete over time)

---

*Document version: 1.0 | Aligned with RELIABILITY_IMPROVEMENT_PLAN*
