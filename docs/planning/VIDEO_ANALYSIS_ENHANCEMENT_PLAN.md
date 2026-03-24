# Video Analysis Enhancement Plan — Overlays, Preprocessing & Beyond MediaPipe

> **Purpose**: A research-grade plan for enriching the user-uploaded video experience with annotated overlays, video preprocessing, and a roadmap beyond MediaPipe. Written from the perspective of an AI expert and video analysis research scientist.

---

## Executive Summary

Today you have:
- **Backend**: MediaPipe Pose (33 landmarks) but only 12 joints extracted (wrist, elbow, shoulder, hip, knee, ankle × L/R)
- **Frontend**: Canvas overlay drawing a simple skeleton (lines + dots) in green/cyan — **no labels**
- **Telemetry**: Per-frame 2D joint coordinates for dip → release + 1.5s post-release
- **Video**: Played as-is (`objectFit: contain`) — no preprocessing or fitting

This plan covers:
1. **Video preprocessing** — fit, crop, letterbox, normalize
2. ** richer overlay system** — labels, release point, shooting arm highlight, optional extra landmarks
3. **MediaPipe vs. alternatives** — what limits you, what to add on top
4. **Phased implementation options**

---

## Part 1: Video Preprocessing ("Fit Perfectly")

### Current State
- Video is displayed with `object-fit: contain` in a 16:9 container
- Raw frames are passed to MediaPipe as-is
- No letterboxing, no aspect-ratio normalization, no crop/zoom

### Options for "Fit Perfectly"

| Option | Description | Use Case | Effort |
|--------|-------------|----------|--------|
| **A. Letterbox / Pad** | Add black bars to force 16:9 or 4:3; keeps full frame | Standardize display; minimal data loss | Low |
| **B. Crop to center** | Crop to desired aspect (e.g. 16:9) from center | Focus on shooter; may cut limbs | Low |
| **C. Auto-crop to person** | Detect bounding box of person; crop + pad to square or 16:9 | Best "fit" for analysis; can lose context | Medium |
| **D. Stabilization** | Reduce camera shake (e.g. OpenCV `videostab`) | Smoother overlays; subtle improvement | Medium |
| **E. Resolution normalization** | Resize to fixed (e.g. 1280×720) before processing | Consistent MediaPipe input; faster on large files | Low |

**Recommendation**: Start with **E + A** — normalize resolution and letterbox for display. This gives:
- Consistent input for pose estimation (MediaPipe works best with 256–512px on the shorter side)
- Predictable overlay coordinates
- No loss of content

**Implementation sketch**:
```python
# Backend: when extracting frames
def preprocess_frame(frame, target_h=720, target_ar=16/9):
    h, w = frame.shape[:2]
    # Scale so height = target_h, then pad width if needed
    scale = target_h / h
    new_w = int(w * scale)
    new_h = target_h
    resized = cv2.resize(frame, (new_w, new_h))
    pad_w = int((target_ar * target_h - new_w) / 2)
    if pad_w > 0:
        resized = cv2.copyMakeBorder(resized, 0, 0, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
    return resized, scale, pad_w
```

For **display only** (no backend change): apply letterbox via CSS so video fills container with correct aspect. Overlay coordinates are in normalized [0,1] so they remain correct.

---

## Part 2: Rich Overlay System — Labels, Release Point, Shooting Arm

### Current Overlay
- Skeleton: 6 points (wrist, elbow, shoulder, hip, knee, ankle) for shooting side only
- Lines connecting them; dots at joints
- Color: green (gather) → cyan (release + follow-through)
- **No text labels, no "Shooting Arm" highlight, no explicit Release Point marker**

### Proposed Overlay Enhancements

#### Tier 1: Essential Labels (Low Effort)
| Element | Implementation | UX |
|---------|----------------|-----|
| **Joint labels** | Text at each joint: "Wrist", "Elbow", "Shoulder", "Hip", "Knee", "Ankle" | Toggle on/off |
| **Left/Right** | Prefix: "L Knee", "R Elbow" | Only if both sides drawn; you currently draw shooting side only |
| **Release point** | Star/bullseye at wrist position in release frame; label "Release" | Always visible when t ≥ release |

#### Tier 2: Shooting-Specific Highlights (Medium Effort)
| Element | Implementation | UX |
|---------|----------------|-----|
| **Shooting arm chain** | Thicker stroke + different color for shoulder→elbow→wrist | Visual emphasis |
| **Release window** | Vertical band or bracket at release time on timeline | Sync with video |
| **Key phases** | Short labels: "Dip", "Drive", "Release", "Follow-through" | Based on frame ranges |

#### Tier 3: Extended Landmarks (Higher Effort)
MediaPipe gives **33 landmarks**. You use 12. You could add:
- **Face**: Nose (0) — orientation cue
- **Hands**: Pinky (17,18), Index (19,20), Thumbs (21,22) — finer release mechanics
- **Feet**: Heels (29,30), Foot index (31,32) — balance, toe-off

**Data change**: Extend `extract_frames()` and telemetry to include these. Overlay would draw them as smaller points with optional labels.

#### Tier 4: Analytical Overlays (Advanced)
| Element | Description | Backend Change |
|---------|-------------|----------------|
| **Joint angle badges** | "147°" at knee, "165°" at elbow at key frames | Already computed; pass to telemetry |
| **Velocity vectors** | Arrow from wrist at release showing direction/magnitude | Need velocity in telemetry |
| **Center of mass** | Dot at hip midpoint; trail over time | Compute from hip landmarks |
| **Balance line** | Line hip-mid to ankle-mid at dip | Visualize balance metric |

---

## Part 3: Is MediaPipe Limiting You?

### MediaPipe Pose Strengths
- **Fast**: Real-time on CPU/mobile
- **Robust**: Works across poses, lighting
- **33 landmarks**: More than COCO (17)
- **3D (world) coordinates**: Relative depth for kinematic chains
- **Well-supported**: Google-maintained, Python/JS/Android/iOS

### MediaPipe Limitations (Relevant to You)

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Monocular depth** | Hip rotation, depth-dependent metrics approximate | Multi-camera fusion; view-angle correction |
| **Single-person bias** | `num_poses=1` in main pipeline; multi-person needs separate pass | You already sample with `num_poses=3` for people count |
| **No ball** | Release velocity from wrist proxy; no true ball trajectory | Add ball detector (YOLO, custom) |
| **33 pts, not 133** | Less fine than whole-body (hands, face) models | MediaPipe Holistic adds hands+face |
| **Black-box depth** | z is heuristic, not metric | Calibration or stereo for true 3D |

### Alternatives & Complements

#### A. Stay on MediaPipe, Add Modules
| Add-on | Purpose | Effort |
|--------|---------|--------|
| **MediaPipe Holistic** | Body + hands + face in one pipeline | Low — same API style |
| **YOLOv8-OBB or similar** | Basketball detection for ball trajectory | Medium |
| **Optical flow (OpenCV)** | Motion vectors for velocity refinement | Low |

#### B. Replace Pose with Higher-Accuracy Models
| Model | Accuracy | Speed | Integration |
|-------|----------|-------|-------------|
| **RTMPose** (OpenMMLab) | Higher mAP than MediaPipe on COCO | Fast (GPU) | Python; different API |
| **ViTPose** | SOTA on benchmarks | Slower | Heavier; research-grade |
| **DWPose** | Whole-body (body+hands+face) | Medium | More keypoints |
| **PoseShot** (research) | Basketball-specific phases | N/A — custom; no off-the-shelf |

**RTMPose** is the most practical upgrade: used in basketball validation studies (e.g. MDPI 2025), open-source, good accuracy. Migration would require:
- New landmark index mapping (RTMPose COCO-133 or similar vs MediaPipe 33)
- Re-deriving joint-angle logic for new topology
- GPU recommended for real-time

#### C. Add Ball Tracking
- **YOLOv8**: Train or use existing basketball detector
- **Track** ball across frames (Kalman filter, SORT)
- **Release point**: Ball position at first frame leaving hand
- **Arc**: Fit parabola to ball trajectory → true shot arc

---

## Part 4: Research Inspiration (2024–2025)

| Source | Insight | Applicable |
|--------|---------|------------|
| **PoseShot** (Scientific Reports) | CNN–BiLSTM–Transformer for free-throw phase recognition (95.76% F1) | Phase labels (dip, drive, release); temporal modeling |
| **RTMPose 3x3 basketball** (MDPI) | CV <5%, SE <3% for displacement/speed with markerless mocap | Validation target; RTMPose as alternative |
| **Basketball-SORT** | Trajectory-based tracking for complex occlusion | Multi-person + ball tracking |
| **ViTPose** | ViT scalability for pose | If you need max accuracy and have GPU |
| **NBAction / BasketTracking** | Open-source ball + player + action pipelines | Reference implementations |

---

## Part 5: Implementation Options (Prioritized)

### Option 1: Quick Wins (1–2 days)
**Scope**: Labels + release point on existing overlay  
- Add text labels to each joint in `drawSkel`
- Add "Release" marker at wrist in release frame range
- Add toggle in UI: "Show labels" (default: on)

**Backend**: None. All data exists in telemetry.

### Option 2: Enhanced Overlay (3–5 days)
**Scope**: Tier 1 + Tier 2  
- Joint labels with L/R where relevant
- Shooting arm highlight (thicker, accent color)
- Phase labels (Dip / Release) at key frames
- Optional: angle badges at knee/elbow in key frames

**Backend**: Extend telemetry with `shooting_side` (already inferred), `phase` per frame.

### Option 3: Video Preprocessing + Option 2 (≈1 week)
**Scope**: Letterbox/normalize video for display and processing  
- Backend: optional resize/letterbox before pose extraction
- Frontend: ensure overlay coordinates map correctly
- Improved consistency across different phone/camera aspect ratios

### Option 4: Beyond MediaPipe (2–4 weeks)
**Scope**: Ball detection or RTMPose  
- Add YOLO ball detector → true release point, trajectory, arc
- Or migrate pose to RTMPose for higher joint accuracy  
- Significant pipeline changes; recommended after Option 2 is solid

### Option 5: Full Research-Grade Stack (1–3 months)
**Scope**:  
- Multi-shot segmentation  
- Ball tracking + trajectory arc  
- RTMPose or equivalent  
- Uncertainty ranges (±°) on metrics  
- Method cards and validation report  

---

## Part 6: Recommended Path

**Phase A (Now)**  
1. Implement **Option 1** — labels + release point. No backend change.
2. Add toggle: "Labels on/off" for cleaner viewing when desired.

**Phase B (Next)**  
3. Implement **Option 2** — shooting arm highlight, phase labels.
4. Add optional **Option 3** — letterbox/normalize for odd aspect ratios.

**Phase C (Later)**  
5. Evaluate **ball detection** for true release velocity and arc.
6. Consider **RTMPose** if you need published validation numbers (e.g. for investors).

---

## Summary Table

| Enhancement | Backend | Frontend | Complexity | Impact |
|-------------|---------|----------|------------|--------|
| Joint labels (Wrist, Knee, etc.) | No | Yes | Low | High (clarity) |
| Release point marker | No | Yes | Low | High |
| Shooting arm highlight | No | Yes | Low | High |
| Phase labels (Dip, Release) | Minor | Yes | Low | Medium |
| Angle badges at key frames | Minor | Yes | Low | Medium |
| Letterbox / normalize video | Yes | Minor | Medium | Medium |
| Extended landmarks (hands, feet) | Yes | Yes | Medium | Medium |
| Ball detection | Yes | Yes | High | Very High |
| RTMPose migration | Yes | Minor | High | High (accuracy) |

---

---

## Implementation Status (Updated)

| Option | Status | Notes |
|--------|--------|------|
| **Option 1** | ✅ Implemented | Joint labels, release point marker, toggle controls |
| **Option 2** | ✅ Implemented | Shooting arm highlight, phase labels (Setup/Drive/Release/Follow-through) |
| **Option 3** | ✅ Implemented | Backend: frame resize to 720p; telemetry: shooting_side; frontend: letterbox via object-fit |

All overlay logic uses **semantic joint names** (wrist, elbow, shoulder, hip, knee, ankle) — fully model-agnostic for future RTMPose/ViTPose migration.
