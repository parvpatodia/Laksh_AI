# Video Analysis Limitations & Failure Modes

> **Purpose**: Document scenarios where the Laksh.ai pipeline may produce inaccurate or fallback results. Use this for user guidance, investor due diligence, and technical iteration.

---

## 1. Video Input Conditions

| Condition | Impact | Mitigation |
|-----------|--------|------------|
| **Resolution < 320×240** | MediaPipe pose accuracy degrades; joint noise increases | Use 720p or higher; we surface a validation warning |
| **Framerate < 20 fps** | Fast motions blur; kinetic sync timing less reliable | 30 fps+ recommended; low-FPS warning shown |
| **Slow-motion (e.g. 120 fps)** | Kinetic sync uses dynamic FPS scaling; generally OK | Note shown when high-FPS detected |
| **Vertical/portrait (9:16)** | Shot arc can be compressed; 2D lever angle distorted | For best accuracy, use landscape at 45° angle |
| **Ultra-wide (21:9)** | Joint positions at frame edges may distort | Warning for aspect ratio > 2.2 |
| **Very short clip (< 1 sec)** | May lack clear dip/release; event detection uncertain | Ensure single, complete jump shot |
| **Rotated/upside-down** | Y-axis assumptions may invert; dip/release wrong | Normal orientation expected |
| **Heavy compression** | Block artifacts can confuse pose model | Higher bitrate improves robustness |

---

## 2. Pose Estimation Failure Modes (MediaPipe)

| Scenario | Effect | User Guidance |
|----------|--------|---------------|
| **No person detected** | Empty landmarks → interpolated NaNs; fallback if severe | Ensure shooter is clearly visible |
| **Occlusion** (arm behind body, ball in front) | Wrist/elbow NaN or wrong; arc/velocity affected | Record from angle where shooting arm is visible |
| **Multiple people** | num_poses=1 picks first detected; may analyze wrong person | Record single shooter only; we reduce confidence when multiple detected |
| **Partial visibility** (player at edge) | Many joints NaN; metrics degrade or fallback | Frame full body in shot |
| **Low light** | Pose confidence drops; more NaNs | Good lighting improves accuracy |
| **Fast motion blur** | Jittery landmarks; fluidity score affected | 30+ fps reduces blur |
| **90° side profile** | 3D depth (hip rotation, knee) compressed; balance 2D-only | 45° front-offset unlocks better 3D |

---

## 3. Event Detection (Dip / Release)

| Scenario | Effect | Notes |
|----------|--------|-------|
| **Multiple shots in video** | First dip/release used; later shots ignored | Trim to single shot for best match |
| **Set shot (minimal dip)** | Dip = peak wrist Y; may misidentify if motion noisy | One-motion shots work best |
| **Very slow release** | 1.5s search window should cover | Extended pauses can confuse |
| **Camera tilted** | Y-axis no longer vertical; dip/release logic assumes upright camera | Upright framing recommended |

---

## 4. Metric-Specific Limitations

| Metric | Limitation |
|--------|------------|
| **Release velocity** | Derived from 2D wrist travel; true 3D would need calibrated cameras |
| **Shot arc** | Lever angle from shoulder→wrist; side-view compresses apparent arc |
| **Knee/elbow angle** | MediaPipe 3D; occlusion or angle adds ±5–10° error; lab goniometers are ground truth |
| **Hip rotation** | Best from 45° view; pure profile view compresses depth |
| **Kinetic sync** | Frame-rate dependent; 30 fps ≈ ±33 ms resolution |
| **Balance** | 2D hip–ankle alignment; true CoM requires force plates |
| **Fluidity** | Sensitive to pose noise; Savitzky–Golay mitigates |

---

## 5. Fallback Triggers

Fallback values are used when:

- MediaPipe fails to initialize
- Fewer than 5 frames with pose data
- Uncaught exception in pipeline

Fallback returns static defaults; user should re-record with improved conditions.

---

## 6. Validation Warnings (Surfaced to User)

The pipeline computes and returns `validation_warnings` when:

- Low resolution, low FPS, or non-ideal aspect ratio
- Low pose visibility (< 50%)
- Knee or elbow angles outside biological range
- Multiple people detected
- Fallback used

These appear in the UI to set appropriate expectations.

---

## 7. Recommendations for Best Results

1. **Single shooter** in frame, full body visible
2. **Landscape** at 45° front-offset (not pure profile)
3. **720p or higher**, 30 fps
4. **Good lighting**, minimal occlusion
5. **One complete shot** per clip; trim extra motion

---

*For investor validation and methodology, see [VALIDATION_STRATEGY.md](./VALIDATION_STRATEGY.md).*
