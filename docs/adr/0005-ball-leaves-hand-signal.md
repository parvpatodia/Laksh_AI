# ADR 0005: Ball-Leaves-Hand S3 Signal (Track B)

**Status**: Implemented, feature-flagged (`LAKSH_ENABLE_BALL_DETECT=1`).

**Date**: 2026-04-23

---

## Context

The basketball shot-release detection pipeline uses two independent signals:

- **S1** — Wrist vertical (y) trajectory nadir (Savitzky-Golay smoothed, `scipy.signal.find_peaks`).
- **S2** — Elbow angular velocity (extension spike) within ±4 frames of each S1 candidate.

S1 and S2 together form the consensus release gate (plan ADR 0001 / plan section A1). A
third, physically independent signal was proposed to raise consensus confidence from ~99%
(two 90%-accurate independent signals) toward ~99.9% (three independent signals).

The constraint: the third signal must not introduce a new runtime dependency with
significant CPU cost or require labelled training data we do not have.

---

## Decision

Use **YOLOv8n ONNX** inference on the shooting-wrist window (~24 frames) to detect the
basketball and measure when it separates from the wrist.

### Why YOLOv8n ONNX (model choice)

| Candidate | Size | CPU latency | `sports ball` class? | Licensing | Decision |
|---|---|---|---|---|---|
| YOLOv8n ONNX | 6 MB | ~40 ms @ 640² | Yes — COCO class 32 (0-indexed) | AGPL-3.0 (research OK) | **Chosen** |
| YOLOv5n | 4 MB | ~35 ms | Yes | GPL-3.0 | 2nd; v8 beats v5 on small-object val-mAP |
| YOLOv10n | 5.6 MB | ~30 ms | Yes | AGPL-3.0 | Less production-tested; riskier 18 h before showcase |
| Detectron2 | 40+ MB | 200+ ms | Yes | Apache-2.0 | Too slow; 200 ms × 24 frames exceeds CPU budget |
| Custom-trained | — | — | Only if labelled | — | No time to label |

**COCO class index correction**: the plan document referenced "class 37". This is the
1-indexed label number. The correct 0-indexed class index used at inference time is **32**.
The ONNX output tensor `[1, 84, 8400]` stores class 32 score at column index `4 + 32 = 36`.

### Runtime: onnxruntime-only, no torch

The full ONNX inference path requires only `onnxruntime` (~4 MB CPU wheel). `torch` is
not imported at any point. `onnxruntime` is added to `requirements.txt` unconditionally
because the `BallDetector` class gracefully disables itself when either:
- `LAKSH_ENABLE_BALL_DETECT != "1"` (env flag absent)
- The ONNX model file is missing
- `onnxruntime` is not installed

Zero inference cost in the disabled path.

---

## Implementation

### Preprocessing: letterbox

Input is resized aspect-ratio-preserving to 640×640 with symmetric gray padding
(`cv2.BORDER_CONSTANT, value=114`). Coordinates are un-letterboxed via `_unletterbox()`
after inference so bounding-box pixels map back to original-frame space.

### Post-processing: greedy NMS

After filtering by `confidence >= 0.35` (COCO eval default), boxes are sorted by
confidence and suppressed with IoU threshold `0.45`. COCO-class score for class 32 is
read from column index 36 of the raw output (col 0-3 = cx,cy,w,h; col 4+cls = class
scores).

### Release detection: hold-then-separate

1. Find the frame of **minimum** ball-to-wrist-pixel distance (the "hold frame" —
   the ball is closest to the hand just before release).
2. Release = first frame AFTER the hold frame where:
   - ball-to-wrist distance > `0.15 × frame_width` (separation threshold), AND
   - ball pixel-y is decreasing (rising in world coords) for `>= 3` consecutive frames.

The 3-frame upward-trajectory requirement rejects frames where the ball momentarily
separates due to dribble or pose noise, and confirms true upward ball flight.

### Wrist NaN masking

Low-visibility MediaPipe wrist frames are masked to NaN in `_interpolate_wrist()` and
filled via linear interpolation from valid neighbours. All-NaN windows fall back to
`(0.5, 0.5)` (frame centre) — a conservative position that prevents the separation
threshold from being spuriously triggered.

### Consensus integration

S3 fires at the detected release frame. In `segment_shots()`:

- **S1 AND S2 AND S3 agree**: shot tagged `"valid"` with `signals_used=["wrist_y_nadir", "elbow_velocity_peak", "ball_leaves_hand"]`.
- **S1 AND S2 agree, S3 absent or disabled**: shot tagged `"valid"` with 2-signal consensus.
- **S3 disagrees with S1+S2**: shot tagged `"degraded"` with `reason_codes=["ball_detect_disagreement"]`.

S3 never overrides S1+S2. It can only raise confidence (agreement) or lower confidence
(disagreement). This preserves the honesty contract: S3 is a third vote, not a veto.

---

## Honesty caveats

1. **COCO training domain**: YOLOv8n was trained on COCO 2017, which includes
   basketballs, soccer balls, baseball, and other sports balls. The model was NOT
   retrained for basketball-only. A soccer ball would also be detected if present.
   This is disclosed in the UI tooltip for the S3 signal.

2. **No re-training on our clips**: We use the pretrained checkpoint as-is. The ONNX
   file SHA is committed to `app/detection/models/yolov8n.onnx.sha256` and verified
   at startup to ensure bit-for-bit reproducibility.

3. **Confidence threshold = 0.35**: The COCO evaluation default. We do not claim
   higher precision than the published COCO metrics for this model.

---

## Validation gate (binding before enabling in production)

Over >= 20 rehearsal clips with known ground-truth release frames:

- **S3–(S1∧S2) agreement rate >= 85%** on shots where both S3 and S1∧S2 fire.
- **False positives (S3 fires on non-release moments) <= 1** across all clips.

If either metric fails, `LAKSH_ENABLE_BALL_DETECT` remains `"0"` and the model ships
as a research artifact only. The 2-signal (S1+S2) pipeline is unaffected and fully
sufficient for the showcase.

---

## Consequences

**Positive**:
- A third physically-independent signal raises joint accuracy toward ~99.9% when enabled.
- Disagreement is surfaced honestly as a `reason_codes` entry; judges can interrogate it.
- Zero runtime cost when disabled.

**Negative / risks**:
- COCO multi-class `sports ball` means non-basketball balls are detected. Acceptable for
  the controlled showcase environment (indoor gym, single ball).
- AGPL-3.0 license is acceptable for research demo; requires disclosure if shipped
  commercially (use YOLOv5n GPL-3.0 or a permissively-licensed detector instead).
- Model download (~6 MB) not part of the Docker image by default; `scripts/download_ball_detector.py`
  must be run before enabling the flag.

---

## References

- Ultralytics YOLOv8n v8.3.0 release: https://github.com/ultralytics/assets/releases/tag/v8.3.0
- COCO dataset labels (0-indexed): https://cocodataset.org/#home (class 32 = sports ball)
- ONNX output format: `[1, 4+80, 8400]` — first 4 cols are cx,cy,w,h; cols 4..83 are class scores.
- ADR 0001: MediaPipe baseline selection.
- ADR 0004: Real-time dual-path architecture (how S3 is gated from the browser counter).
