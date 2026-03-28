# `app.pose` — gym Phase A pose spine

Small, testable modules **below** `physics_engine` basketball logic. Goal: **one contract** (`PoseBaselineResult` + JSONL) so MediaPipe, RTMPose, or other 2D backbones can be swapped after benchmarked comparison.

| Module | Purpose |
|--------|---------|
| `preprocess.py` | FFmpeg normalize (shared with `KinematicAnalyzer`). |
| `mediapipe_common.py` | Landmarker construction + **single source** for confidence constants. |
| `mediapipe_baseline.py` | Frame loop, metrics, `pose_usable_heuristic` vs calibration file. |
| `calibration.py` | Load `evaluation/gym_pose_calibration.json` with validation + fallback. |
| `provenance.py` | Versions, model hash, gate snapshot, calibration record for JSONL. |
| `types.py` | `PoseBaselineResult`, `merge_reason_codes`. |
| `gym_manifest.py` | CSV load + expectation checks (**no OpenCV** — safe for `--validate-only`). |
| `backends/` | `MediaPipePoseBackend`; extend with `rtmpose` when ready. |

Specs: [docs/POSE_EVALUATION_PROTOCOL.md](../docs/POSE_EVALUATION_PROTOCOL.md), [docs/gym_pose_evaluation.md](../docs/gym_pose_evaluation.md).
