# `app.pose` — gym Phase A pose spine

Small, testable modules **below** `physics_engine` basketball logic. Goal: **one contract** (`PoseBaselineResult` + JSONL) so MediaPipe, RTMPose, or other 2D backbones can be swapped after benchmarked comparison.

| Module | Purpose |
|--------|---------|
| `preprocess.py` | FFmpeg normalize (shared with `KinematicAnalyzer`). |
| `mediapipe_common.py` | Landmarker construction + **single source** for confidence constants. |
| `mediapipe_baseline.py` | Frame loop, metrics, `pose_usable_heuristic` vs calibration file. |
| `calibration.py` | Load `evaluation/gym_pose_calibration.json` with validation + fallback. |
| `provenance.py` | Versions, model hash, gate snapshot, calibration record for JSONL. |
| `types.py` | `PoseBaselineResult` (re-exports `merge_reason_codes`). |
| `reason_codes.py` | Stable `reason_codes` taxonomy + `merge_reason_codes`. |
| `canonical.py` | COCO-17 **names** + `JointObservation` + schema version (cross-backbone contract). |
| `mapping_mediapipe.py` | BlazePose 33 landmarks → canonical joints. |
| `gym_manifest.py` | CSV load + expectation checks (**no OpenCV** — safe for `--validate-only`). |
| `gym_baseline_metrics.py` | Shared detection/visibility/stability aggregates for all baselines. |
| `mapping_rtmpose_coco17.py` | RTMPose COCO-17 pixels → canonical + gym row layout. |
| `rtmpose_baseline.py` | Full-clip RTMPose eval (optional `rtmlib`). |
| `backends/` | `MediaPipePoseBackend`, `RTMPosePoseBackend` (`get_pose_backend`). |
| `person_isolation.py` | **P2** optional Haar + TrackerMIL ROI; remap to full-frame normalized coords. |
| `pose_baseline_compare.py` | Load/compare JSONL runs (P1b L0 A/B). |
| `eval_readiness.py` | Static readiness report (JSON schema 1.2.0: dep blocks + `pose_landmarker_task` SHA-256; heavy imports probed in subprocess). |
| `scorecard_command.py` | Build quoted `eval_scorecard_header.py` command for orchestration reports (P1b / P2). |

Specs: [docs/POSE_EVALUATION_PROTOCOL.md](../docs/POSE_EVALUATION_PROTOCOL.md), [docs/gym_pose_evaluation.md](../docs/gym_pose_evaluation.md).
