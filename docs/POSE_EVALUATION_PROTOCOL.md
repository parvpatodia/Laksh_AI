# Pose evaluation protocol (gym Phase A)

This document defines **what we measure**, **what we do not claim**, and **how to report** results so they are technically defensible—aligned with common practice in computer-vision benchmarking (fixed pipeline, recorded artifacts, explicit limits).

## Scope

- **In scope:** Per-clip **2D pose presence** and **visibility** statistics from MediaPipe Pose Landmarker (heavy), on video normalized when FFmpeg is available.
- **Out of scope:** Sub-pixel accuracy vs. motion capture, 3D joint torques, exercise classification, rep counting, clinical assessment.

## Metric definitions

| Field | Definition |
|--------|------------|
| `n_frames` | Frames decoded and passed through the landmarker after optional FFmpeg normalize and internal resize (max 720p long side). |
| `n_frames_with_pose` | Frames where **both hips** have finite 2D image coordinates in the **first** detected pose. Proxy for “at least one body tracked.” |
| `detection_rate` | `n_frames_with_pose / max(1, n_frames)`. **Not** “probability of correct pose”—only in-model detection continuity. |
| `visibility_core_when_detected` | Mean MediaPipe visibility score on shoulders, hips, knees, ankles **on frames counted as having pose**. |
| `visibility_core_all_frames` | Same joints averaged over **all** frames; missing visibility treated as 0 in the mean. |
| `hip_mid_displacement_median_norm` | Median L2 distance between consecutive hip midpoints in **normalized** image coordinates (stability proxy). |
| `pose_usable_heuristic` | Provisional gate loaded from [evaluation/gym_pose_calibration.json](../evaluation/gym_pose_calibration.json) (validated ranges); embedded in JSONL `provenance`. **Recalibrate** on a curated manifest with product review. |
| `error` | Present when `ok` is false — short human-readable message (not a stable API contract). |
| `reason_codes` | Stable machine-readable strings; see **Reason code registry** below. |

## Reason code registry

Canonical definitions live in `app/pose/reason_codes.py` (`REASON_CODE_DESCRIPTIONS`). Summary:

| Code | When |
|------|------|
| `decode_error` | `ok=false` — capture/decode failed. |
| `pose_init_failed` | `ok=false` — landmarker or model init failed. |
| `short_clip` | `n_frames` &lt; 3. |
| `very_low_detection` | `detection_rate` &lt; 0.05. |
| `low_detection` | 0.05 ≤ `detection_rate` &lt; 0.15. |
| `low_visibility_core` | `visibility_core_when_detected` &lt; 0.25. |
| `multiple_people_detected` | Landmarker saw &gt;1 pose in at least one frame; metrics use **first** pose only. |
| `pose_not_usable_heuristic` | Failed versioned usable gate (after successful decode). |

## Preprocessing invariants

1. **FFmpeg** (when on `PATH`): H.264, constant 30 fps, max 720p height, rotation baked, audio stripped. Same as `KinematicAnalyzer`.
2. **Without FFmpeg:** Original file is used; `ffmpeg_preprocess_applied: false` and summary may include a **warning**. Results are **not** automatically comparable to runs with FFmpeg on HEVC/VFR sources.
3. **`--multipass`:** Selects best of baseline / gamma / denoise frame variants (matches basketball analyzer utility). Default single-pass is **stricter** for raw benchmarking.
4. **`--person-isolation haar_mil_v1` (P2):** Optional OpenCV Haar (upper/full body) + TrackerMIL ROI **before** the pose model. Landmarks are **re-mapped** to normalized coordinates of the **full working frame** (post–max-720), so `detection_rate` and visibility remain comparable to non-isolated runs. `provenance.person_isolation` includes **`haar_detection_attempts`** (actual Haar runs; when no person is ever found, this is about one per `redetect_every_n_frames`, not once per frame), **`frames_full_frame_fallback`**, and **`tracker_update_failures`**. The field **`redetect_events`** duplicates `haar_detection_attempts` for older readers. **`max_people_seen`** is still “max concurrent instances on the **tensor passed to the pose head**,” so it often drops when the crop shows a single subject—do not equate that with “no second person in the original scene.”

## Reproducibility (`provenance`)

Each JSONL row includes `provenance` (when the backend provides it):

- `pose_baseline_schema_version` — bump when metric semantics change (includes **1.2.0** with canonical joint metadata).
- `canonical_joint_schema_version` — version of [app/pose/canonical.py](../app/pose/canonical.py) joint set / axis semantics.
- `canonical_joint_set` — e.g. `coco_17_names` (vocabulary for cross-backbone mapping).
- `canonical_mapping_id` — e.g. `mediapipe_blazepose33_v1` (which index table was used).
- `landmarker_options` — detection/presence/tracking thresholds and model URL (single source with `app/pose/mediapipe_common.py`).
- `mediapipe_package_version` — PyPI wheel version.
- `pose_model_sha256` — SHA-256 of on-disk `.task` file (capped read for files above 64 MiB: digest suffix indicates truncation).
- `platform_sys` — `sys.platform` only (not a full hardware spec).

**Limitation:** Floating-point outputs may still differ slightly across CPUs; hashes and versions capture **software and asset** identity, not bitwise numerical equality.

## CLI semantics (`eval_pose_baseline.py`)

- **Exit 2:** Manifest CSV failed to load (e.g. empty `path` on a data row). stderr prints JSON with `manifest_parse_error`.
- **`--validate-only`:** Exits **1** if any manifest path is missing; does not import MediaPipe/OpenCV.
- **`--strict-manifest`:** After processing, exits **1** if **any** row had `file_not_found` **or** any `expect_*` mismatch on a row that reached the backend. Summary includes `strict_violations` (count of failing rows under this rule). Without `--strict-manifest`, missing files are still logged in JSONL and in `files_missing_on_disk` but do not fail the process.

## Reporting aggregates

- **Do not** treat `mean_detection_rate` over **N** clips as a population statistic without stating **N**, clip diversity, and preprocess path.
- **Do** attach `run_provenance_sample` (or full per-row `provenance`) when sharing JSONL or summary JSON externally.

## Related files

- Implementation: [app/pose/mediapipe_baseline.py](../app/pose/mediapipe_baseline.py), [app/pose/provenance.py](../app/pose/provenance.py)
- Runner: `scripts/eval_pose_baseline.py`
- Capture / internal data: [GYM_EVAL_CAPTURE_AND_DATA.md](./GYM_EVAL_CAPTURE_AND_DATA.md)
- Phase A backbone ADR: [adr/0001-phase-a-mediapipe-baseline.md](./adr/0001-phase-a-mediapipe-baseline.md)
- Backbone upgrade contract: [POSE_UPGRADE_EXECUTION_PLAN.md](./POSE_UPGRADE_EXECUTION_PLAN.md)
- Product roadmap: [docs/RESEARCH_PLAN_POSE_AND_LTX.md](./RESEARCH_PLAN_POSE_AND_LTX.md)
