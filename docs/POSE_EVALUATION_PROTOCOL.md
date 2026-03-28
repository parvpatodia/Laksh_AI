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

## Preprocessing invariants

1. **FFmpeg** (when on `PATH`): H.264, constant 30 fps, max 720p height, rotation baked, audio stripped. Same as `KinematicAnalyzer`.
2. **Without FFmpeg:** Original file is used; `ffmpeg_preprocess_applied: false` and summary may include a **warning**. Results are **not** automatically comparable to runs with FFmpeg on HEVC/VFR sources.
3. **`--multipass`:** Selects best of baseline / gamma / denoise frame variants (matches basketball analyzer utility). Default single-pass is **stricter** for raw benchmarking.

## Reproducibility (`provenance`)

Each JSONL row includes `provenance` (when the backend provides it):

- `pose_baseline_schema_version` — bump when metric semantics change.
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
- Product roadmap: [docs/RESEARCH_PLAN_POSE_AND_LTX.md](./RESEARCH_PLAN_POSE_AND_LTX.md)
