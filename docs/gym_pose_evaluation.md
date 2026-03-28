# Gym pose baseline (Phase A)

This implements **Phase A — measurement spine** from [RESEARCH_PLAN_POSE_AND_LTX.md](./RESEARCH_PLAN_POSE_AND_LTX.md): pose detection and quality metrics **without** basketball shot logic, so we can benchmark **gym** clips and later compare **MediaPipe vs RTMPose** (or others) on the same manifest.

**Formal definitions and reporting rules:** [POSE_EVALUATION_PROTOCOL.md](./POSE_EVALUATION_PROTOCOL.md). Each JSONL row includes **`provenance`** (versions, model hash, landmarker options) for reproducibility.

## What gets measured

For each clip, [app/pose/mediapipe_baseline.py](../app/pose/mediapipe_baseline.py) reports:

| Field | Meaning |
|--------|--------|
| `n_frames` | Frames processed after FFmpeg normalize (H.264, 30 fps, max 720p). |
| `n_frames_with_pose` | Frames where both hips have finite 2D coordinates (proxy for “a body is tracked”). |
| `detection_rate` | `n_frames_with_pose / n_frames`. |
| `visibility_core_when_detected` | Mean MediaPipe visibility score on shoulders, hips, knees, ankles **on frames with pose**. |
| `visibility_core_all_frames` | Same joints averaged over **all** frames (missing → 0 contribution to mean). |
| `hip_mid_displacement_median_norm` | Median frame-to-frame L2 move of hip midpoint in **normalized** image coordinates (stability proxy; high values = jitter or fast motion). |
| `max_people_seen` | Max concurrent poses returned in any frame (`num_poses=2` config). |
| `selected_pass` | `baseline_only` or `multipass_best:…` matching [KinematicAnalyzer](../app/physics_engine.py) preprocess variants when `--multipass` is set. |
| `pose_usable_heuristic` | **Provisional** gate for “good enough for kinematics v0” (see below). |

These are **diagnostic**, not clinical. They do not claim 3D ground truth.

## Provisional “usable” gate

Default thresholds in code (`app/pose/mediapipe_baseline.py`):

- `detection_rate >= 0.25`
- `visibility_core_when_detected >= 0.35`
- `n_frames >= 15`

**You must recalibrate** these against real gym reels and product tolerance; commit changes with a note in the manifest run or PR.

## Manifest

- Template: [evaluation/gym_manifest.template.csv](../evaluation/gym_manifest.template.csv)
- Copy to `evaluation/gym_manifest.csv` and replace `PLACEHOLDER_*.mp4` with real files under `evaluation/gym_clips/` (gitignored).

**Path resolution:** Relative `path` cells are joined to the **repository root** (where `app/` lives), not to the folder containing the CSV. Example: `evaluation/clips/foo.mp4` resolves to `<repo>/evaluation/clips/foo.mp4`. Override with `--manifest-dir` if needed.

Columns:

- `clip_id`, `path`, `tags`, `notes` — same spirit as basketball manifest.
- `exercise_id` — optional label for future rep segmentation (e.g. `squat`).
- `expect_pose_usable` — optional `yes` / `no` for strict CI once clips are curated.
- `expect_min_detection_rate` — optional float (0–1).

## Usable-gate calibration

Thresholds for `pose_usable_heuristic` are **versioned in** [gym_pose_calibration.json](gym_pose_calibration.json) (not hardcoded only). Invalid JSON or out-of-range values fall back to built-in defaults with a logged error; provenance records `calibration_source` and file SHA-256.

## FFmpeg

The same **FFmpeg normalize** step as `KinematicAnalyzer` is applied when `ffmpeg` is on your `PATH`. If it is missing, the pipeline **falls back to the original file** (your log: `FFmpeg unavailable … using original video`). That is **expected** and not a MediaPipe bug; iPhone **HEVC / VFR** clips may behave differently than after normalize.

Install (macOS): `brew install ffmpeg`

Each JSONL row includes **`ffmpeg_preprocess_applied`** so you can tell whether numbers are comparable to Docker/CI runs that include FFmpeg.

## Commands

```bash
# Check paths before a long MediaPipe run (exit 1 if any file missing)
python scripts/eval_pose_baseline.py --manifest evaluation/gym_manifest.csv --validate-only

# After adding videos
python scripts/eval_pose_baseline.py \
  --manifest evaluation/gym_manifest.csv \
  --out evaluation/pose_baseline.jsonl

# Align preprocess with production basketball analyzer (slower, often higher utility)
python scripts/eval_pose_baseline.py \
  --manifest evaluation/gym_manifest.csv \
  --out evaluation/pose_baseline.jsonl \
  --multipass

# Fail CI if expectations on manifest rows are violated
python scripts/eval_pose_baseline.py \
  --manifest evaluation/gym_manifest.csv \
  --out evaluation/pose_baseline.jsonl \
  --strict-manifest
```

A one-line JSON **summary** prints to stdout (mean detection rate, usable count).

## Shared FFmpeg path

[app/pose/preprocess.py](../app/pose/preprocess.py) implements the same normalization as `KinematicAnalyzer._prepare_video` used to, so **basketball** and **gym baseline** stay aligned on decode/VFR behavior.

## Next steps (still Phase A)

1. Fill `gym_clips/` with licensed footage; set `expect_*` on rows you trust.
2. Add a second backend implementing the same `PoseBaselineResult` fields; run twice and use `scripts/compare_benchmark_results.py` patterns (or extend that script for pose JSONL).
3. Replace or tune the heuristic gate from measured data.
