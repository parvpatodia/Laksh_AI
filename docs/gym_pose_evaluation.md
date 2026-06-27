# Gym pose baseline (Phase A)

This implements **Phase A — measurement spine** from [RESEARCH_PLAN_POSE_AND_LTX.md](./RESEARCH_PLAN_POSE_AND_LTX.md): pose detection and quality metrics **without** basketball shot logic, so we can benchmark **gym** clips and later compare **MediaPipe vs RTMPose** (or others) on the same manifest.

**Formal definitions and reporting rules:** [POSE_EVALUATION_PROTOCOL.md](./POSE_EVALUATION_PROTOCOL.md). **How to record clips and handle internal eval data:** [GYM_EVAL_CAPTURE_AND_DATA.md](./GYM_EVAL_CAPTURE_AND_DATA.md). Each JSONL row includes **`provenance`** (versions, model hash, landmarker options) for reproducibility.

**Before a long eval:** run **`make check-pose-readiness`** (or **`make check-pose-readiness-strict`**) or `python3 scripts/check_pose_eval_readiness.py --manifest evaluation/gym_manifest.csv`. The JSON report (`report_schema_version` **1.2.0**) uses per-package **`version`** plus **`probe_error`** (native deps are probed in a subprocess so the reporter survives bad wheels/sandboxes). Override the interpreter with **`make PYTHON=python …`** if `python3` is not where you installed `requirements.txt`. No inference; CI runs strict readiness after the model download step.

**P2 (optional person ROI):** `--person-isolation haar_mil_v1` on `scripts/eval_pose_baseline.py` runs OpenCV Haar (bundled cascades) + TrackerMIL before the pose model; see [POSE_EVALUATION_PROTOCOL.md](./POSE_EVALUATION_PROTOCOL.md) §preprocessing and `app/pose/person_isolation.py`.

**P2 full-frame vs isolation (L0 A/B):** `make eval-pose-isolation-ab` or `python3 scripts/run_pose_isolation_ab_compare.py --manifest evaluation/gym_manifest.csv` runs the manifest twice (full frame, then isolation) and prints `compare_pose_baseline_jsonl` output including **`p2_l0`** (`multiple_people_detected` counts and cleared/introduced clip samples). Curate crowded clips using [evaluation/gym_manifest_hard.template.csv](../evaluation/gym_manifest_hard.template.csv) (copy to a real CSV with paths on disk).

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
| `max_people_seen` | Max concurrent poses returned in any frame (`num_poses=2` config). Values &gt;1 add `multiple_people_detected` to `reason_codes` (first pose only is aggregated). |
| `reason_codes` | Stable diagnostic / failure codes — [POSE_EVALUATION_PROTOCOL.md](./POSE_EVALUATION_PROTOCOL.md#reason-code-registry). |
| `selected_pass` | `baseline_only` or `multipass_best:…` matching [KinematicAnalyzer](../app/physics_engine.py) preprocess variants when `--multipass` is set. |
| `pose_usable_heuristic` | **Provisional** gate for “good enough for kinematics v0” (see below). |

These are **diagnostic**, not clinical. They do not claim 3D ground truth.

## Provisional “usable” gate

Default numeric thresholds are loaded from [evaluation/gym_pose_calibration.json](../evaluation/gym_pose_calibration.json) (validated by `app/pose/calibration.py`); invalid files fall back to the same built-in defaults:

- `min_detection_rate` 0.25
- `min_visibility_core_when_detected` 0.35
- `min_n_frames` 15

**Recalibrate** against curated gym reels and product tolerance; commit JSON changes with the eval run or PR that justifies them.

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

## Backends

| `--backend` | Notes |
|-------------|--------|
| `mediapipe` | Default; MediaPipe Pose Landmarker heavy. |
| `rtmpose` | Optional: `pip install -r requirements-pose-optional.txt` (`rtmlib` + `onnxruntime`). Uses YOLOX + RTMPose ONNX; **first run may download** OpenMMLab zips. Env: `RTMPOSE_MODE` (`lightweight` / `balanced` / `performance`), `RTMPOSE_DEVICE` (`cpu` / `cuda` / `mps`). |

Provenance differs by backend; both include `canonical_joint_schema_version` and `pose_baseline_schema_version`.

## P1b — compare two JSONL runs (L0)

After generating two files with the **same** `clip_id` rows (same manifest, different `--backend`):

```bash
python scripts/compare_pose_baseline_jsonl.py \
  --a evaluation/pose_baseline_mediapipe.jsonl \
  --b evaluation/pose_baseline_rtmpose.jsonl \
  --per-clip-out evaluation/pose_ab_per_clip.jsonl
```

The printed JSON includes `comparison_purpose`: this is **pipeline A vs B**, not labeled keypoint accuracy. It also includes **`confound_notes`** when `ffmpeg_preprocess_applied` differs between runs for the same clip (deltas mix decode path + backbone). **Robust stats:** `median_delta_detection_rate_b_minus_a` and min/max deltas alongside the mean.

Use `make compare-pose-ab` if you use the default filenames from the Makefile, or **`make eval-pose-ab-orchestrate`** to run both evals and compare in one step (RTMPose optional; exit `2` if RTMPose fails but MediaPipe succeeded).

## Commands

```bash
# Check paths before a long pose run (exit 1 if any file missing)
python scripts/eval_pose_baseline.py --manifest evaluation/gym_manifest.csv --validate-only

# After adding videos (MediaPipe)
python scripts/eval_pose_baseline.py \
  --manifest evaluation/gym_manifest.csv \
  --out evaluation/pose_baseline.jsonl

# Same manifest, RTMPose path (optional deps)
python scripts/eval_pose_baseline.py \
  --manifest evaluation/gym_manifest.csv \
  --backend rtmpose \
  --out evaluation/pose_baseline_rtmpose.jsonl

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

1. Fill `gym_clips/` following [GYM_EVAL_CAPTURE_AND_DATA.md](./GYM_EVAL_CAPTURE_AND_DATA.md); set `expect_*` only on rows after review.
2. Add a second backend implementing the same `PoseBaselineResult` fields; run twice and use `scripts/compare_benchmark_results.py` patterns (or extend that script for pose JSONL). Backbone rationale: [adr/0001-phase-a-mediapipe-baseline.md](./adr/0001-phase-a-mediapipe-baseline.md).
3. Replace or tune the heuristic gate from measured data on the curated set.
