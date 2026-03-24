# Evaluation set specification

Purpose: a **fixed reel** of real-world clips to measure whether the video analysis pipeline degrades honestly (partial metrics + reasons) instead of failing silently or over-claiming precision.

See also [VALIDATION_STRATEGY.md](./VALIDATION_STRATEGY.md) for lab-scale validation; this doc is for **continuous QA** on consumer-grade footage.

## Target size

- **Minimum**: 20 clips covering every category below at least once.
- **Comfortable**: 40–50 clips with 2–3 examples per hard subcategory (e.g. side-view Shorts, heavy overlays).

## Clip inventory (categories)

Use consistent **tags** (comma-separated in the manifest) so benchmarks can filter results.

| Category ID | Description | Why it matters |
|-------------|-------------|----------------|
| `phone_clean` | Single shooter, good light, ~720p+, landscape or portrait | Regression anchor; expect `full` or near-full availability |
| `phone_low_light` | Dark gym / evening | Preprocess pass selection, visibility |
| `phone_far_subject` | Shooter small in frame | Scale / detection limits |
| `yt_short_reencode` | Downloaded Short (H.264, overlays, graphics) | Decode + artifact passes |
| `broadcast_crop` | TV/wide shot, small body | Same as far subject + compression |
| `side_view_ft` | Free throw or set shot, side camera | `set_shot` phase path, 2D fallbacks |
| `side_view_jumper` | Jump shot, side camera | World-z weakness; predicted hip/knee common |
| `multi_person` | Defender or crowd visible | Subject consistency; partial mode |
| `occlusion` | Ball or limb occludes joints | Unavailable metrics with clear reasons |
| `vfr_hevc` | iPhone-style VFR or HEVC source (before normalize) | FFmpeg normalize path |
| `short_clip` | Intentionally &lt; ~1 s usable pose | `short_clip` / `low_detections` handling |

## Required metadata per clip

[evaluation/manifest.csv](../evaluation/manifest.csv) is tracked by default (same rows as [manifest.template.csv](../evaluation/manifest.template.csv)). [manifest.example.csv](../evaluation/manifest.example.csv) is a minimal two-row sample.

- `clip_id` — stable string (e.g. `sv_ft_001`).
- `path` — relative path from repo root or from `--manifest-dir`.
- `tags` — category IDs from the table above.
- `notes` — one line (source, athlete if public, date).
- `expect_analysis_mode` — optional: `full`, `partial`, or `fallback` (empty = no strict gate).
- `expect_min_measured` — optional integer: minimum count of metrics with `source=measured` (empty = skip).

## Pass / fail criteria (pipeline-level)

These apply to **current** backend output (`analysis_mode`, `fallback_reason_codes`, `metric_status`, `telemetry.shot_type`).

### Hard fail (must fix before release)

1. **`analysis_mode=fallback`** on a clip tagged `phone_clean` with no `expect_analysis_mode` override — usually indicates deploy or decode regression.
2. **`analysis_exception`** in `fallback_reason_codes` on any clip in the golden set — uncaught error.
3. **Silent “all measured”** when tags include `side_view_jumper` or `yt_short_reencode` and expert review says depth is unreliable — means gating is lying; adjust `metric_status` / sources.

### Soft fail (track over time)

1. **`partial`** on `phone_clean` — investigate preprocess or detection.
2. **More than 4 `unavailable` metrics** on `side_view_ft` — acceptable only if at least **arc + one of knee/elbow** is non-null with `predicted` or `measured` after 2D fallback work.
3. **`set_shot` vs `jump_shot`**: for tags `side_view_ft`, prefer `telemetry.shot_type=set_shot` when motion is minimal (heuristic check); log mismatches for tuning.

### Per-metric expectations (when expert label exists)

For a subset of clips (5–10) with manual labels:

| Metric | Rule |
|--------|------|
| `knee_angle` | Within ±15° of expert label **or** marked `predicted`/`unavailable` with reason (not fake `measured` high confidence). |
| `elbow_angle` | Same as knee. |
| `hip_rotation_deg` | Side-view: `predicted` or `unavailable` acceptable; `measured` only if labeler agrees within ±10°. |
| `shot_arc_deg` | Plausible band 30–75° **or** explicit low-confidence / predicted path. |

## How to run the benchmark

From repo root (with dependencies and `pose_landmarker_heavy.task` available):

```bash
# manifest.csv already exists; add videos under evaluation/clips/ (see evaluation/README.md)
make eval-bench
# or: python scripts/benchmark_pipeline.py --manifest evaluation/manifest.csv --out evaluation/results.jsonl --backend mediapipe
```

Or scan a directory of `.mp4` files:

```bash
python scripts/benchmark_pipeline.py --dir path/to/clips --out evaluation/results.jsonl --backend mediapipe
```

Optional gates (exit non-zero if any row fails):

```bash
python scripts/benchmark_pipeline.py --manifest evaluation/manifest.csv --strict-manifest --out evaluation/results.jsonl
```

Compare two JSONL runs (e.g. after a second backbone exists):

```bash
python scripts/compare_benchmark_results.py evaluation/run_a.jsonl evaluation/run_b.jsonl -o evaluation/compare.csv --label-a mediapipe --label-b other
```

CI: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs `pytest` (includes manifest schema test).

## Second pose backbone (future)

When adding RTMPose, YOLO-pose, or another extractor:

1. Keep this manifest unchanged; add column `backend` or run the script twice with `--backend mediapipe|rtmpose`.
2. Compare JSONL summaries: same `clip_id`, different `summary.*` fields.
3. Promote a backbone only if it **improves** `measured` count on `yt_short_reencode` and `side_view_*` without increasing `fallback` on `phone_clean`.

## Privacy

Use only footage you have rights to store (your own, licensed, or public with attribution per your policy). Do not commit large binaries to git; keep clips in private storage and point `path` at checkout location.
