# Eval scorecard (claim discipline)

Use this template when reporting **pose**, **basketball**, or **gym** benchmark results internally or in PRs.

## Manifest hygiene

- Every **`clip_id`** in `evaluation/gym_manifest*.csv` must be **unique** (loader raises on duplicates) so regressions stay attributable to a single clip.

## Required context (no number without this)

- **Hardware:** CPU/GPU model; CI vs laptop vs server.
- **Mode:** `single_pass` vs `multipass_best` (see [docs/POSE_UPGRADE_EXECUTION_PLAN.md](../docs/POSE_UPGRADE_EXECUTION_PLAN.md) §4).
- **Harness:** Paste output of:

  ```bash
  python3 scripts/eval_scorecard_header.py --manifest evaluation/gym_manifest.csv
  ```

  For archived **JSONL** outputs (P1b backbone A/B or **P2** full-frame vs isolation), add hashes:

  ```bash
  python3 scripts/eval_scorecard_header.py --manifest evaluation/gym_manifest.csv \
    --jsonl evaluation/gym_manifest_pose_full.jsonl \
    --jsonl evaluation/gym_manifest_pose_haar_mil_v1.jsonl
  ```

  (`scorecard_schema_version` **1.1.0** includes `pose_jsonl_artifacts` when `--jsonl` is used.)

- **Backbone:** MediaPipe / RTMPose / other, plus preprocess flags.

## Table

| Metric | Before | After | Notes |
|--------|--------|-------|-------|
| detection_rate (mean) | | | |
| pose_usable_heuristic | | | |
| … | | | |

## Claim tier

Match [docs/POSE_UPGRADE_EXECUTION_PLAN.md](../docs/POSE_UPGRADE_EXECUTION_PLAN.md) §5 — do not upgrade tier without evidence.
