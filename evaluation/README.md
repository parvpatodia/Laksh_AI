# Evaluation reel (consumer video QA)

1. **Manifest**  
   [manifest.csv](manifest.csv) is tracked and matches [manifest.template.csv](manifest.template.csv). Reset or fork with:  
   `cp evaluation/manifest.template.csv evaluation/manifest.csv`

2. **Add videos**  
   Place `.mp4` files in `evaluation/clips/` using the filenames in the template, **or** edit `path` in `manifest.csv` to match your files.

3. **Run the benchmark**  
   ```bash
   make eval-bench
   # or: bash scripts/run_evaluation_local.sh
   # or: python scripts/benchmark_pipeline.py --manifest evaluation/manifest.csv --out evaluation/results.jsonl --backend mediapipe
   ```

4. **Optional strict gate** (after you set `expect_analysis_mode` / `expect_min_measured` on rows you trust)  
   ```bash
   python scripts/benchmark_pipeline.py --manifest evaluation/manifest.csv --out evaluation/results.jsonl --strict-manifest
   ```

5. **Compare two runs** (e.g. after adding a second pose backend)  
   ```bash
   python scripts/compare_benchmark_results.py evaluation/mp.jsonl evaluation/rtmpose.jsonl -o evaluation/compare.csv \
     --label-a mediapipe --label-b rtmpose
   ```

Details and pass/fail rules: [docs/evaluation_set_spec.md](../docs/evaluation_set_spec.md).

`results.jsonl` and `clips/*.mp4` are gitignored; keep the manifest in version control once curated.

## Gym pose baseline (Phase A)

Gym-specific manifest template: [gym_manifest.template.csv](gym_manifest.template.csv). Place videos under `gym_clips/`. Spec and metrics: [docs/gym_pose_evaluation.md](../docs/gym_pose_evaluation.md).

```bash
python scripts/eval_pose_baseline.py --manifest evaluation/gym_manifest.template.csv \
  --out evaluation/pose_baseline.jsonl
```
