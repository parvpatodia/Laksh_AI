# Laksh_AI — common dev commands
.PHONY: test eval-model eval-bench eval-bench-strict eval-pose-gym eval-gym-validate

test:
	pytest tests/ -q

eval-model:
	python scripts/download_pose_model.py

# Requires evaluation/clips/*.mp4; uses evaluation/manifest.csv
eval-bench:
	@sh scripts/run_evaluation_local.sh

# Gym Phase A: pose-only metrics (needs gym_clips + manifest paths that exist)
eval-pose-gym:
	python scripts/eval_pose_baseline.py --manifest evaluation/gym_manifest.template.csv --out evaluation/pose_baseline.jsonl

# Manifest path check only (no MediaPipe)
eval-gym-validate:
	python scripts/eval_pose_baseline.py --manifest evaluation/gym_manifest.csv --validate-only

# After setting expect_* columns in manifest.csv
eval-bench-strict:
	python scripts/benchmark_pipeline.py --manifest evaluation/manifest.csv --out evaluation/results.jsonl --backend mediapipe --strict-manifest
