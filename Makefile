# Laksh_AI — common dev commands
.PHONY: test eval-model eval-bench eval-bench-strict

test:
	pytest tests/ -q

eval-model:
	python scripts/download_pose_model.py

# Requires evaluation/clips/*.mp4; uses evaluation/manifest.csv
eval-bench:
	@sh scripts/run_evaluation_local.sh

# After setting expect_* columns in manifest.csv
eval-bench-strict:
	python scripts/benchmark_pipeline.py --manifest evaluation/manifest.csv --out evaluation/results.jsonl --backend mediapipe --strict-manifest
