# Laksh_AI — common dev commands
# Use the same interpreter you install requirements into (override: make PYTHON=python3.12 …).
PYTHON ?= python3

.PHONY: test test-pose-core eval-model eval-bench eval-bench-strict eval-pose-gym eval-gym-validate compare-pose-ab eval-pose-ab-orchestrate eval-pose-isolation-ab check-pose-readiness check-pose-readiness-strict lint lint-fix mypy-pose scorecard-header scorecard pose-parity-report freeze-exercise-v0 verify-exercise-v0

test:
	pytest tests/ -q

# Requires: pip install -r requirements-dev.txt
lint:
	ruff check app tests scripts

lint-fix:
	ruff check app tests scripts --fix

mypy-pose:
	mypy app/pose

scorecard-header:
	$(PYTHON) scripts/eval_scorecard_header.py --manifest evaluation/gym_manifest.csv

# Full release scorecard (markdown): header hashes + per-backend aggregates + per-clip drill-down.
# Examples:
#   make scorecard                       # header-only (no JSONL)
#   make scorecard JSONL=evaluation/pose_baseline.jsonl
#   make scorecard JSONL="evaluation/pose_baseline.jsonl evaluation/pose_baseline_roi.jsonl"
scorecard:
	$(PYTHON) scripts/build_scorecard.py --manifest evaluation/gym_manifest.csv \
		$(foreach j,$(JSONL),--jsonl $(j))

# Pose contract + gym manifest/calibration (no full test suite / no cv2-heavy physics tests)
test-pose-core:
	pytest tests/test_canonical_mapping.py tests/test_mapping_rtmpose_coco17.py \
		tests/test_pose_provenance.py tests/test_rtmlib_provenance.py \
		tests/test_gym_baseline_metrics.py tests/test_pose_backends_registry.py \
		tests/test_pose_baseline_compare.py tests/test_eval_readiness.py \
		tests/test_gym_manifest_loader.py tests/test_gym_manifest_template.py \
		tests/test_calibration_load.py tests/test_pose_types.py tests/test_reason_codes_registry.py \
		tests/test_person_isolation.py tests/test_kinematic_canonical_probe.py \
		tests/test_eval_scorecard_header.py tests/test_gym_manifest_hard_template.py \
		tests/test_scorecard_command.py tests/test_build_scorecard.py \
		tests/test_pose_jitter.py tests/test_subject_split_check.py \
		tests/test_pose_calibration.py tests/test_pose_parity_report.py \
		tests/test_exercises_v0.py tests/test_rep_segmenter.py -q

eval-model:
	$(PYTHON) scripts/download_pose_model.py

# Requires evaluation/clips/*.mp4; uses evaluation/manifest.csv
eval-bench:
	@sh scripts/run_evaluation_local.sh

# Gym Phase A: pose-only metrics (needs real paths in manifest + MediaPipe)
# Prefer evaluation/gym_manifest.csv when curated; template is for structure only (PLACEHOLDER paths).
eval-pose-gym:
	$(PYTHON) scripts/eval_pose_baseline.py --manifest evaluation/gym_manifest.csv --out evaluation/pose_baseline.jsonl

# Manifest path check only (no MediaPipe)
eval-gym-validate:
	$(PYTHON) scripts/eval_pose_baseline.py --manifest evaluation/gym_manifest.csv --validate-only

# Static checks only (no inference): deps, .task file, optional gym manifest path stats
check-pose-readiness:
	$(PYTHON) scripts/check_pose_eval_readiness.py --manifest evaluation/gym_manifest.csv

check-pose-readiness-strict:
	$(PYTHON) scripts/check_pose_eval_readiness.py --manifest evaluation/gym_manifest.csv --strict

# P1b: compare two JSONL runs (L0 summary). Example paths — generate with eval_pose_baseline first.
compare-pose-ab:
	$(PYTHON) scripts/compare_pose_baseline_jsonl.py \
		--a evaluation/pose_baseline_mediapipe.jsonl \
		--b evaluation/pose_baseline_rtmpose.jsonl

# Run MediaPipe + RTMPose eval on gym manifest then compare (needs optional rtmlib + network for first RTMPose run)
eval-pose-ab-orchestrate:
	$(PYTHON) scripts/run_pose_ab_eval_compare.py --manifest evaluation/gym_manifest.csv

# P2: same manifest — full frame vs haar_mil_v1 ROI; prints compare JSONL + p2_l0 multi-person tallies
eval-pose-isolation-ab:
	$(PYTHON) scripts/run_pose_isolation_ab_compare.py --manifest evaluation/gym_manifest.csv

# After setting expect_* columns in manifest.csv
eval-bench-strict:
	$(PYTHON) scripts/benchmark_pipeline.py --manifest evaluation/manifest.csv --out evaluation/results.jsonl --backend mediapipe --strict-manifest

# ADR 0002 Phase C: canonical-vs-legacy parity gate.
# Generate input JSONL with LAKSH_USE_CANONICAL_JOINTS=1 first, e.g.:
#   LAKSH_USE_CANONICAL_JOINTS=1 make eval-bench-strict
# Then: make pose-parity-report JSONL=evaluation/results.jsonl
pose-parity-report:
	$(PYTHON) scripts/pose_parity_report.py --jsonl $(JSONL)

# Milestone 1 (GOALS.md): freeze / verify exercise v0 taxonomy.
# Run freeze-exercise-v0 after editing app/gym/exercises_v0.py; commit the
# regenerated evaluation/exercise_v0_manifest.json alongside the code change.
freeze-exercise-v0:
	$(PYTHON) scripts/freeze_exercise_v0.py

verify-exercise-v0:
	$(PYTHON) scripts/freeze_exercise_v0.py --verify
