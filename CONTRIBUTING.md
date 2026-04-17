# Contributing

## First PR checklist

1. **Python 3.11** — same as CI (see `.github/workflows/ci.yml`).
2. **Install:** `pip install -r requirements.txt` and optionally `pip install -r requirements-dev.txt` before `make lint` / `make mypy-pose`.
3. **Pose model:** `python scripts/download_pose_model.py` (verifies SHA-256 of `pose_landmarker_heavy.task`).
4. **Tests:** `pytest tests/ -q` or at least `make test-pose-core` for pose-only changes.
5. **Eval claims:** If you change preprocess, pose, or metrics, read [docs/POSE_UPGRADE_EXECUTION_PLAN.md](docs/POSE_UPGRADE_EXECUTION_PLAN.md) and attach a scorecard per [evaluation/SCORECARD_TEMPLATE.md](evaluation/SCORECARD_TEMPLATE.md) when reporting numbers.
6. **Gym manifest CSV:** Each **`clip_id`** must be unique (see [app/pose/gym_manifest.py](app/pose/gym_manifest.py)); duplicates break longitudinal eval comparison.
7. **P3 canonical parity (optional):** Set `LAKSH_USE_CANONICAL_JOINTS=1` to record per-frame canonical maps during extraction and populate `telemetry.canonical_joint_path` with legacy vs canonical 2D angle deltas at dip/release ([docs/adr/0002-p3-canonical-in-kinematic-analyzer.md](docs/adr/0002-p3-canonical-in-kinematic-analyzer.md)). Default metrics unchanged. When multivariant preprocessing runs, canonical mapping is computed for each variant (extra CPU); analysis still uses the winning variant’s metrics.

## Layout

Application code lives in **`app/`**. The dashboard is **`static/dashboard.html`**. See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for data flow.

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # ruff + mypy (optional but matches CI)
python scripts/download_pose_model.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Optional: `LOG_LEVEL=DEBUG` for verbose pose pipeline logs.

**FFmpeg** should be on `PATH` for video normalize (HEVC→H.264, VFR, rotation). macOS: `brew install ffmpeg`. Without it, analysis still runs on the original file; pose baseline JSONL marks `ffmpeg_preprocess_applied: false`.

## Lint and types (CI-parity)

```bash
make lint
make mypy-pose
```

## Tests

```bash
pytest tests/ -q
```

Golden regression runs only when `tests/fixtures/golden_shot.mp4` and `golden_expected.json` are present (see `docs/GOLDEN_VIDEO_GUIDE.md`).

Gym Phase A pose baseline: see `docs/gym_pose_evaluation.md` and `make eval-pose-gym` (requires valid paths in `evaluation/gym_manifest.csv` and MediaPipe). Check manifest paths without running pose: `make eval-gym-validate`.

**Pose upgrade roadmap (canonical joints, second backbone, A/B gates):** [docs/POSE_UPGRADE_EXECUTION_PLAN.md](docs/POSE_UPGRADE_EXECUTION_PLAN.md). Run contract tests only: `make test-pose-core`.

**RTMPose baseline (optional):** `pip install -r requirements-pose-optional.txt` then `python scripts/eval_pose_baseline.py --backend rtmpose ...`. May install `opencv-contrib-python` alongside headless OpenCV—check your deployment. First run can download ONNX bundles (network).

**Pose eval readiness (no video):** `make check-pose-readiness` or `make check-pose-readiness-strict` — JSON report (`report_schema_version` **1.2.0**: dep blocks + `pose_landmarker_task` **SHA-256** vs [app/pose/expected_artifacts.py](app/pose/expected_artifacts.py)). The Makefile defaults to **`PYTHON ?= python3`** so it matches a typical install; override if your venv uses another binary. CI runs strict readiness after `download_pose_model.py`.

**Scorecard header (for archived eval JSONL):** `make scorecard-header` — git commit + file hashes ([evaluation/SCORECARD_TEMPLATE.md](evaluation/SCORECARD_TEMPLATE.md)).

**P1b compare (L0):** After two JSONL files from the same manifest: `python scripts/compare_pose_baseline_jsonl.py --a evaluation/pose_mp.jsonl --b evaluation/pose_rtm.jsonl` — summarizes detection-rate deltas per clip union; not ground-truth accuracy. Orchestrated run: `make eval-pose-ab-orchestrate` or `python scripts/run_pose_ab_eval_compare.py` (checks FFmpeg mismatch confounds in output).

## Style

- Prefer **`logging`** over **`print`** in application code.
- Keep API response fields backward-compatible when possible.
