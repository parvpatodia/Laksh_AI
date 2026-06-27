#!/usr/bin/env python3
"""
Run KinematicAnalyzer on a manifest or directory of videos; write JSONL for QA / backbone comparison.

For **gym pose-only** metrics (Phase A, no basketball metrics), use:
  python scripts/eval_pose_baseline.py --manifest evaluation/gym_manifest.template.csv --out evaluation/pose_baseline.jsonl

Usage:
  python scripts/benchmark_pipeline.py --dir ./evaluation/clips --out ./evaluation/results.jsonl
  python scripts/benchmark_pipeline.py --manifest evaluation/manifest.csv --strict-manifest
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.physics_engine import KinematicAnalyzer, METRIC_KEYS  # noqa: E402


def _metric_counts(metric_status: dict) -> dict[str, int]:
    out = {"measured": 0, "predicted": 0, "unavailable": 0}
    for v in (metric_status or {}).values():
        if not isinstance(v, dict):
            continue
        s = v.get("source")
        if s in out:
            out[s] += 1
    return out


def summarize_analysis(result: dict) -> dict:
    tel = result.get("telemetry") or {}
    dm = tel.get("detection_metadata") or {}
    ms = result.get("metric_status") or {}
    counts = _metric_counts(ms)
    row = {
        "analysis_mode": result.get("analysis_mode"),
        "fallback_reason_codes": result.get("fallback_reason_codes") or [],
        "metric_source_counts": counts,
        "shot_type": tel.get("shot_type"),
        "selected_preprocess_pass": dm.get("selected_preprocess_pass"),
        "people_detected_max": dm.get("people_detected_max"),
    }
    # ADR 0002 Phase C: pass through canonical-vs-legacy parity probe when emitted
    # (LAKSH_USE_CANONICAL_JOINTS=1). Absent when flag off; scripts/pose_parity_report.py
    # reduces this across a basketball manifest run for the default-flip gate.
    cprobe = tel.get("canonical_joint_path")
    if cprobe is not None:
        row["canonical_joint_path"] = cprobe
    return row


def check_manifest_expectations(summary: dict, expect_mode: str | None, expect_min_measured: int | None) -> list[str]:
    errs: list[str] = []
    mode = summary.get("analysis_mode")
    if expect_mode and mode != expect_mode:
        errs.append(f"expected analysis_mode {expect_mode!r}, got {mode!r}")
    if expect_min_measured is not None:
        n = summary["metric_source_counts"].get("measured", 0)
        if n < expect_min_measured:
            errs.append(f"expected at least {expect_min_measured} measured metrics, got {n}")
    return errs


def run_one(video_path: Path, clip_id: str, backend: str) -> dict:
    t0 = time.perf_counter()
    try:
        result = KinematicAnalyzer(str(video_path)).analyze()
        err = None
    except Exception as e:
        result = None
        err = f"{type(e).__name__}: {e}"
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    base = {
        "clip_id": clip_id,
        "video_path": str(video_path),
        "backend": backend,
        "ok": err is None,
        "error": err,
        "elapsed_ms": elapsed_ms,
    }
    if result is not None:
        base["summary"] = summarize_analysis(result)
        base["stats_present"] = {k: result.get(k) is not None for k in METRIC_KEYS}
    return base


def load_manifest(
    path: Path, base_dir: Path
) -> list[tuple[str, Path, str | None, int | None, str | None]]:
    rows: list[tuple[str, Path, str | None, int | None, str | None]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cid = (r.get("clip_id") or "").strip() or Path(r["path"]).stem
            rel = r["path"].strip()
            vp = (base_dir / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
            em = (r.get("expect_analysis_mode") or "").strip() or None
            emm = (r.get("expect_min_measured") or "").strip()
            emm_val = int(emm) if emm.isdigit() else None
            tags = (r.get("tags") or "").strip() or None
            rows.append((cid, vp, em, emm_val, tags))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark video analysis pipeline (JSONL out).")
    ap.add_argument("--manifest", type=Path, help="CSV with columns clip_id,path,tags,notes,expect_*")
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Base dir for relative manifest paths (default: repo root)",
    )
    ap.add_argument("--dir", type=Path, help="Directory of .mp4 files")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    ap.add_argument(
        "--strict-manifest",
        action="store_true",
        help="Exit 1 if any clip path is missing or any expect_* expectation fails",
    )
    ap.add_argument(
        "--backend",
        choices=("mediapipe",),
        default="mediapipe",
        help="Pose stack label for JSONL (only mediapipe in this repo; Apex.ai worktree may add rtmpose).",
    )
    args = ap.parse_args()

    if bool(args.manifest) == bool(args.dir):
        ap.error("Provide exactly one of --manifest or --dir")

    jobs: list[tuple[str, Path, str | None, int | None, str | None]] = []
    if args.manifest:
        # Paths in CSV are repo-root-relative (see docs/evaluation_set_spec.md).
        base = args.manifest_dir or REPO_ROOT
        jobs = load_manifest(args.manifest, base)
    else:
        for p in sorted(args.dir.glob("*.mp4")):
            jobs.append((p.stem, p.resolve(), None, None, None))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    strict_violations = 0
    with args.out.open("w", encoding="utf-8") as out_f:
        for clip_id, vp, expect_mode, expect_min_meas, tags in jobs:
            if not vp.is_file():
                row = {
                    "clip_id": clip_id,
                    "video_path": str(vp),
                    "backend": args.backend,
                    "tags": tags,
                    "ok": False,
                    "error": "file_not_found",
                    "elapsed_ms": 0.0,
                }
                out_f.write(json.dumps(row) + "\n")
                if args.strict_manifest:
                    strict_violations += 1
                continue
            row = run_one(vp, clip_id, args.backend)
            if tags:
                row["tags"] = tags
            if row.get("summary"):
                errs = check_manifest_expectations(row["summary"], expect_mode, expect_min_meas)
                if errs:
                    row["expectation_errors"] = errs
                    if args.strict_manifest:
                        strict_violations += 1
            out_f.write(json.dumps(row) + "\n")

    return 1 if args.strict_manifest and strict_violations > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
