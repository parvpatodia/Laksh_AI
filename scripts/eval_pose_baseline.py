#!/usr/bin/env python3
"""
Phase A: run pose-only baseline on the gym evaluation manifest (MediaPipe).

Produces JSONL rows compatible with backbone A/B once RTMPose (or other) implements
the same PoseBaselineResult fields via app.pose.backends.get_pose_backend.

Usage:
  python scripts/eval_pose_baseline.py --manifest evaluation/gym_manifest.template.csv \\
      --out evaluation/pose_baseline.jsonl
  python scripts/eval_pose_baseline.py --manifest evaluation/gym_manifest.csv --strict-manifest --multipass
  python scripts/eval_pose_baseline.py --manifest evaluation/gym_manifest.csv --validate-only
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Gym pose baseline (Phase A) — JSONL output.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Base for relative paths in CSV (default: git repo root)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL (required unless --validate-only)",
    )
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="Check manifest paths exist; print JSON summary; exit 1 if any missing. No MediaPipe.",
    )
    ap.add_argument("--backend", type=str, default="mediapipe")
    ap.add_argument("--multipass", action="store_true", help="Match KinematicAnalyzer preprocess sweep")
    ap.add_argument(
        "--strict-manifest",
        action="store_true",
        help="Exit 1 if any clip path is missing on disk OR any expect_* column fails on a processed clip",
    )
    args = ap.parse_args()

    if args.validate_only and args.out is not None:
        ap.error("--out is not used with --validate-only (omit --out)")
    if not args.validate_only and args.out is None:
        ap.error("--out is required unless you pass --validate-only")

    base = args.manifest_dir or REPO_ROOT

    from app.pose.gym_manifest import (
        check_manifest_expectations,
        load_gym_manifest,
        summarize_manifest_path_status,
    )

    try:
        jobs = load_gym_manifest(args.manifest, base)
    except ValueError as e:
        print(
            json.dumps(
                {"manifest_parse_error": str(e), "manifest_path_base": str(base.resolve())},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    if args.validate_only:
        stat = summarize_manifest_path_status(jobs)
        stat["manifest_path_base"] = str(base.resolve())
        print(json.dumps(stat, indent=2))
        return 1 if stat["files_missing"] > 0 else 0

    from app.pose.backends import get_pose_backend  # noqa: E402

    backend = get_pose_backend(args.backend)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    strict_violations = 0
    rates: list[float] = []
    usable_n = 0
    ok_n = 0

    missing_files = 0
    ok_without_ffmpeg = 0
    run_provenance_sample: dict | None = None
    with args.out.open("w", encoding="utf-8") as out_f:
        for job in jobs:
            vp: Path = job["video_path"]
            clip_id = job["clip_id"]
            t0 = time.perf_counter()
            if not vp.is_file():
                missing_files += 1
                row = {
                    "clip_id": clip_id,
                    "video_path": str(vp),
                    "backend": backend.name,
                    "tags": job.get("tags"),
                    "exercise_id": job.get("exercise_id"),
                    "ok": False,
                    "error": "file_not_found",
                    "elapsed_ms": 0.0,
                }
                out_f.write(json.dumps(row) + "\n")
                if args.strict_manifest:
                    strict_violations += 1
                continue

            res = backend.run(str(vp), multipass=args.multipass)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            d = res.to_dict()
            d["clip_id"] = clip_id
            d["tags"] = job.get("tags")
            d["exercise_id"] = job.get("exercise_id")
            d["elapsed_ms"] = elapsed_ms

            errs = check_manifest_expectations(
                d, job["expect_pose_usable"], job["expect_min_detection_rate"]
            )
            if errs:
                d["expectation_errors"] = errs
                if args.strict_manifest:
                    strict_violations += 1

            out_f.write(json.dumps(d) + "\n")

            if d.get("ok"):
                ok_n += 1
                rates.append(float(d.get("detection_rate") or 0.0))
                if d.get("pose_usable_heuristic"):
                    usable_n += 1
                if not d.get("ffmpeg_preprocess_applied"):
                    ok_without_ffmpeg += 1
                if run_provenance_sample is None and d.get("provenance"):
                    run_provenance_sample = d["provenance"]

    n = len(jobs)
    summary = {
        "clips_in_manifest": n,
        "files_found_ok": ok_n,
        "manifest_path_base": str(base.resolve()),
        "pose_usable_heuristic_count": usable_n,
        "mean_detection_rate": round(statistics.mean(rates), 4) if rates else None,
        "backend": backend.name,
        "multipass": args.multipass,
        "output": str(args.out),
    }
    if missing_files:
        summary["files_missing_on_disk"] = missing_files
    if ok_n and ok_without_ffmpeg:
        summary["warning"] = (
            f"{ok_without_ffmpeg}/{ok_n} successful clip(s) ran without FFmpeg preprocess "
            "(install ffmpeg and re-run to match production H.264/VFR path; metrics may still be valid)."
        )
    if run_provenance_sample:
        summary["run_provenance_sample"] = run_provenance_sample
    if args.strict_manifest:
        summary["strict_violations"] = strict_violations
    print(json.dumps(summary, indent=2))

    return 1 if args.strict_manifest and strict_violations > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
