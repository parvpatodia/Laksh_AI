#!/usr/bin/env python3
"""
Static readiness for gym pose eval (no video, no inference).

Prints JSON to stdout (``report_schema_version`` 1.2.0: dep blocks + ``pose_landmarker_task`` SHA-256).
Use before investing time in long A/B runs.

  python3 scripts/check_pose_eval_readiness.py
  python3 scripts/check_pose_eval_readiness.py --manifest evaluation/gym_manifest.csv
  python3 scripts/check_pose_eval_readiness.py --strict   # exit 1 if mediapipe_gym_eval_minimal is false

Makefile: ``make check-pose-readiness`` / ``make check-pose-readiness-strict`` (honours ``PYTHON``).

See app.pose.eval_readiness.collect_eval_readiness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Pose eval readiness (static checks only).")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional gym CSV to validate (paths resolved from repo root)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if mediapipe_gym_eval_minimal is false",
    )
    args = ap.parse_args()

    from app.pose.eval_readiness import collect_eval_readiness

    report = collect_eval_readiness(
        manifest_path=args.manifest,
        repo_root=REPO_ROOT,
    )
    print(json.dumps(report, indent=2))

    if args.strict and not report.get("mediapipe_gym_eval_minimal"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
