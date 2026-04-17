#!/usr/bin/env python3
"""
Compare two pose baseline JSONL files (P1b — same manifest, two backends).

L0 interpretation only: compares pipeline outputs per clip_id, not mocap ground truth.
See app.pose.pose_baseline_compare and docs/POSE_UPGRADE_EXECUTION_PLAN.md §5.

P2: summary includes ``p2_l0`` (``multiple_people_detected`` in ``reason_codes``); per-clip rows
include ``haar_detection_attempts_*`` when ``provenance.person_isolation`` is present.

Usage:
  python scripts/compare_pose_baseline_jsonl.py \\
      --a evaluation/pose_mediapipe.jsonl --b evaluation/pose_rtmpose.jsonl

  python scripts/compare_pose_baseline_jsonl.py -a a.jsonl -b b.jsonl --per-clip-out evaluation/pose_ab_per_clip.jsonl
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
    ap = argparse.ArgumentParser(description="Compare two pose_baseline JSONL runs (L0 A/B).")
    ap.add_argument("--a", type=Path, required=True, help="First JSONL (e.g. MediaPipe run)")
    ap.add_argument("--b", type=Path, required=True, help="Second JSONL (e.g. RTMPose run)")
    ap.add_argument("--label-a", type=str, default=None, help="Label for summary (default: file stem)")
    ap.add_argument("--label-b", type=str, default=None)
    ap.add_argument(
        "--per-clip-out",
        type=Path,
        default=None,
        help="Optional JSONL path with one row per clip_id (intersection only)",
    )
    args = ap.parse_args()

    from app.pose.pose_baseline_compare import (
        compare_pose_baseline_rows,
        load_pose_baseline_rows,
        per_clip_diff_rows,
    )

    la = args.label_a or args.a.stem
    lb = args.label_b or args.b.stem

    try:
        ra = load_pose_baseline_rows(args.a)
        rb = load_pose_baseline_rows(args.b)
    except (OSError, ValueError) as e:
        print(json.dumps({"error": str(e), "path_a": str(args.a), "path_b": str(args.b)}, indent=2))
        return 2

    summary = compare_pose_baseline_rows(ra, rb, label_a=la, label_b=lb)
    summary["path_a"] = str(args.a.resolve())
    summary["path_b"] = str(args.b.resolve())
    print(json.dumps(summary, indent=2))

    if args.per_clip_out:
        args.per_clip_out.parent.mkdir(parents=True, exist_ok=True)
        rows = per_clip_diff_rows(ra, rb, label_a=la, label_b=lb)
        with args.per_clip_out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
