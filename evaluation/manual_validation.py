"""
Manual concurrent-validity check for the pose-derived joint angles.

WHY THIS EXISTS
---------------
The existing regression test compares the analyzer against its OWN past output,
so it proves self-consistency, not accuracy. This script measures accuracy the
only honest way available to a solo project with no motion-capture lab: compare
the analyzer's joint angles against a small set of HAND-LABELLED ground-truth
angles, and report the mean absolute error (MAE).

It is intentionally decoupled from the app (no MediaPipe / app imports) so it
always runs, even in a bare environment. You provide a CSV of labels; it prints
per-joint and overall MAE and (optionally) writes a JSON summary.

HOW TO PRODUCE LABELS
---------------------
1. Pick ~5-10 shot frames where the relevant joint is clearly visible (clean
   side view). Tools like ImageJ, Kinovea, or even a printed protractor work.
2. For each frame, measure the true joint angle (e.g. elbow = shoulder-elbow-
   wrist) by hand.
3. Run the analyzer on the same clips and read its reported angle for that frame.
4. Record both in a CSV (see FORMAT below).

CSV FORMAT (header required):
    clip,frame,joint,manual_deg,predicted_deg
    curry_01.mp4,42,elbow,168.0,171.5
    curry_01.mp4,42,knee,150.0,144.0
    ...

USAGE
-----
    python evaluation/manual_validation.py evaluation/validation_labels.csv
    python evaluation/manual_validation.py labels.csv --json results.json

Report the resulting MAE in docs/VALIDATION.md. One honest number ("elbow MAE =
X deg over N hand-labelled frames") is worth more than a confident fake one.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict


def _abs_err(a: float, b: float) -> float:
    return abs(a - b)


def load_rows(path: str) -> list[dict]:
    required = {"clip", "frame", "joint", "manual_deg", "predicted_deg"}
    rows: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"[error] CSV missing required columns: {sorted(missing)}")
        for i, r in enumerate(reader, start=2):  # line 2 = first data row
            try:
                rows.append({
                    "clip": r["clip"].strip(),
                    "frame": r["frame"].strip(),
                    "joint": r["joint"].strip().lower(),
                    "manual_deg": float(r["manual_deg"]),
                    "predicted_deg": float(r["predicted_deg"]),
                })
            except (ValueError, KeyError) as e:
                sys.exit(f"[error] bad numeric value on CSV line {i}: {e}")
    if not rows:
        sys.exit("[error] no data rows found in CSV.")
    return rows


def compute_mae(rows: list[dict]) -> dict:
    per_joint_errs: dict[str, list[float]] = defaultdict(list)
    all_errs: list[float] = []
    for r in rows:
        e = _abs_err(r["manual_deg"], r["predicted_deg"])
        per_joint_errs[r["joint"]].append(e)
        all_errs.append(e)

    def _stats(errs: list[float]) -> dict:
        n = len(errs)
        mae = sum(errs) / n
        worst = max(errs)
        return {"n": n, "mae_deg": round(mae, 2), "max_abs_err_deg": round(worst, 2)}

    return {
        "overall": _stats(all_errs),
        "per_joint": {j: _stats(errs) for j, errs in sorted(per_joint_errs.items())},
    }


def print_report(result: dict) -> None:
    print("=" * 60)
    print("  MANUAL CONCURRENT-VALIDITY REPORT (pose angle vs hand-label)")
    print("=" * 60)
    print(f"  {'joint':<14}{'N':>4}{'MAE (deg)':>12}{'max |err|':>12}")
    print("  " + "-" * 42)
    for joint, s in result["per_joint"].items():
        print(f"  {joint:<14}{s['n']:>4}{s['mae_deg']:>12}{s['max_abs_err_deg']:>12}")
    o = result["overall"]
    print("  " + "-" * 42)
    print(f"  {'OVERALL':<14}{o['n']:>4}{o['mae_deg']:>12}{o['max_abs_err_deg']:>12}")
    print()
    print("  Honest caveats: small N; single-camera side view; knee/elbow are the")
    print("  most trustworthy joints, hip rotation (depth axis) is not validated here.")
    print("  Report this number in docs/VALIDATION.md.")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute joint-angle MAE vs hand-labelled ground truth.")
    ap.add_argument("csv", help="path to labels CSV (clip,frame,joint,manual_deg,predicted_deg)")
    ap.add_argument("--json", help="optional path to write a JSON summary")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    result = compute_mae(rows)
    print_report(result)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
