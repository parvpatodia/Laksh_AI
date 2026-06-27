#!/usr/bin/env python3
"""
Check that train/val/test splits in a manifest CSV do NOT leak subjects.

Rationale (roadmap Phase B — "Dataset hygiene"): random frame splits inflate
scores when the same person/background repeats across splits. The only
defensible split for a small clip corpus is subject- or session-level
separation. This script enforces that invariant.

Contract
--------
- The manifest must declare two columns: `subject_id` and `split`.
- `split` values are restricted to {train, val, test}.
- Each `subject_id` must appear in exactly ONE split.
- Optional `session_id` column: if present, sessions must also not cross
  splits (stricter than subject check; useful when a single subject returns
  across data-collection sessions).

Exit codes
----------
- 0: clean. No leakage detected. A `split_coverage` summary is printed.
- 1: leakage detected or schema invalid. Offending rows are printed.
- 2: columns not present. This is a WARNING not an error — we treat the
  manifest as needing a schema upgrade rather than a bug. Use `--strict`
  to promote to exit 1.

Usage
-----
  python scripts/subject_split_check.py --manifest evaluation/gym_manifest.csv
  python scripts/subject_split_check.py --manifest evaluation/gym_manifest.csv --strict
  python scripts/subject_split_check.py --manifest evaluation/gym_manifest.csv --sessions
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SPLITS = {"train", "val", "test"}


def _load_manifest(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    return header, rows


def check_splits(
    rows: list[dict],
    check_sessions: bool = False,
) -> dict:
    """Return a leakage report. `leaked_subjects` is the key signal."""
    subjects_to_splits: dict[str, set[str]] = defaultdict(set)
    sessions_to_splits: dict[str, set[str]] = defaultdict(set)
    bad_split_values: list[dict] = []

    for row in rows:
        split = (row.get("split") or "").strip()
        if split and split not in ALLOWED_SPLITS:
            bad_split_values.append(
                {"clip_id": row.get("clip_id"), "split": split}
            )
            continue
        subj = (row.get("subject_id") or "").strip()
        if subj and split:
            subjects_to_splits[subj].add(split)
        if check_sessions:
            sess = (row.get("session_id") or "").strip()
            if sess and split:
                sessions_to_splits[sess].add(split)

    leaked_subjects = {
        subj: sorted(splits)
        for subj, splits in subjects_to_splits.items()
        if len(splits) > 1
    }
    leaked_sessions = {
        sess: sorted(splits)
        for sess, splits in sessions_to_splits.items()
        if len(splits) > 1
    } if check_sessions else {}

    coverage: dict[str, int] = {s: 0 for s in ALLOWED_SPLITS}
    for row in rows:
        s = (row.get("split") or "").strip()
        if s in coverage:
            coverage[s] += 1

    return {
        "n_rows": len(rows),
        "split_coverage": coverage,
        "bad_split_values": bad_split_values,
        "leaked_subjects": leaked_subjects,
        "leaked_sessions": leaked_sessions,
        "n_subjects": len(subjects_to_splits),
        "n_sessions": len(sessions_to_splits),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Check manifest for subject/session split leakage.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Promote missing-schema warning to hard failure (exit 1).",
    )
    ap.add_argument(
        "--sessions",
        action="store_true",
        help="Also require session_id uniqueness across splits.",
    )
    args = ap.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    if not manifest.is_file():
        print(f"[error] manifest not found: {manifest}", file=sys.stderr)
        return 1

    header, rows = _load_manifest(manifest)
    has_subject = "subject_id" in header
    has_split = "split" in header

    if not (has_subject and has_split):
        missing = [c for c, present in [("subject_id", has_subject), ("split", has_split)] if not present]
        payload = {
            "manifest": str(manifest.resolve()),
            "status": "schema_upgrade_needed",
            "missing_columns": missing,
            "recommendation": (
                "Add `subject_id` (stable id per athlete/session recorder) and "
                "`split` ({train,val,test}) columns. Fit thresholds on val; "
                "touch test only at release."
            ),
        }
        print(json.dumps(payload, indent=2))
        return 1 if args.strict else 2

    report = check_splits(rows, check_sessions=args.sessions)
    report["manifest"] = str(manifest.resolve())
    report["status"] = (
        "clean"
        if not report["leaked_subjects"]
        and not report["leaked_sessions"]
        and not report["bad_split_values"]
        else "leakage_detected"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "clean" else 1


if __name__ == "__main__":
    raise SystemExit(main())
