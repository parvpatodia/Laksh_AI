#!/usr/bin/env python3
"""
Diff two benchmark_pipeline.py JSONL outputs (e.g. mediapipe vs future backbone).

Usage:
  python scripts/compare_benchmark_results.py evaluation/run_a.jsonl evaluation/run_b.jsonl -o evaluation/compare.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            cid = o.get("clip_id")
            if cid:
                by_id[cid] = o
    return by_id


def measured_count(row: dict) -> int | None:
    s = row.get("summary") or {}
    mc = s.get("metric_source_counts") or {}
    return mc.get("measured") if row.get("ok") else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two benchmark JSONL files.")
    ap.add_argument("jsonl_a", type=Path)
    ap.add_argument("jsonl_b", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="Write CSV summary")
    ap.add_argument(
        "--label-a",
        default="run_a",
        help="Column prefix for first file (default: run_a)",
    )
    ap.add_argument(
        "--label-b",
        default="run_b",
        help="Column prefix for second file (default: run_b)",
    )
    args = ap.parse_args()

    ma = load_jsonl(args.jsonl_a)
    mb = load_jsonl(args.jsonl_b)
    all_ids = sorted(set(ma) | set(mb))
    if not all_ids:
        print("No clip_id entries in either file.", file=sys.stderr)
        return 1

    rows_out = []
    for cid in all_ids:
        ra, rb = ma.get(cid), mb.get(cid)
        na = measured_count(ra) if ra else None
        nb = measured_count(rb) if rb else None
        delta = (nb - na) if na is not None and nb is not None else ""
        rows_out.append(
            {
                "clip_id": cid,
                f"{args.label_a}_ok": ra.get("ok") if ra else False,
                f"{args.label_a}_measured": na if na is not None else "",
                f"{args.label_a}_mode": (ra.get("summary") or {}).get("analysis_mode") if ra else "",
                f"{args.label_b}_ok": rb.get("ok") if rb else False,
                f"{args.label_b}_measured": nb if nb is not None else "",
                f"{args.label_b}_mode": (rb.get("summary") or {}).get("analysis_mode") if rb else "",
                "delta_measured": delta,
            }
        )

    fieldnames = list(rows_out[0].keys())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
