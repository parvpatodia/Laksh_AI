#!/usr/bin/env python3
"""
Temporal sanity report for a pose_baseline JSONL run.

Rationale (roadmap Phase B — "Temporal pose quality"): accuracy alone is not
enough; we must track jitter and implausible jumps. This tool summarises the
frame-to-frame signal already captured in the JSONL
(`hip_mid_displacement_median_norm`) across the clip population and flags
outliers that merit review.

Limitations
-----------
- The current aggregate JSONL stores one metric per clip, not per frame. A
  population-level distribution is all we can produce from it. Per-frame
  jitter (variance of each landmark across frames) requires the per-frame
  pose trace — not emitted by `scripts/eval_pose_baseline.py` today. When
  that trace is added, extend this tool with a `--per-frame-jsonl` mode.
- Outlier flagging uses Tukey fences (Q3 + k*IQR) with k=1.5 by default.
  Hard-coded absolute thresholds would hide drift as backbones change; the
  fence adapts to the run.

Usage
-----
  python scripts/pose_jitter.py --jsonl evaluation/pose_baseline.jsonl
  python scripts/pose_jitter.py --jsonl evaluation/pose_baseline.jsonl \\
      --out evaluation/pose_jitter_report.json
  python scripts/pose_jitter.py --jsonl a.jsonl --jsonl b.jsonl --k 2.0
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JITTER_SCHEMA_VERSION = "1.0.0"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    idx = q * (len(xs) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(xs) - 1)
    frac = idx - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def tukey_upper_fence(xs: list[float], k: float = 1.5) -> float | None:
    """Upper fence = Q3 + k*IQR. Returns None on empty input."""
    if len(xs) < 2:
        return None
    q1 = _quantile(xs, 0.25)
    q3 = _quantile(xs, 0.75)
    if q1 is None or q3 is None:
        return None
    iqr = q3 - q1
    return q3 + k * iqr


def summarise(rows: list[dict], k: float = 1.5) -> dict:
    """Aggregate temporal signals across a JSONL run."""
    ok_rows = [r for r in rows if r.get("ok")]
    jitter_proxy = [
        float(r["hip_mid_displacement_median_norm"])
        for r in ok_rows
        if r.get("hip_mid_displacement_median_norm") is not None
    ]
    det_rates = [
        float(r["detection_rate"])
        for r in ok_rows
        if r.get("detection_rate") is not None
    ]
    fence = tukey_upper_fence(jitter_proxy, k=k)

    outliers: list[dict] = []
    if fence is not None:
        for r in ok_rows:
            val = r.get("hip_mid_displacement_median_norm")
            if val is not None and float(val) > fence:
                outliers.append(
                    {
                        "video_path": r.get("video_path"),
                        "hip_mid_displacement_median_norm": float(val),
                        "detection_rate": r.get("detection_rate"),
                        "pose_usable_heuristic": r.get("pose_usable_heuristic"),
                        "reason_codes": r.get("reason_codes") or [],
                    }
                )

    return {
        "jitter_schema_version": JITTER_SCHEMA_VERSION,
        "k_fence": k,
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "hip_mid_displacement_median_norm": {
            "n": len(jitter_proxy),
            "mean": statistics.fmean(jitter_proxy) if jitter_proxy else None,
            "median": _quantile(jitter_proxy, 0.5),
            "p90": _quantile(jitter_proxy, 0.9),
            "p99": _quantile(jitter_proxy, 0.99),
            "upper_fence": fence,
        },
        "detection_rate": {
            "n": len(det_rates),
            "mean": statistics.fmean(det_rates) if det_rates else None,
            "p10": _quantile(det_rates, 0.1),
            "median": _quantile(det_rates, 0.5),
        },
        "outliers_above_fence": outliers,
        "note": (
            "hip_mid_displacement_median_norm is a per-clip jitter PROXY. "
            "Full per-landmark temporal variance needs per-frame pose traces "
            "(not emitted by eval_pose_baseline.py today)."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Temporal sanity / jitter summary for pose JSONL.")
    ap.add_argument(
        "--jsonl",
        action="append",
        required=True,
        type=Path,
        help="Pose baseline JSONL (repeatable to summarise multiple backends).",
    )
    ap.add_argument(
        "--k",
        type=float,
        default=1.5,
        help="Tukey fence multiplier (default 1.5; use 3.0 for extreme-only).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path. If omitted, prints to stdout.",
    )
    args = ap.parse_args()

    reports: list[dict] = []
    for raw in args.jsonl:
        p = raw if raw.is_absolute() else REPO_ROOT / raw
        if not p.is_file():
            print(f"[error] JSONL not found: {p}", file=sys.stderr)
            return 1
        rows = _load_jsonl(p)
        report = summarise(rows, k=args.k)
        report["source_jsonl"] = str(p.resolve())
        reports.append(report)

    payload = {"runs": reports}
    rendered = json.dumps(payload, indent=2)
    if args.out is None:
        print(rendered)
    else:
        out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
