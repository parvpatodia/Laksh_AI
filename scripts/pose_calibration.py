#!/usr/bin/env python3
"""
Reliability-diagram bins for a pose_baseline JSONL run.

Rationale (roadmap Phase B — "Calibration / user trust"): separate raw model
confidence from user-facing "ready / degraded / unknown". Plot predicted
confidence vs empirical error on L1. We emit bin statistics as JSON — PNG
rendering is intentionally skipped (matplotlib is not a runtime dependency).
A downstream notebook or plot tool can consume the JSON.

Definitions
-----------
- Predicted confidence: `visibility_core_when_detected` (model's own signal
  about whether it saw the core keypoints well).
- Empirical "success" label: `pose_usable_heuristic` — the heuristic the
  product uses to decide whether to show metrics. This is not ground truth;
  it is a product-aligned proxy until L2 labels exist. See
  docs/product-grade_laksh_roadmap_05e7df02.plan.md Phase B.
- Expected Calibration Error (ECE): weighted sum over bins of
  |predicted_mean - empirical_rate|, weighted by bin population.

A well-calibrated classifier has ECE near 0 and each bin's predicted_mean
close to its empirical_rate on the diagonal.

Usage
-----
  python scripts/pose_calibration.py --jsonl evaluation/pose_baseline.jsonl
  python scripts/pose_calibration.py --jsonl evaluation/pose_baseline.jsonl --bins 5
  python scripts/pose_calibration.py --jsonl evaluation/pose_baseline.jsonl --out evaluation/calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_SCHEMA_VERSION = "1.0.0"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def reliability_bins(
    pairs: list[tuple[float, bool]],
    n_bins: int = 10,
) -> list[dict]:
    """
    pairs: (predicted_confidence_in_[0,1], empirical_success_bool).
    Returns one dict per bin with [lo, hi) edges + counts + means.
    The final bin is inclusive on the right so 1.0 lands there.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    buckets: list[dict] = []
    step = 1.0 / n_bins
    for i in range(n_bins):
        lo = i * step
        hi = (i + 1) * step
        inclusive_right = i == n_bins - 1
        sel = [
            (p, s)
            for (p, s) in pairs
            if lo <= p < hi or (inclusive_right and p == hi)
        ]
        n = len(sel)
        predicted_mean = sum(p for p, _ in sel) / n if n else None
        empirical_rate = (sum(1 for _, s in sel if s) / n) if n else None
        buckets.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "n": n,
                "predicted_mean": predicted_mean,
                "empirical_rate": empirical_rate,
            }
        )
    return buckets


def expected_calibration_error(bins: list[dict], total_n: int) -> float | None:
    """Weighted mean of |predicted_mean - empirical_rate| over non-empty bins."""
    if total_n == 0:
        return None
    ece = 0.0
    for b in bins:
        if b["n"] == 0:
            continue
        gap = abs((b["predicted_mean"] or 0.0) - (b["empirical_rate"] or 0.0))
        ece += (b["n"] / total_n) * gap
    return ece


def analyse(rows: list[dict], n_bins: int = 10) -> dict:
    ok_rows = [r for r in rows if r.get("ok")]
    pairs: list[tuple[float, bool]] = []
    for r in ok_rows:
        pred = r.get("visibility_core_when_detected")
        label = r.get("pose_usable_heuristic")
        if pred is None or label is None:
            continue
        # Clamp defensively — upstream may emit tiny floating-point excursions.
        p = max(0.0, min(1.0, float(pred)))
        pairs.append((p, bool(label)))

    buckets = reliability_bins(pairs, n_bins=n_bins)
    ece = expected_calibration_error(buckets, total_n=len(pairs))
    return {
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "n_pairs": len(pairs),
        "n_bins": n_bins,
        "predicted_field": "visibility_core_when_detected",
        "empirical_field": "pose_usable_heuristic",
        "expected_calibration_error": ece,
        "bins": buckets,
        "note": (
            "pose_usable_heuristic is a product-aligned proxy, not ground "
            "truth. Replace with human labels once L2 subset exists "
            "(roadmap Phase B)."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Reliability-diagram bins for pose JSONL.")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    p = args.jsonl if args.jsonl.is_absolute() else REPO_ROOT / args.jsonl
    if not p.is_file():
        print(f"[error] JSONL not found: {p}", file=sys.stderr)
        return 1

    rows = _load_jsonl(p)
    report = analyse(rows, n_bins=args.bins)
    report["source_jsonl"] = str(p.resolve())

    rendered = json.dumps(report, indent=2)
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
