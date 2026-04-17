#!/usr/bin/env python3
"""Pose canonical-vs-legacy parity report (ADR 0002 Phase C promotion gate).

Reduces a benchmark JSONL produced with ``LAKSH_USE_CANONICAL_JOINTS=1`` on a
basketball manifest into per-angle 2D-delta statistics between the canonical
joint path and the legacy index path at key frames (dip knee / release elbow).

Promotion rule (from ``docs/adr/0002-p3-canonical-in-kinematic-analyzer.md``):
the default may flip on once ``p90_abs_delta_{knee,elbow}_deg`` sits at or
below ``--threshold-deg`` (default 2.0) on a frozen basketball manifest.

Input row shape (tolerant): the canonical probe dict is read from the first
of these paths that exists on a JSONL row::

    row["telemetry"]["canonical_joint_path"]        # full analysis dump
    row["summary"]["canonical_joint_path"]          # scripts/benchmark_pipeline.py
    row["canonical_joint_path"]                     # flattened rows

Expected probe keys (see ``app.physics_engine._canonical_joint_path_telemetry``):
``delta_knee_vs_legacy_2d_deg``, ``delta_elbow_vs_legacy_2d_deg``, optional
``error`` when a key frame or joint is missing.

Exit codes: ``0`` gate passes, ``1`` gate fails, ``2`` no probe rows present
(likely flag off or wrong JSONL).

Usage::

    python scripts/pose_parity_report.py --jsonl evaluation/results.jsonl
    python scripts/pose_parity_report.py --jsonl run.jsonl --threshold-deg 1.5
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

PARITY_SCHEMA_VERSION = "1.0.0"
DEFAULT_THRESHOLD_DEG = 2.0
_PROBE_PATHS: tuple[tuple[str, ...], ...] = (
    ("telemetry", "canonical_joint_path"),
    ("summary", "canonical_joint_path"),
    ("canonical_joint_path",),
)
_KNOWN_ERRORS: tuple[str, ...] = (
    "key_frame_out_of_range",
    "missing_canonical_pose_at_key_frame",
    "missing_joint_in_canonical_frame",
)


def extract_probe(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the canonical_joint_path dict at any known location, else None."""
    for path in _PROBE_PATHS:
        cur: Any = row
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and isinstance(cur, dict):
            return cur
    return None


def _finite_abs(x: Any) -> float | None:
    if not isinstance(x, (int, float)):
        return None
    xf = float(x)
    if math.isnan(xf) or math.isinf(xf):
        return None
    return abs(xf)


def _quantile(xs: list[float], q: float) -> float | None:
    """Linear-interpolated quantile on a sorted copy; ``None`` when xs is empty."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    return {
        "n": len(xs),
        "mean": round(sum(xs) / len(xs), 4),
        "median": round(_quantile(xs, 0.5) or 0.0, 4),
        "p90": round(_quantile(xs, 0.90) or 0.0, 4),
        "p99": round(_quantile(xs, 0.99) or 0.0, 4),
        "max": round(max(xs), 4),
    }


def summarise(
    rows: list[dict[str, Any]],
    threshold_deg: float = DEFAULT_THRESHOLD_DEG,
) -> dict[str, Any]:
    """Aggregate per-row canonical_joint_path probes into a parity report dict."""
    knee_abs: list[float] = []
    elbow_abs: list[float] = []
    error_counts: dict[str, int] = {k: 0 for k in _KNOWN_ERRORS}
    error_counts["other"] = 0
    n_present = 0
    n_absent = 0

    for row in rows:
        probe = extract_probe(row)
        if probe is None:
            n_absent += 1
            continue
        n_present += 1
        err = probe.get("error")
        if isinstance(err, str) and err:
            if err in error_counts:
                error_counts[err] += 1
            else:
                error_counts["other"] += 1
            continue
        ka = _finite_abs(probe.get("delta_knee_vs_legacy_2d_deg"))
        ea = _finite_abs(probe.get("delta_elbow_vs_legacy_2d_deg"))
        if ka is not None:
            knee_abs.append(ka)
        if ea is not None:
            elbow_abs.append(ea)

    knee_stats = _stats(knee_abs)
    elbow_stats = _stats(elbow_abs)
    knee_p90 = knee_stats["p90"]
    elbow_p90 = elbow_stats["p90"]
    knee_pass = knee_p90 is not None and knee_p90 <= threshold_deg
    elbow_pass = elbow_p90 is not None and elbow_p90 <= threshold_deg

    return {
        "parity_schema_version": PARITY_SCHEMA_VERSION,
        "n_rows": len(rows),
        "n_probe_present": n_present,
        "n_probe_absent": n_absent,
        "error_counts": error_counts,
        "knee_delta_abs_deg": knee_stats,
        "elbow_delta_abs_deg": elbow_stats,
        "threshold_deg": float(threshold_deg),
        "knee_p90_pass": bool(knee_pass),
        "elbow_p90_pass": bool(elbow_pass),
        "promotion_gate_pass": bool(knee_pass and elbow_pass),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected object, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def _exit_code(report: dict[str, Any]) -> int:
    if report["n_probe_present"] == 0:
        return 2
    return 0 if report["promotion_gate_pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, required=True, help="Benchmark JSONL with canonical probe")
    ap.add_argument(
        "--threshold-deg",
        type=float,
        default=DEFAULT_THRESHOLD_DEG,
        help=f"P90 abs-delta gate per angle in degrees (default {DEFAULT_THRESHOLD_DEG})",
    )
    args = ap.parse_args()
    try:
        rows = _load_jsonl(args.jsonl)
    except (OSError, ValueError) as e:
        print(json.dumps({"error": str(e), "jsonl": str(args.jsonl)}, indent=2))
        return 2
    report = summarise(rows, threshold_deg=args.threshold_deg)
    report["jsonl_path"] = str(args.jsonl.resolve())
    print(json.dumps(report, indent=2))
    return _exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
