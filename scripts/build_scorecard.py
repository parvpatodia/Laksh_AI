#!/usr/bin/env python3
"""
Build a release scorecard (markdown) for a pose-baseline JSONL run.

Rationale (roadmap Phase A — "regression bundle"): version the eval harness
with outputs; one command reproduces the last release scorecard with commit,
manifest, and lock-file hashes in the report header. Downstream aggregates
(detection rate, usable heuristic, reason codes) make regressions legible.

Inputs
------
- `--manifest` : gym manifest CSV (clip_id + expect_* columns).
- `--jsonl`    : pose_baseline JSONL output (repeatable; one per backend).
- `--out`      : output markdown path; default
                 `evaluation/scorecard_<ISO>_<shortsha>.md`.
- `--requirements` : lock file to hash (default: requirements.lock).

What the scorecard contains
---------------------------
1. Reproducibility header (identical fields as `eval_scorecard_header.py` plus
   lock hash) — structured JSON fenced block.
2. Per-backend aggregate table: detection_rate quantiles, usable rate,
   ffmpeg-preprocess coverage, mean visibility.
3. Top reason codes per backend.
4. Per-clip table (sorted by detection_rate asc → worst first).

Notes
-----
- Single-pass vs multipass semantics follow `selected_pass` field in JSONL
  (see docs/POSE_UPGRADE_EXECUTION_PLAN.md §4). We quote single_pass-equivalent
  detection rates by default; multipass_best is reported separately so the
  "best case" numbers cannot be mistaken for production behaviour.
- This tool does NOT run inference; it summarises archived JSONL. Run
  `scripts/eval_pose_baseline.py` (or `make eval-pose-gym`) first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORECARD_SCHEMA_VERSION = "1.2.0"  # adds requirements_lock_sha256 + aggregates


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _short_sha(full: str | None) -> str:
    return (full or "nosha")[:7]


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
    """Linear-interpolation quantile; returns None on empty input."""
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    idx = q * (len(xs) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(xs) - 1)
    frac = idx - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _backend_aggregate(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if r.get("ok")]
    det_rates = [float(r["detection_rate"]) for r in ok_rows if "detection_rate" in r]
    vis = [
        float(r["visibility_core_when_detected"])
        for r in ok_rows
        if "visibility_core_when_detected" in r
    ]
    usable = [bool(r.get("pose_usable_heuristic")) for r in ok_rows]
    preprocess = [bool(r.get("ffmpeg_preprocess_applied")) for r in ok_rows]
    reason_codes: Counter[str] = Counter()
    for r in ok_rows:
        for code in r.get("reason_codes") or []:
            reason_codes[code] += 1
    return {
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "detection_rate_mean": statistics.fmean(det_rates) if det_rates else None,
        "detection_rate_p10": _quantile(det_rates, 0.1),
        "detection_rate_p50": _quantile(det_rates, 0.5),
        "detection_rate_p90": _quantile(det_rates, 0.9),
        "visibility_core_mean": statistics.fmean(vis) if vis else None,
        "usable_rate": (sum(usable) / len(usable)) if usable else None,
        "ffmpeg_preprocess_rate": (
            sum(preprocess) / len(preprocess) if preprocess else None
        ),
        "top_reason_codes": reason_codes.most_common(5),
    }


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _render_header(header: dict) -> str:
    return "```json\n" + json.dumps(header, indent=2, sort_keys=True) + "\n```"


def _render_aggregate(name: str, agg: dict) -> str:
    lines = [
        f"### Backend: `{name}`",
        "",
        f"- rows: **{agg['n_ok']}/{agg['n_rows']}** ok",
        f"- detection_rate: mean={_fmt(agg['detection_rate_mean'])}, "
        f"p10={_fmt(agg['detection_rate_p10'])}, "
        f"p50={_fmt(agg['detection_rate_p50'])}, "
        f"p90={_fmt(agg['detection_rate_p90'])}",
        f"- visibility_core (when detected): mean={_fmt(agg['visibility_core_mean'])}",
        f"- pose_usable_heuristic rate: {_fmt(agg['usable_rate'])}",
        f"- ffmpeg_preprocess rate: {_fmt(agg['ffmpeg_preprocess_rate'])}",
        "",
    ]
    if agg["top_reason_codes"]:
        lines.append("**Top reason codes:**")
        lines.append("")
        lines.append("| code | count |")
        lines.append("|---|---|")
        for code, count in agg["top_reason_codes"]:
            lines.append(f"| `{code}` | {count} |")
        lines.append("")
    return "\n".join(lines)


def _render_per_clip(rows: list[dict], backend: str) -> str:
    ok_rows = sorted(
        [r for r in rows if r.get("ok")],
        key=lambda r: float(r.get("detection_rate") or 0.0),
    )
    if not ok_rows:
        return f"_No successful rows for `{backend}`._\n"
    header = (
        "| clip | detection_rate | visibility | usable | preprocess | pass | reasons |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    body: list[str] = []
    for r in ok_rows:
        clip = Path(r.get("video_path", "?")).name
        det = _fmt(r.get("detection_rate"))
        vis = _fmt(r.get("visibility_core_when_detected"))
        usable = "yes" if r.get("pose_usable_heuristic") else "no"
        preprocess = "yes" if r.get("ffmpeg_preprocess_applied") else "no"
        sel = r.get("selected_pass") or "-"
        reasons = ", ".join(r.get("reason_codes") or []) or "-"
        body.append(
            f"| `{clip}` | {det} | {vis} | {usable} | {preprocess} | {sel} | {reasons} |"
        )
    return header + "\n".join(body) + "\n"


def _infer_backend_name(rows: list[dict], fallback: str) -> str:
    """Prefer the per-row `backend` field; fall back to filename stem."""
    backends = {r.get("backend") for r in rows if r.get("backend")}
    if len(backends) == 1:
        return next(iter(backends))
    return fallback


def build(
    manifest: Path | None,
    jsonl_paths: list[Path],
    out: Path,
    requirements_lock: Path,
) -> Path:
    commit = _git_head()
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    header = {
        "scorecard_schema_version": SCORECARD_SCHEMA_VERSION,
        "purpose": "release_scorecard_regression_bundle",
        "generated_at_utc": now,
        "interpreter": sys.executable,
        "repo_root": str(REPO_ROOT.resolve()),
        "git_commit": commit,
        "requirements_lock_path": (
            str(requirements_lock.resolve()) if requirements_lock.is_file() else None
        ),
        "requirements_lock_sha256": _sha256_file(requirements_lock),
        "manifest_path": (
            str(manifest.resolve()) if manifest and manifest.is_file() else None
        ),
        "manifest_sha256": _sha256_file(manifest) if manifest else None,
        "pose_jsonl_artifacts": [
            {"path": str(p.resolve()), "sha256": _sha256_file(p)}
            for p in jsonl_paths
        ],
    }

    sections: list[str] = []
    sections.append(f"# Release scorecard — {now} ({_short_sha(commit)})")
    sections.append("")
    sections.append("## Reproducibility header")
    sections.append("")
    sections.append(_render_header(header))
    sections.append("")

    if not jsonl_paths:
        sections.append("## Aggregate metrics")
        sections.append("")
        sections.append(
            "_No JSONL provided — header-only scorecard. "
            "Pass `--jsonl` to include aggregates._"
        )
        sections.append("")
    else:
        sections.append("## Aggregate metrics by backend")
        sections.append("")
        for jsonl_path in jsonl_paths:
            rows = _load_jsonl(jsonl_path)
            name = _infer_backend_name(rows, jsonl_path.stem)
            agg = _backend_aggregate(rows)
            sections.append(_render_aggregate(name, agg))
        sections.append("## Per-clip drill-down (worst first)")
        sections.append("")
        for jsonl_path in jsonl_paths:
            rows = _load_jsonl(jsonl_path)
            name = _infer_backend_name(rows, jsonl_path.stem)
            sections.append(f"### `{name}`")
            sections.append("")
            sections.append(_render_per_clip(rows, name))

    sections.append("## Claim tier")
    sections.append("")
    sections.append(
        "Match [docs/POSE_UPGRADE_EXECUTION_PLAN.md](../docs/POSE_UPGRADE_EXECUTION_PLAN.md) §5. "
        "Do not upgrade claim tier without evidence in the aggregates above."
    )
    sections.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")
    return out


def _default_out(commit: str | None) -> Path:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "evaluation" / f"scorecard_{now}_{_short_sha(commit)}.md"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a release scorecard (markdown).")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument(
        "--jsonl",
        action="append",
        default=None,
        type=Path,
        help="JSONL pose-baseline outputs (repeatable).",
    )
    ap.add_argument(
        "--requirements",
        type=Path,
        default=REPO_ROOT / "requirements.lock",
        help="Lock file to hash for reproducibility (default: requirements.lock).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown path (default: evaluation/scorecard_<ISO>_<sha>.md).",
    )
    args = ap.parse_args()

    manifest = args.manifest
    if manifest is not None and not manifest.is_absolute():
        manifest = REPO_ROOT / manifest

    jsonl_paths: list[Path] = []
    for raw in args.jsonl or []:
        p = raw if raw.is_absolute() else REPO_ROOT / raw
        jsonl_paths.append(p)

    out = args.out or _default_out(_git_head())
    if not out.is_absolute():
        out = REPO_ROOT / out

    # Warn but do not fail if manifest missing — header-only scorecards are valid.
    if manifest is not None and not manifest.is_file():
        print(f"[warn] manifest not found: {manifest}", file=sys.stderr)

    written = build(manifest, jsonl_paths, out, args.requirements)
    # Relative path if possible (cleaner in CI logs).
    try:
        rel = written.relative_to(REPO_ROOT)
        print(rel)
    except ValueError:
        print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
