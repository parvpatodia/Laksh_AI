#!/usr/bin/env python3
"""
Emit a reproducible header for offline eval reports (scorecard discipline).

Prints JSON with git commit (if available), SHA-256 of key files, and interpreter.
Use when archiving pose baseline JSONL or benchmark results so claims stay tied
to exact inputs.

  python3 scripts/eval_scorecard_header.py
  python3 scripts/eval_scorecard_header.py --manifest evaluation/gym_manifest.csv
  python3 scripts/eval_scorecard_header.py --manifest evaluation/gym_manifest.csv \\
      --jsonl evaluation/gym_manifest_pose_full.jsonl --jsonl evaluation/gym_manifest_pose_haar_mil_v1.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval scorecard header (hashes + git).")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path (repo-relative or absolute)",
    )
    ap.add_argument(
        "--requirements",
        type=Path,
        default=REPO_ROOT / "requirements.txt",
        help="Requirements file to hash (default: requirements.txt)",
    )
    ap.add_argument(
        "--jsonl",
        action="append",
        default=None,
        metavar="PATH",
        help="Optional JSONL artifact(s) to hash (repeatable; e.g. pose baseline outputs for P1b/P2 archives)",
    )
    args = ap.parse_args()

    req_path = args.requirements
    manifest_path = args.manifest
    if manifest_path is not None and not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path

    header: dict = {
        "scorecard_schema_version": "1.1.0",
        "purpose": "reproducibility_header_for_eval_runs",
        "interpreter": sys.executable,
        "repo_root": str(REPO_ROOT.resolve()),
        "git_commit": _git_head(),
        "requirements_txt_sha256": _sha256_file(req_path),
        "requirements_txt_path": str(req_path.resolve()) if req_path.is_file() else None,
    }
    if manifest_path is not None:
        header["manifest_path"] = str(manifest_path.resolve())
        header["manifest_sha256"] = _sha256_file(manifest_path)

    jsonl_paths = args.jsonl or []
    if jsonl_paths:
        artifacts: list[dict[str, str | None]] = []
        for raw in jsonl_paths:
            p = Path(raw)
            if not p.is_absolute():
                p = REPO_ROOT / p
            artifacts.append(
                {
                    "path": str(p.resolve()),
                    "sha256": _sha256_file(p),
                }
            )
        header["pose_jsonl_artifacts"] = artifacts

    print(json.dumps(header, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
