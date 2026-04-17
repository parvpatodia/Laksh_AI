#!/usr/bin/env python3
"""
Download MediaPipe pose_landmarker_heavy.task if missing.

Default destination: repo root, where `app.physics_engine` expects `pose_landmarker_heavy.task`.
Override with --dest or env LAKSH_POSE_MODEL_DEST (e.g. /app/pose_landmarker_heavy.task in Docker).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pose.expected_artifacts import POSE_LANDMARKER_HEAVY_TASK_SHA256

DEFAULT_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_dest() -> Path:
    env = os.environ.get("LAKSH_POSE_MODEL_DEST", "").strip()
    if env:
        return Path(env)
    return REPO_ROOT / "pose_landmarker_heavy.task"


def main() -> int:
    ap = argparse.ArgumentParser(description="Download MediaPipe Pose landmarker heavy model.")
    ap.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Output path (default: repo root or LAKSH_POSE_MODEL_DEST)",
    )
    ap.add_argument("--url", default=DEFAULT_URL, help="Model URL")
    args = ap.parse_args()

    dest = args.dest or default_dest()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 1_000_000:
        got = _sha256_file(dest)
        if got == POSE_LANDMARKER_HEAVY_TASK_SHA256:
            print(f"Pose model already present: {dest} ({dest.stat().st_size // 1024 // 1024} MB)")
            return 0
        print(f"Existing file SHA-256 mismatch ({got[:12]}…); re-downloading")
        dest.unlink()

    print(f"Downloading pose model → {dest}")
    urllib.request.urlretrieve(args.url, str(dest))
    print(f"Done: {dest.stat().st_size} bytes")
    got = _sha256_file(dest)
    if got != POSE_LANDMARKER_HEAVY_TASK_SHA256:
        print(
            "ERROR: Downloaded file SHA-256 does not match expected.\n"
            f"  got:      {got}\n"
            f"  expected: {POSE_LANDMARKER_HEAVY_TASK_SHA256}\n"
            "Update app/pose/expected_artifacts.py if MediaPipe published a new blob at the same URL."
        )
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
