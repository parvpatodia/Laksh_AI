#!/usr/bin/env python3
"""
Download MediaPipe pose_landmarker_heavy.task if missing.

Default destination: repo root, where `app.physics_engine` expects `pose_landmarker_heavy.task`.
Override with --dest or env LAKSH_POSE_MODEL_DEST (e.g. /app/pose_landmarker_heavy.task in Docker).
"""
from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)

REPO_ROOT = Path(__file__).resolve().parents[1]


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
        print(f"Pose model already present: {dest} ({dest.stat().st_size // 1024 // 1024} MB)")
        return 0

    print(f"Downloading pose model → {dest}")
    urllib.request.urlretrieve(args.url, str(dest))
    print(f"Done: {dest.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
