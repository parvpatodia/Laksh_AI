"""
CSV manifest loading for gym pose evaluation (stdlib only — safe for --validate-only).
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_expect_min_detection_rate(raw: str | None) -> float | None:
    """
    Parse optional manifest threshold; must lie in [0, 1] when set (detection_rate is a ratio).
    Malformed values log a warning and behave as unset.
    """
    s = (raw or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        logger.warning("Invalid expect_min_detection_rate %r — ignoring", s)
        return None
    if not (0.0 <= v <= 1.0):
        logger.warning("expect_min_detection_rate %s outside [0,1] — ignoring", v)
        return None
    return v


def parse_expect_pose_usable(raw: str | None) -> bool | None:
    s = (raw or "").strip().lower()
    if s in ("yes", "true", "1", "y"):
        return True
    if s in ("no", "false", "0", "n"):
        return False
    return None


def load_gym_manifest(path: Path, base_dir: Path) -> list[dict[str, Any]]:
    """
    Load gym manifest CSV. Paths are joined to base_dir unless absolute.

    Expected columns: clip_id, path, tags, notes, exercise_id, expect_pose_usable,
    expect_min_detection_rate (extras ignored).

    Raises:
        ValueError: if a data row has an empty ``path`` (invalid manifest).
    """
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, r in enumerate(reader, start=2):
            rel = (r.get("path") or "").strip()
            if not rel:
                raise ValueError(
                    f"{path}: row {row_num}: empty 'path' (clip_id={r.get('clip_id')!r})"
                )
            cid = (r.get("clip_id") or "").strip() or Path(rel).stem
            vp = (base_dir / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
            min_dr = parse_expect_min_detection_rate(r.get("expect_min_detection_rate"))
            rows.append(
                {
                    "clip_id": cid,
                    "video_path": vp,
                    "tags": (r.get("tags") or "").strip() or None,
                    "exercise_id": (r.get("exercise_id") or "").strip() or None,
                    "expect_pose_usable": parse_expect_pose_usable(r.get("expect_pose_usable")),
                    "expect_min_detection_rate": min_dr,
                }
            )
    return rows


def check_manifest_expectations(
    result: dict[str, Any],
    expect_usable: bool | None,
    min_dr: float | None,
) -> list[str]:
    """
    Compare a single JSONL-shaped result dict to optional manifest expectations.

    ``min_dr`` is assumed already validated to [0,1] or None (see parse_expect_min_detection_rate).
    """
    errs: list[str] = []
    if not result.get("ok"):
        if expect_usable is True:
            errs.append("expected pose_usable=yes but backend failed")
        return errs
    if expect_usable is True and not result.get("pose_usable_heuristic"):
        errs.append("expected pose_usable=yes but heuristic was false")
    if expect_usable is False and result.get("pose_usable_heuristic"):
        errs.append("expected pose_usable=no but heuristic was true")
    if min_dr is not None:
        dr = float(result.get("detection_rate") or 0.0)
        if dr < min_dr:
            errs.append(f"expected detection_rate>={min_dr}, got {dr}")
    return errs


def summarize_manifest_path_status(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """For --validate-only: counts and first few missing paths."""
    missing: list[str] = []
    found = 0
    for job in jobs:
        vp: Path = job["video_path"]
        if vp.is_file():
            found += 1
        else:
            missing.append(str(vp))
    return {
        "clips_in_manifest": len(jobs),
        "files_present": found,
        "files_missing": len(missing),
        "missing_paths_sample": missing[:5],
    }
