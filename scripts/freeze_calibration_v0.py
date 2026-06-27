#!/usr/bin/env python3
"""Verify the gym calibration v0 config against the in-code policy.

Unlike ``freeze_exercise_v0.py`` (where the Python module is the source of
truth), the calibration config's source of truth is the JSON file itself
(``evaluation/gym_calibration_v0.json``). Each labeled eval run that
justifies new reference ranges will edit that JSON directly; this script
just enforces the schema + GOALS.md calibration policy on the file.

Modes
-----

Verify (default): parse + validate the config, compute SHA-256, exit 0 on
success::

    python scripts/freeze_calibration_v0.py --config evaluation/gym_calibration_v0.json

Print: parse, re-serialise in canonical form, print to stdout::

    python scripts/freeze_calibration_v0.py --print

Expected-sha: verify the on-disk SHA-256 matches a pinned value (CI guard)::

    python scripts/freeze_calibration_v0.py --expected-sha <hex>

Exit codes
----------

* ``0`` — config parsed, validated, and SHA matched (if provided).
* ``1`` — validation failed OR SHA mismatched.
* ``2`` — file not found / parse error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.gym.calibration_v0 import (  # noqa: E402
    CALIBRATION_V0_MANIFEST_VERSION,
    CALIBRATION_V0_SCHEMA_VERSION,
    compute_manifest_sha256,
    load_calibration_v0,
    manifest_to_dict,
)

DEFAULT_CONFIG = REPO_ROOT / "evaluation" / "gym_calibration_v0.json"


def _emit(obj: dict) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def verify(config_path: Path, expected_sha: str | None) -> int:
    if not config_path.is_file():
        _emit({"ok": False, "error": "config_not_found", "path": str(config_path)})
        return 2
    try:
        manifest = load_calibration_v0(config_path)
    except (OSError, json.JSONDecodeError) as e:
        _emit({"ok": False, "error": f"parse_error: {e}", "path": str(config_path)})
        return 2
    except ValueError as e:
        _emit({"ok": False, "error": f"validation_error: {e}", "path": str(config_path)})
        return 1
    sha = compute_manifest_sha256(manifest)
    n_uncalibrated = sum(
        1 for e in manifest.entries.values() if e.evidence_status == "uncalibrated_v0"
    )
    n_cited = sum(1 for e in manifest.entries.values() if e.evidence_status == "cited")
    payload = {
        "ok": True,
        "path": str(config_path),
        "schema_version": manifest.schema_version,
        "manifest_version": manifest.manifest_version,
        "sha256": sha,
        "n_entries": len(manifest.entries),
        "n_uncalibrated_v0": n_uncalibrated,
        "n_cited": n_cited,
        "evidence_source": manifest.evidence_source,
    }
    if expected_sha is not None and sha != expected_sha:
        payload["ok"] = False
        payload["error"] = "sha_mismatch"
        payload["expected_sha256"] = expected_sha
        _emit(payload)
        return 1
    _emit(payload)
    return 0


def print_canonical(config_path: Path) -> int:
    if not config_path.is_file():
        _emit({"ok": False, "error": "config_not_found", "path": str(config_path)})
        return 2
    try:
        manifest = load_calibration_v0(config_path)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        _emit({"ok": False, "error": str(e), "path": str(config_path)})
        return 2
    print(json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to calibration JSON (default: evaluation/gym_calibration_v0.json)",
    )
    ap.add_argument(
        "--expected-sha",
        type=str,
        default=None,
        help="Pin: fail if computed SHA-256 does not match this value",
    )
    ap.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Re-serialise config canonically to stdout; no validation summary",
    )
    ap.add_argument(
        "--show-versions",
        action="store_true",
        help="Print the schema/manifest versions the installed code expects",
    )
    args = ap.parse_args()

    if args.show_versions:
        _emit(
            {
                "schema_version": CALIBRATION_V0_SCHEMA_VERSION,
                "manifest_version": CALIBRATION_V0_MANIFEST_VERSION,
            }
        )
        return 0
    if args.print_only:
        return print_canonical(args.config)
    return verify(args.config, args.expected_sha)


if __name__ == "__main__":
    raise SystemExit(main())
