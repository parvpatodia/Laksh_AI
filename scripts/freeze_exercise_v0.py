#!/usr/bin/env python3
"""Freeze or verify the exercise v0 taxonomy (GOALS.md Milestone 1).

Reads :mod:`app.gym.exercises_v0` and emits a deterministic JSON manifest
with schema version, manifest version, and SHA-256. Intended artifact path:
``evaluation/exercise_v0_manifest.json`` — committed so the scorecard header
and manifest validators can pin the exact taxonomy used at eval time.

Modes
-----

Freeze (default): write ``--out`` from the in-code registry::

    python scripts/freeze_exercise_v0.py --out evaluation/exercise_v0_manifest.json

Verify: exit non-zero if the on-disk JSON drifts from the in-code registry
(for CI and ``make check-exercise-v0``)::

    python scripts/freeze_exercise_v0.py --verify --out evaluation/exercise_v0_manifest.json

Print only (no file I/O)::

    python scripts/freeze_exercise_v0.py --print

Exit codes
----------

* ``0`` — freeze succeeded / verify matched / print succeeded.
* ``1`` — verify mismatch (on-disk drifted from source of truth).
* ``2`` — file I/O or parse error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.gym.exercises_v0 import (  # noqa: E402
    EXERCISE_V0_MANIFEST_VERSION,
    EXERCISE_V0_SCHEMA_VERSION,
    compute_manifest_sha256,
    to_manifest_dict,
)

DEFAULT_OUT = REPO_ROOT / "evaluation" / "exercise_v0_manifest.json"


def build_frozen_payload() -> dict[str, Any]:
    """In-code registry -> JSON-ready payload with sha256 appended."""
    payload = to_manifest_dict()
    payload["sha256"] = compute_manifest_sha256()
    return payload


def _dump(payload: dict[str, Any]) -> str:
    # indent=2 for human diff-friendliness; sort_keys=True for stable ordering
    # except we already preserve `exercises` list order via the list serialisation.
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def freeze(out: Path) -> int:
    payload = build_frozen_payload()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_dump(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(out),
                "schema_version": payload["schema_version"],
                "manifest_version": payload["manifest_version"],
                "sha256": payload["sha256"],
                "n_exercises": len(payload["exercises"]),
            },
            indent=2,
        )
    )
    return 0


def verify(out: Path) -> int:
    if not out.is_file():
        print(
            json.dumps(
                {"error": "frozen manifest not found", "expected_path": str(out)},
                indent=2,
            )
        )
        return 2
    try:
        on_disk = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(json.dumps({"error": f"failed to read {out}: {e}"}, indent=2))
        return 2
    expected = build_frozen_payload()
    drift: list[str] = []
    for key in ("schema_version", "manifest_version", "sha256"):
        if on_disk.get(key) != expected.get(key):
            drift.append(f"{key}: on_disk={on_disk.get(key)!r} expected={expected.get(key)!r}")
    # Structural equality on the whole payload — catches silent field changes
    # that don't change the three headline fields.
    if on_disk != expected:
        if not drift:
            drift.append("payload differs (same headline fields, different body)")
    if drift:
        print(
            json.dumps(
                {
                    "ok": False,
                    "path": str(out),
                    "drift": drift,
                    "expected_sha256": expected["sha256"],
                    "on_disk_sha256": on_disk.get("sha256"),
                },
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(out),
                "schema_version": EXERCISE_V0_SCHEMA_VERSION,
                "manifest_version": EXERCISE_V0_MANIFEST_VERSION,
                "sha256": expected["sha256"],
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Frozen manifest path")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Check on-disk matches in-code registry")
    mode.add_argument("--print", dest="print_only", action="store_true", help="Print payload, no file I/O")
    args = ap.parse_args()

    if args.print_only:
        print(_dump(build_frozen_payload()), end="")
        return 0
    if args.verify:
        return verify(args.out)
    return freeze(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
