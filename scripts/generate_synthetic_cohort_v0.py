#!/usr/bin/env python3
"""Generate a synthetic per-rep cohort for v0 calibration validation.

Honesty contract
----------------
This script does NOT generate the cited reference ranges. The cited ranges
live in ``evaluation/gym_calibration_v0.json`` and are sourced from
``evaluation/calibration_evidence_v0/literature_bundle_v0.md``.

This script generates a *separate* artifact: 30 simulated reps per exercise
drawn from biomech-realistic Gaussians whose means/SDs come from the same
literature. The artifact lives at ``evaluation/synthetic_cohort_v0/`` and
serves two purposes:

1. Sanity-check that the cited bands cover ~p10..p90 of a realistic cohort
   (sanity output: per-exercise coverage % printed to stdout).
2. Give downstream code (and future eval runs) a synthetic-but-deterministic
   per-rep dataset to exercise the calibration / parity pipelines without
   needing real video.

Determinism
-----------
Seeded with ``--seed`` (default 0). The same seed produces the same artifact
byte-for-byte, so the SHA can be pinned in CI just like the manifest SHAs.

Usage
-----
    python scripts/generate_synthetic_cohort_v0.py \\
        --out evaluation/synthetic_cohort_v0 --seed 0

    python scripts/generate_synthetic_cohort_v0.py --check  # verify on-disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.gym.calibration_v0 import load_calibration_v0  # noqa: E402
from app.gym.exercises_v0 import EXERCISES_V0  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "evaluation" / "synthetic_cohort_v0"
DEFAULT_CALIBRATION = REPO_ROOT / "evaluation" / "gym_calibration_v0.json"

N_REPS = 30
COHORT_SCHEMA_VERSION = "1.0.0"


# Per-exercise (mean, sd) for each comparable field. Means come from the
# midpoint of the cited literature bands in literature_bundle_v0.md; SDs
# are sized so ~95% of draws fall inside the band (sd = (band_hi-band_lo)/4).
# This is intentional — the synthetic cohort should be a realistic *sample*
# of well-formed reps, not an adversarial set.
PER_EX_PARAMS: dict[str, dict[str, tuple[float, float]]] = {
    # --- compound barbell lifts --------------------------------------------
    "back_squat": {
        "rep_duration_s": (3.5, 1.0),
        "eccentric_duration_s": (2.0, 0.6),
        "concentric_duration_s": (1.4, 0.4),
        "tempo_ratio_ecc_over_con": (1.5, 0.5),
        "signal_amplitude": (0.20, 0.06),  # normalized_y
        "primary_joints_min_visibility": (0.80, 0.08),
        "primary_joints_missing_frac": (0.05, 0.05),
    },
    "front_squat": {
        "rep_duration_s": (3.5, 1.0),
        "eccentric_duration_s": (2.0, 0.6),
        "concentric_duration_s": (1.4, 0.4),
        "tempo_ratio_ecc_over_con": (1.5, 0.5),
        "signal_amplitude": (0.20, 0.06),
        "primary_joints_min_visibility": (0.78, 0.08),
        "primary_joints_missing_frac": (0.06, 0.05),
    },
    "conventional_deadlift": {
        "rep_duration_s": (3.5, 1.0),
        "eccentric_duration_s": (1.6, 0.5),
        "concentric_duration_s": (1.6, 0.5),
        "tempo_ratio_ecc_over_con": (1.0, 0.3),
        "signal_amplitude": (0.20, 0.07),
        "primary_joints_min_visibility": (0.78, 0.08),
        "primary_joints_missing_frac": (0.06, 0.05),
    },
    "romanian_deadlift": {
        "rep_duration_s": (3.5, 1.0),
        "eccentric_duration_s": (2.0, 0.6),
        "concentric_duration_s": (1.4, 0.4),
        "tempo_ratio_ecc_over_con": (1.5, 0.5),
        "signal_amplitude": (65.0, 15.0),  # deg (hip_angle change)
        "primary_joints_min_visibility": (0.80, 0.07),
        "primary_joints_missing_frac": (0.05, 0.04),
    },
    "bench_press": {
        "rep_duration_s": (3.0, 0.8),
        "eccentric_duration_s": (1.8, 0.5),
        "concentric_duration_s": (1.2, 0.4),
        "tempo_ratio_ecc_over_con": (1.5, 0.5),
        "signal_amplitude": (95.0, 15.0),  # deg (elbow flexion change)
        "primary_joints_min_visibility": (0.78, 0.08),
        "primary_joints_missing_frac": (0.06, 0.05),
    },
    "overhead_press": {
        "rep_duration_s": (3.0, 0.8),
        "eccentric_duration_s": (1.8, 0.5),
        "concentric_duration_s": (1.2, 0.4),
        "tempo_ratio_ecc_over_con": (1.5, 0.5),
        "signal_amplitude": (105.0, 18.0),
        "primary_joints_min_visibility": (0.80, 0.08),
        "primary_joints_missing_frac": (0.05, 0.05),
    },
    "barbell_row": {
        "rep_duration_s": (2.8, 0.7),
        "eccentric_duration_s": (1.6, 0.5),
        "concentric_duration_s": (1.2, 0.4),
        "tempo_ratio_ecc_over_con": (1.4, 0.4),
        "signal_amplitude": (75.0, 18.0),
        "primary_joints_min_visibility": (0.78, 0.08),
        "primary_joints_missing_frac": (0.06, 0.05),
    },
    # --- bodyweight cyclic --------------------------------------------------
    "pull_up": {
        "rep_duration_s": (2.5, 0.8),
        "eccentric_duration_s": (1.4, 0.5),
        "concentric_duration_s": (1.1, 0.4),
        "tempo_ratio_ecc_over_con": (1.3, 0.4),
        "signal_amplitude": (110.0, 18.0),
        "primary_joints_min_visibility": (0.80, 0.07),
        "primary_joints_missing_frac": (0.05, 0.05),
    },
    "push_up": {
        "rep_duration_s": (2.2, 0.6),
        "eccentric_duration_s": (1.2, 0.4),
        "concentric_duration_s": (1.0, 0.3),
        "tempo_ratio_ecc_over_con": (1.3, 0.4),
        "signal_amplitude": (95.0, 18.0),
        "primary_joints_min_visibility": (0.78, 0.09),
        "primary_joints_missing_frac": (0.07, 0.05),
    },
    "dumbbell_bicep_curl": {
        "rep_duration_s": (2.4, 0.6),
        "eccentric_duration_s": (1.4, 0.4),
        "concentric_duration_s": (1.0, 0.3),
        "tempo_ratio_ecc_over_con": (1.5, 0.4),
        "signal_amplitude": (120.0, 18.0),
        "primary_joints_min_visibility": (0.80, 0.07),
        "primary_joints_missing_frac": (0.05, 0.04),
    },
    # --- lunge --------------------------------------------------------------
    "walking_lunge": {
        "rep_duration_s": (3.0, 0.8),
        "eccentric_duration_s": (1.5, 0.5),
        "concentric_duration_s": (1.5, 0.5),
        "tempo_ratio_ecc_over_con": (1.0, 0.3),
        "signal_amplitude": (0.18, 0.06),
        "primary_joints_min_visibility": (0.76, 0.09),
        "primary_joints_missing_frac": (0.07, 0.05),
    },
    # --- isometric / gait (limited fields) ----------------------------------
    "plank": {
        "rep_duration_s": (60.0, 30.0),
        "signal_amplitude": (0.04, 0.02),  # instability proxy (small = stable)
        "primary_joints_min_visibility": (0.78, 0.08),
        "primary_joints_missing_frac": (0.05, 0.04),
    },
    "farmer_carry": {
        "signal_amplitude": (0.05, 0.02),
        "primary_joints_min_visibility": (0.74, 0.10),
        "primary_joints_missing_frac": (0.10, 0.06),
    },
}


# ----------------------------------------------------------------------------
# Deterministic Gaussian (Box-Muller on a seeded LCG so we don't depend on
# numpy's random state across versions).
# ----------------------------------------------------------------------------


class SeededRng:
    """Tiny deterministic PRNG (LCG, Numerical Recipes constants).

    Don't use for cryptography; perfect for reproducible test fixtures.
    """

    def __init__(self, seed: int) -> None:
        self.state = (seed * 1103515245 + 12345) & 0x7FFFFFFF

    def uniform(self) -> float:
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        # Avoid exact 0.0 (Box-Muller takes a log).
        return (self.state + 1) / 0x80000000

    def gauss(self, mean: float, sd: float) -> float:
        u1 = self.uniform()
        u2 = self.uniform()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + sd * z


def _draw_field(rng: SeededRng, mean: float, sd: float, low_clamp: float = 0.0) -> float:
    """One Gaussian draw, clamped at zero so durations / fractions stay valid."""
    return max(low_clamp, rng.gauss(mean, sd))


def _build_rep(
    rng: SeededRng,
    exercise_id: str,
    rep_index: int,
    params: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """One synthetic rep matching the RepFeatureVector wire shape."""
    features: dict[str, dict[str, Any]] = {}
    for fname, (mean, sd) in sorted(params.items()):
        # missing_frac is bounded above by 1; visibility too.
        if fname in ("primary_joints_missing_frac", "primary_joints_min_visibility"):
            v = max(0.0, min(1.0, rng.gauss(mean, sd)))
        else:
            v = _draw_field(rng, mean, sd)
        # Pick a unit consistent with rep_features._amplitude_feature.
        if fname.endswith("_s"):
            unit = "s"
        elif fname.endswith("_ratio_ecc_over_con") or fname.endswith("_ratio"):
            unit = "ratio"
        elif fname == "signal_amplitude":
            ex = EXERCISES_V0[exercise_id]
            if ex.rep_signal_type == "cyclic_angle":
                unit = "deg"
            else:
                unit = "normalized_y"
        elif fname == "primary_joints_min_visibility":
            unit = "visibility"
        elif fname == "primary_joints_missing_frac":
            unit = "frac"
        else:
            unit = ""
        features[fname] = {
            "value": round(v, 4),
            "unit": unit,
            "status": "valid",
            "reason_codes": ["synthetic_cohort_v0"],
        }
    return {
        "exercise_id": exercise_id,
        "rep_index": rep_index,
        "features": features,
    }


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (numpy-free, deterministic)."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _coverage_check(
    cohort: dict[str, list[dict[str, Any]]],
    cited_ranges: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, dict[str, float]]:
    """Per-exercise per-field % of synthetic reps falling inside the cited band."""
    out: dict[str, dict[str, float]] = {}
    for ex_id, reps in cohort.items():
        per_field: dict[str, float] = {}
        bands = cited_ranges.get(ex_id, {})
        for fname, (lo, hi) in bands.items():
            vals = [r["features"][fname]["value"] for r in reps if fname in r["features"]]
            if not vals:
                continue
            inside = sum(1 for v in vals if lo <= v <= hi)
            per_field[fname] = round(inside / len(vals), 3)
        out[ex_id] = per_field
    return out


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------


def _build_cohort(seed: int) -> dict[str, list[dict[str, Any]]]:
    cohort: dict[str, list[dict[str, Any]]] = {}
    # Per-exercise seeds derived from the master seed so adding a new exercise
    # later doesn't shift the draws of previously-frozen exercises.
    for ex_id in sorted(PER_EX_PARAMS.keys()):
        rng = SeededRng(seed + (sum(ord(c) for c in ex_id) << 4))
        params = PER_EX_PARAMS[ex_id]
        cohort[ex_id] = [
            _build_rep(rng, ex_id, i, params) for i in range(N_REPS)
        ]
    return cohort


def _write_cohort(cohort: dict[str, list[dict[str, Any]]], out_dir: Path, seed: int) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "seed": seed,
        "n_reps_per_exercise": N_REPS,
        "exercises": sorted(cohort.keys()),
    }
    files: list[tuple[str, str]] = []
    for ex_id, reps in sorted(cohort.items()):
        payload = {
            "schema_version": COHORT_SCHEMA_VERSION,
            "exercise_id": ex_id,
            "seed": seed,
            "n_reps": N_REPS,
            "reps": reps,
        }
        body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        path = out_dir / f"{ex_id}.json"
        path.write_text(body, encoding="utf-8")
        files.append((ex_id, hashlib.sha256(body.encode("utf-8")).hexdigest()))
    summary["files"] = [{"exercise_id": e, "sha256": s} for e, s in files]
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (out_dir / "manifest.json").write_text(summary_text, encoding="utf-8")
    return hashlib.sha256(summary_text.encode("utf-8")).hexdigest()


def _print_coverage(
    cohort: dict[str, list[dict[str, Any]]], calibration_path: Path
) -> None:
    if not calibration_path.is_file():
        print("calibration JSON missing; skipping coverage check")
        return
    manifest = load_calibration_v0(calibration_path)
    cited: dict[str, dict[str, tuple[float, float]]] = {
        e.exercise_id: dict(e.reference_ranges) for e in manifest.entries.values()
    }
    coverage = _coverage_check(cohort, cited)
    print("\n--- Coverage of cited bands by synthetic cohort (% of reps inside) ---")
    for ex_id in sorted(coverage.keys()):
        per_field = coverage[ex_id]
        if not per_field:
            print(f"  {ex_id}: no cited ranges")
            continue
        items = ", ".join(f"{k}={v:.0%}" for k, v in sorted(per_field.items()))
        print(f"  {ex_id}: {items}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="Calibration JSON to coverage-check against (read-only).",
    )
    ap.add_argument("--check", action="store_true", help="Verify on-disk cohort matches")
    args = ap.parse_args()

    cohort = _build_cohort(args.seed)

    if args.check:
        # Compare current build against on-disk manifest.json hash list.
        on_disk_manifest = args.out / "manifest.json"
        if not on_disk_manifest.is_file():
            print(json.dumps({"ok": False, "error": "manifest_not_found", "path": str(on_disk_manifest)}, indent=2))
            return 1
        # Re-emit to a temp dir, compare hashes via the in-memory cohort.
        # Here we just re-hash each ex's JSON and compare.
        on_disk = json.loads(on_disk_manifest.read_text(encoding="utf-8"))
        seen = {f["exercise_id"]: f["sha256"] for f in on_disk["files"]}
        drift: list[str] = []
        for ex_id, reps in sorted(cohort.items()):
            payload = {
                "schema_version": COHORT_SCHEMA_VERSION,
                "exercise_id": ex_id,
                "seed": args.seed,
                "n_reps": N_REPS,
                "reps": reps,
            }
            body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if seen.get(ex_id) != sha:
                drift.append(f"{ex_id}: on_disk={seen.get(ex_id)} expected={sha}")
        if drift:
            print(json.dumps({"ok": False, "drift": drift}, indent=2))
            return 1
        print(json.dumps({"ok": True, "n_files": len(seen)}, indent=2))
        return 0

    summary_sha = _write_cohort(cohort, args.out, args.seed)
    print(json.dumps({"ok": True, "out": str(args.out), "manifest_sha256": summary_sha}, indent=2))
    _print_coverage(cohort, args.calibration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
