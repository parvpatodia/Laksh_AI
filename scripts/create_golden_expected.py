#!/usr/bin/env python3
"""
Create golden_expected.json from golden_shot.mp4.

Run from project root:
  python scripts/create_golden_expected.py

Requires: .venv activated, or: .venv/bin/python scripts/create_golden_expected.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_VIDEO = PROJECT_ROOT / "tests" / "fixtures" / "golden_shot.mp4"
GOLDEN_OUT = PROJECT_ROOT / "tests" / "fixtures" / "golden_expected.json"

# Tolerances: allow small drift across runs (lighting, MediaPipe non-determinism)
TOLERANCES = {
    "release_velocity_mps": 0.6,
    "shot_arc_deg": 6.0,
    "knee_angle": 10.0,
    "elbow_angle": 10.0,
    "kinetic_sync_ms": 50.0,
    "hip_rotation_deg": 8.0,
    "balance_index": 15,
    "fluidity_score": 15,
}


def main():
    if not GOLDEN_VIDEO.exists():
        print(f"Error: {GOLDEN_VIDEO} not found")
        sys.exit(1)

    sys.path.insert(0, str(PROJECT_ROOT))
    from app.physics_engine import KinematicAnalyzer

    print("Analyzing golden video...")
    result = KinematicAnalyzer(str(GOLDEN_VIDEO)).analyze()

    # Check for fallback
    tel = result.get("telemetry", {})
    if tel.get("validation_warnings"):
        for w in tel["validation_warnings"]:
            if "fallback" in str(w).lower():
                print("WARNING: Analysis used fallback. Video may lack a detectable shot.")
                break

    baseline = {}
    for key in TOLERANCES:
        val = result.get(key)
        if val is not None:
            baseline[key] = float(val) if not isinstance(val, (int, float)) else val

    out = {
        "baseline": baseline,
        "tolerances": TOLERANCES,
        "_comment": "Auto-generated from golden_shot.mp4. Tolerances allow CI drift.",
    }

    GOLDEN_OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {GOLDEN_OUT}")
    print("Baseline metrics:", json.dumps(baseline, indent=2))
    print("\nRun: pytest tests/test_physics_regression.py -v")


if __name__ == "__main__":
    main()
