"""
Regression test for the physics engine.

Runs KinematicAnalyzer on a golden video and asserts metrics are within expected ranges.
Skips if tests/fixtures/golden_shot.mp4 or golden_expected.json is missing.

To add a golden video: see docs/GOLDEN_VIDEO_GUIDE.md
"""
from pathlib import Path

import pytest

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_VIDEO = PROJECT_ROOT / "tests" / "fixtures" / "golden_shot.mp4"
GOLDEN_EXPECTED = PROJECT_ROOT / "tests" / "fixtures" / "golden_expected.json"


def _golden_available() -> bool:
    return GOLDEN_VIDEO.exists() and GOLDEN_EXPECTED.exists()


@pytest.mark.skipif(not _golden_available(), reason="Golden video or expected JSON not found")
def test_physics_engine_reproduces_golden_output():
    """
    Run KinematicAnalyzer on golden_shot.mp4 and assert metrics are within tolerance.
    """
    import json

    from app.physics_engine import KinematicAnalyzer

    result = KinematicAnalyzer(str(GOLDEN_VIDEO)).analyze()
    if result.get("analysis_mode") == "fallback":
        pytest.skip(f"Pose backend unavailable in this environment: {result.get('fallback_reason_codes')}")

    with open(GOLDEN_EXPECTED) as f:
        expected = json.load(f)

    tolerances = expected.get("tolerances", {})
    baseline = expected.get("baseline", {})

    errors = []
    for key, base_val in baseline.items():
        if key == "telemetry" or key == "tolerances":
            continue
        actual = result.get(key)
        if actual is None:
            errors.append(f"{key}: missing in output")
            continue
        tol = tolerances.get(key, 0)
        if tol == 0:
            continue  # No tolerance defined, skip
        if abs(float(actual) - float(base_val)) > tol:
            errors.append(
                f"{key}: expected ~{base_val} (±{tol}), got {actual}"
            )

    assert not errors, "Metric drift:\n  " + "\n  ".join(errors)
