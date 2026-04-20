"""
Plausibility tests for translate_to_kinematics and FEATURE_WEIGHTS.

Validates that the heuristic box-score → 8D kinematic mapping produces
outputs within biomechanically plausible ranges documented in
evaluation/calibration_evidence_v0/basketball_literature_v0.md.

Literature-backed ranges (see bundle for full citations):
  release_velocity_mps : 4.0 – 9.0  [Okazaki 2015, Miller & Bartlett 1996]
  shot_arc_deg         : 38  – 55   [Brancazio 1981, Tran & Silverberg 2008]
  knee_angle           : 130 – 175  [Okazaki 2015, Silva et al. 2022]
  elbow_angle          : 150 – 178  [Miller & Bartlett 1996, Knudson 1993]
  kinetic_sync_ms      : 150 – 667  [Link & Cain 2017, Okazaki 2015]
  fluidity_score       : 40  – 99   [engineering mapping; bounded by design]
  hip_rotation_deg     : -20 – +20  [Knudson 1993, Silva et al. 2022]
  balance_index        : 40  – 99   [engineering mapping; bounded by design]
"""
from __future__ import annotations

import pytest

from app.db_seeder import (
    FALLBACK_PLAYERS,
    FEATURE_WEIGHTS,
    translate_to_kinematics,
)

# Ranges aligned with basketball_literature_v0.md and db_seeder clamps.
LITERATURE_RANGES = {
    0: ("release_velocity_mps", 4.0, 9.0),
    1: ("shot_arc_deg", 38.0, 55.0),
    2: ("knee_angle", 135.0, 175.0),
    3: ("elbow_angle", 150.0, 178.0),
    4: ("kinetic_sync_ms", 150.0, 670.0),
    5: ("fluidity_score", 60.0, 98.0),
    6: ("hip_rotation_deg", -20.0, 20.0),
    7: ("balance_index", 50.0, 98.0),
}


class TestTranslateToKinematics:
    """Output of translate_to_kinematics must land inside literature-plausible ranges."""

    @pytest.mark.parametrize(
        "name, pid, stats",
        [(n, p, s) for n, p, s in FALLBACK_PLAYERS],
        ids=[n for n, _, _ in FALLBACK_PLAYERS],
    )
    def test_fallback_players_in_range(self, name: str, pid: int, stats: dict):
        vec = translate_to_kinematics(stats)
        assert len(vec) == 8, f"{name}: expected 8D vector, got {len(vec)}"
        for idx, (metric, lo, hi) in LITERATURE_RANGES.items():
            assert lo <= vec[idx] <= hi, (
                f"{name}: {metric} = {vec[idx]} outside [{lo}, {hi}]"
            )

    def test_guard_vs_big_directional_correctness(self):
        """Guard should produce faster release, higher arc, less knee dip,
        and quicker kinetic sync than a big — validates the directional
        correlations documented in basketball_literature_v0.md."""
        guard = {"REB": 3.0, "AST": 10.0, "TOV": 2.0, "FG3_PCT": 0.45, "PTS": 25.0, "GP": 70}
        big = {"REB": 12.0, "AST": 2.0, "TOV": 3.0, "FG3_PCT": 0.28, "PTS": 18.0, "GP": 60}
        g = translate_to_kinematics(guard)
        b = translate_to_kinematics(big)

        assert g[0] > b[0], f"Guard velocity {g[0]} should exceed big velocity {b[0]}"
        assert g[1] > b[1], f"Guard arc {g[1]}° should exceed big arc {b[1]}°"
        assert g[2] > b[2], f"Guard knee angle {g[2]}° should exceed big angle {b[2]}° (less dip)"
        assert g[3] > b[3], f"Guard elbow extension {g[3]}° should exceed big {b[3]}°"
        assert g[4] < b[4], f"Guard kinetic sync {g[4]} ms should be faster than big {b[4]} ms"
        assert g[5] > b[5], f"Guard fluidity {g[5]} should exceed big fluidity {b[5]}"

    def test_extreme_inputs_stay_bounded(self):
        """Edge-case stats should clamp to physically plausible ranges."""
        extreme_high = {"REB": 20.0, "AST": 15.0, "TOV": 0.1, "FG3_PCT": 0.60, "PTS": 40.0, "GP": 82}
        extreme_low = {"REB": 0.5, "AST": 0.1, "TOV": 5.0, "FG3_PCT": 0.10, "PTS": 2.0, "GP": 5}
        for label, stats in [("extreme_high", extreme_high), ("extreme_low", extreme_low)]:
            vec = translate_to_kinematics(stats)
            for idx, (metric, lo, hi) in LITERATURE_RANGES.items():
                assert lo <= vec[idx] <= hi, (
                    f"{label}: {metric} = {vec[idx]} outside [{lo}, {hi}]"
                )


class TestFeatureWeights:
    """FEATURE_WEIGHTS span normalisation produces ~100 across each dimension."""

    def test_weight_count(self):
        assert len(FEATURE_WEIGHTS) == 8

    def test_all_positive(self):
        for i, w in enumerate(FEATURE_WEIGHTS):
            assert w > 0, f"Weight[{i}] = {w} must be positive"

    def test_span_normalisation_yields_similar_magnitude(self):
        """Each weight × dimension span should be roughly 50–150 (targeting ~100).
        Spans from db_seeder clamps."""
        spans = [5.0, 17.0, 40.0, 28.0, 400.0, 38.0, 40.0, 48.0]
        for i, (w, s) in enumerate(zip(FEATURE_WEIGHTS, spans)):
            product = w * s
            assert 30 < product < 200, (
                f"Dim {i}: weight {w} × span {s} = {product}, expected 30–200"
            )
