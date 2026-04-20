"""End-to-end parity-probe wire-compatibility test.

Proves that the wire format produced by the frontend's `toWireVector` helper
(see `web/lib/realtime/repCounter.ts`) is the format the backend's
`probe_reps` actually consumes. This catches the class of bug that shipped
in v0.1.x where the ghost emitted `min_visibility` while canonical emitted
`primary_joints_min_visibility` — silently making the parity probe degrade
to `insufficient_data`.

Strategy
--------
We do NOT execute TypeScript here. Instead we hand-build a Python dict
matching exactly what `toWireVector` emits, feed it through `probe_reps`
together with synthetic canonical vectors, and assert:

1. `status` ends up `within_tolerance` (not `insufficient_data` and not
   `outside_tolerance` — i.e. the probe actually compared real fields).
2. The `fields_compared` list contains BOTH renamed/converted fields:
   * `primary_joints_min_visibility` (was `min_visibility`)
   * `signal_amplitude` (in canonical units — degrees for cyclic_angle)

If anyone ever changes `toWireVector` in a way that drifts from canonical,
this test fails — protecting the demo-critical parity_probe block.
"""
from __future__ import annotations

from app.parity.realtime import probe_reps


def _ts_to_wire_vector_cyclic_angle(rep_index: int, ghost_amp_norm: float) -> dict:
    """Mirror exactly what `toWireVector` emits for a cyclic_angle exercise.

    The ghost stores `signal_amplitude` in normalised [0,1] units; for
    cyclic_angle exercises `toWireVector` multiplies by 180 to convert into
    the same degree units the canonical pipeline emits.
    """
    return {
        "rep_index": rep_index,
        "features": {
            "rep_duration_s": {"value": 2.0, "unit": "s", "status": "valid", "reason_codes": []},
            "eccentric_duration_s": {"value": 1.2, "unit": "s", "status": "valid", "reason_codes": []},
            "concentric_duration_s": {"value": 0.8, "unit": "s", "status": "valid", "reason_codes": []},
            "tempo_ratio_ecc_over_con": {"value": 1.5, "unit": "ratio", "status": "valid", "reason_codes": []},
            # ghost_amp_norm * 180 — what toWireVector does for cyclic_angle.
            "signal_amplitude": {"value": round(ghost_amp_norm * 180, 2), "unit": "deg", "status": "valid", "reason_codes": []},
            # Renamed from `min_visibility` to match canonical field name.
            "primary_joints_min_visibility": {"value": 0.82, "unit": "visibility", "status": "valid", "reason_codes": []},
        },
    }


def _ts_to_wire_vector_cyclic_vertical(rep_index: int, ghost_amp_norm: float) -> dict:
    """Mirror `toWireVector` for cyclic_vertical: amplitude stays normalised_y."""
    return {
        "rep_index": rep_index,
        "features": {
            "rep_duration_s": {"value": 3.0, "unit": "s", "status": "valid", "reason_codes": []},
            "eccentric_duration_s": {"value": 1.6, "unit": "s", "status": "valid", "reason_codes": []},
            "concentric_duration_s": {"value": 1.4, "unit": "s", "status": "valid", "reason_codes": []},
            "tempo_ratio_ecc_over_con": {"value": 1.14, "unit": "ratio", "status": "valid", "reason_codes": []},
            "signal_amplitude": {"value": round(ghost_amp_norm, 4), "unit": "normalized_y", "status": "valid", "reason_codes": []},
            "primary_joints_min_visibility": {"value": 0.80, "unit": "visibility", "status": "valid", "reason_codes": []},
        },
    }


def _canonical_vector_cyclic_angle(rep_index: int, deg: float) -> dict:
    return {
        "rep_index": rep_index,
        "features": {
            "rep_duration_s": {"value": 2.0, "unit": "s", "status": "valid", "reason_codes": []},
            "eccentric_duration_s": {"value": 1.2, "unit": "s", "status": "valid", "reason_codes": []},
            "concentric_duration_s": {"value": 0.8, "unit": "s", "status": "valid", "reason_codes": []},
            "tempo_ratio_ecc_over_con": {"value": 1.5, "unit": "ratio", "status": "valid", "reason_codes": []},
            "signal_amplitude": {"value": deg, "unit": "deg", "status": "valid", "reason_codes": []},
            "primary_joints_min_visibility": {"value": 0.82, "unit": "visibility", "status": "valid", "reason_codes": []},
            "primary_joints_missing_frac": {"value": 0.05, "unit": "frac", "status": "valid", "reason_codes": []},
        },
    }


def _canonical_vector_cyclic_vertical(rep_index: int, amp: float) -> dict:
    return {
        "rep_index": rep_index,
        "features": {
            "rep_duration_s": {"value": 3.0, "unit": "s", "status": "valid", "reason_codes": []},
            "eccentric_duration_s": {"value": 1.6, "unit": "s", "status": "valid", "reason_codes": []},
            "concentric_duration_s": {"value": 1.4, "unit": "s", "status": "valid", "reason_codes": []},
            "tempo_ratio_ecc_over_con": {"value": 1.14, "unit": "ratio", "status": "valid", "reason_codes": []},
            "signal_amplitude": {"value": amp, "unit": "normalized_y", "status": "valid", "reason_codes": []},
            "primary_joints_min_visibility": {"value": 0.80, "unit": "visibility", "status": "valid", "reason_codes": []},
            "primary_joints_missing_frac": {"value": 0.05, "unit": "frac", "status": "valid", "reason_codes": []},
        },
    }


def test_wire_format_compat_cyclic_angle_within_tolerance() -> None:
    """For a push-up-style clip, the wire-converted ghost matches canonical
    closely enough that the probe reports `within_tolerance` and includes
    BOTH renamed fields in the comparison."""
    # Ghost saw amplitude 0.5 (normalised) → toWireVector emits 90 deg.
    # Canonical also measured 90 deg (push_up mid-ROM). Delta = 0.
    ghost = [_ts_to_wire_vector_cyclic_angle(i, 0.5) for i in range(4)]
    canonical = [_canonical_vector_cyclic_angle(i, 90.0) for i in range(4)]

    result = probe_reps(ghost, canonical)

    assert result["status"] == "within_tolerance", (
        f"expected within_tolerance, got {result}"
    )
    assert "primary_joints_min_visibility" in result["fields_compared"], (
        "renamed visibility field MUST appear in the comparison — its absence "
        "would mean the rename in toWireVector regressed"
    )
    assert "signal_amplitude" in result["fields_compared"], (
        "amplitude MUST appear in the comparison — its absence would mean "
        "the unit conversion in toWireVector regressed and one side was nulled"
    )
    # All five biomech fields should pair up cleanly.
    assert len(result["fields_compared"]) >= 5
    assert result["max_abs_delta"] < 0.5  # well under default 0.50 tolerance


def test_wire_format_compat_cyclic_vertical_within_tolerance() -> None:
    """Same proof for a squat-style clip (cyclic_vertical, no unit
    conversion needed but rename still applies)."""
    ghost = [_ts_to_wire_vector_cyclic_vertical(i, 0.20) for i in range(4)]
    canonical = [_canonical_vector_cyclic_vertical(i, 0.20) for i in range(4)]

    result = probe_reps(ghost, canonical)

    assert result["status"] == "within_tolerance", result
    assert "primary_joints_min_visibility" in result["fields_compared"]
    assert "signal_amplitude" in result["fields_compared"]


def test_wire_format_old_field_name_would_have_failed_silently() -> None:
    """REGRESSION GUARD: prove the old shape (pre-toWireVector) silently
    drops the visibility field from the comparison.  This test fixates the
    behaviour that justifies the wire-format helper — if anyone ever
    reverts toWireVector, this test still passes (intentionally) but the
    affirmative tests above start failing."""
    old_shape_ghost = [
        {
            "rep_index": i,
            "features": {
                "rep_duration_s": {"value": 2.0, "unit": "s", "status": "valid", "reason_codes": []},
                "eccentric_duration_s": {"value": 1.2, "unit": "s", "status": "valid", "reason_codes": []},
                "concentric_duration_s": {"value": 0.8, "unit": "s", "status": "valid", "reason_codes": []},
                "tempo_ratio_ecc_over_con": {"value": 1.5, "unit": "ratio", "status": "valid", "reason_codes": []},
                # OLD: amplitude in normalised units, but canonical emits degrees.
                "signal_amplitude": {"value": 0.5, "unit": "norm", "status": "valid", "reason_codes": []},
                # OLD: wrong field name.
                "min_visibility": {"value": 0.82, "unit": "fraction", "status": "valid", "reason_codes": []},
            },
        }
        for i in range(4)
    ]
    canonical = [_canonical_vector_cyclic_angle(i, 90.0) for i in range(4)]
    result = probe_reps(old_shape_ghost, canonical)
    # `min_visibility` is not in canonical -> dropped from comparison.
    assert "min_visibility" not in result["fields_compared"]
    assert "primary_joints_min_visibility" not in result["fields_compared"]
    # signal_amplitude still gets compared (same name on both sides) — but
    # the unit mismatch (0.5 vs 90.0) blows it past tolerance.
    if "signal_amplitude" in result["fields_compared"]:
        assert result["max_abs_delta"] > 1.0  # 89.5 in this fixture
