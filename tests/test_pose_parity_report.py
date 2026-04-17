"""Unit tests for scripts/pose_parity_report.py (ADR 0002 promotion gate)."""
from __future__ import annotations

from scripts import pose_parity_report as pr


def _probe(knee: float | None, elbow: float | None, error: str | None = None) -> dict:
    out: dict = {
        "enabled": True,
        "canonical_joint_schema_version": "test-v1",
        "mapping_id": "mediapipe_blazepose33_to_canonical",
    }
    if error is not None:
        out["error"] = error
        return out
    if knee is not None:
        out["delta_knee_vs_legacy_2d_deg"] = knee
    if elbow is not None:
        out["delta_elbow_vs_legacy_2d_deg"] = elbow
    return out


def test_extract_probe_handles_all_known_paths() -> None:
    p = _probe(0.1, 0.2)
    assert pr.extract_probe({"telemetry": {"canonical_joint_path": p}}) is p
    assert pr.extract_probe({"summary": {"canonical_joint_path": p}}) is p
    assert pr.extract_probe({"canonical_joint_path": p}) is p
    assert pr.extract_probe({"other": 1}) is None
    # Non-dict at probe position is ignored.
    assert pr.extract_probe({"canonical_joint_path": "not a dict"}) is None


def test_summarise_empty_returns_gate_absent() -> None:
    report = pr.summarise([])
    assert report["n_rows"] == 0
    assert report["n_probe_present"] == 0
    assert report["knee_delta_abs_deg"]["n"] == 0
    assert report["elbow_delta_abs_deg"]["n"] == 0
    # With no data, gate cannot pass.
    assert report["promotion_gate_pass"] is False


def test_summarise_passes_when_deltas_under_threshold() -> None:
    rows = [
        {"canonical_joint_path": _probe(0.5, 0.8)},
        {"canonical_joint_path": _probe(-0.4, 1.1)},
        {"canonical_joint_path": _probe(0.9, -0.7)},
    ]
    report = pr.summarise(rows, threshold_deg=2.0)
    assert report["n_probe_present"] == 3
    assert report["knee_delta_abs_deg"]["n"] == 3
    assert report["knee_delta_abs_deg"]["max"] <= 2.0
    assert report["elbow_delta_abs_deg"]["max"] <= 2.0
    assert report["knee_p90_pass"] is True
    assert report["elbow_p90_pass"] is True
    assert report["promotion_gate_pass"] is True


def test_summarise_fails_gate_when_p90_exceeds_threshold() -> None:
    # 5 small + 5 large: sorted p90 at pos 9*0.9=8.1 lands inside the outlier
    # cluster, so p90 = 5.0 > threshold 2.0.
    rows = [{"canonical_joint_path": _probe(0.2, 0.2)} for _ in range(5)]
    rows += [{"canonical_joint_path": _probe(5.0, 5.0)} for _ in range(5)]
    report = pr.summarise(rows, threshold_deg=2.0)
    assert report["knee_delta_abs_deg"]["p90"] is not None
    assert report["knee_delta_abs_deg"]["p90"] > 2.0
    assert report["knee_p90_pass"] is False
    assert report["promotion_gate_pass"] is False


def test_summarise_counts_errors_and_excludes_from_stats() -> None:
    rows = [
        {"canonical_joint_path": _probe(0.1, 0.1)},
        {"canonical_joint_path": _probe(None, None, error="missing_joint_in_canonical_frame")},
        {"canonical_joint_path": _probe(None, None, error="key_frame_out_of_range")},
        {"canonical_joint_path": _probe(None, None, error="unknown_future_code")},
    ]
    report = pr.summarise(rows)
    assert report["n_probe_present"] == 4
    assert report["error_counts"]["missing_joint_in_canonical_frame"] == 1
    assert report["error_counts"]["key_frame_out_of_range"] == 1
    assert report["error_counts"]["other"] == 1
    # Only the clean row contributes to stats.
    assert report["knee_delta_abs_deg"]["n"] == 1
    assert report["elbow_delta_abs_deg"]["n"] == 1


def test_summarise_counts_rows_without_probes() -> None:
    rows = [
        {"clip_id": "a", "ok": True},
        {"canonical_joint_path": _probe(0.3, 0.3)},
    ]
    report = pr.summarise(rows)
    assert report["n_rows"] == 2
    assert report["n_probe_present"] == 1
    assert report["n_probe_absent"] == 1


def test_summarise_ignores_nan_and_inf_deltas() -> None:
    rows = [
        {"canonical_joint_path": _probe(float("nan"), float("inf"))},
        {"canonical_joint_path": _probe(0.25, 0.25)},
    ]
    report = pr.summarise(rows, threshold_deg=1.0)
    # Only the clean row should contribute.
    assert report["knee_delta_abs_deg"]["n"] == 1
    assert report["elbow_delta_abs_deg"]["n"] == 1
    assert report["promotion_gate_pass"] is True


def test_quantile_linear_interpolation() -> None:
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    # P50 of 5 evenly spaced values is the middle element.
    assert pr._quantile(xs, 0.5) == 2.0
    # P90 is between idx 3 and 4 at fraction 0.6 -> 3 + 0.6 = 3.6.
    assert pr._quantile(xs, 0.9) == 3.6
    assert pr._quantile([], 0.5) is None
    assert pr._quantile([7.0], 0.9) == 7.0
