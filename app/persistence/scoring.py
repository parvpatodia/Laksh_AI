"""Transparent, measured-only leaderboard index.

The honesty contract forbids claiming an uncalibrated value sits inside a
reference band. This scorer therefore produces a RELATIVE ranking index, not a
form grade: a 0-100 number derived only from MEASURED (``status="valid"``)
quantities, tagged ``uncalibrated_demo_index`` and returned with its component
breakdown so any judge can audit exactly what produced the number.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

FORM_INDEX_UNIT = "index_0_100"

# WHY: these weights are an explicit, documented engineering choice -- not a
# fitted model. They are renormalized over whichever components are actually
# measurable for a session, so a missing component never silently counts as 0.
_WEIGHTS = {
    "valid_rep_ratio": 0.40,
    "tracking_quality": 0.40,
    "tempo_consistency": 0.20,
}

_TEMPO_FIELD = "tempo_ratio_ecc_over_con"
_VISIBILITY_FIELD = "primary_joints_min_visibility"


@dataclass
class FormIndex:
    """A relative, uncalibrated leaderboard index -- NOT a validated form grade."""

    value: float | None
    unit: str
    status: str  # "valid" | "degraded" | "unknown"
    reason_codes: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)


def _measured(feature: dict | None) -> float | None:
    """Return a feature's value only when it is a measured (``status="valid"``) number."""
    if not feature or feature.get("status") != "valid":
        return None
    v = feature.get("value")
    return float(v) if isinstance(v, (int, float)) else None


def compute_form_index(feature_vectors: list[dict]) -> FormIndex:
    """Build the leaderboard index from a response's ``feature_vectors``.

    Args:
        feature_vectors: list of per-rep dicts, each with ``rep_status`` and a
            ``features`` map of ``{name: {value, unit, status, reason_codes}}``.

    Returns:
        A :class:`FormIndex`. ``value`` is ``None`` with ``status="unknown"``
        when no rep was cleanly measured.
    """
    reason_codes = ["uncalibrated_demo_index"]
    n_total = len(feature_vectors)
    valid_reps = [fv for fv in feature_vectors if fv.get("rep_status") == "valid"]
    n_valid = len(valid_reps)

    if n_valid == 0:
        reason_codes.append("no_valid_reps")
        return FormIndex(None, FORM_INDEX_UNIT, "unknown", reason_codes, {})

    components: dict[str, float] = {}

    # 1. valid_rep_ratio -- fraction of segmented reps that came out clean.
    components["valid_rep_ratio"] = n_valid / n_total if n_total else 0.0

    # 2. tracking_quality -- mean min-joint-visibility over valid reps, counting
    #    ONLY measured visibility fields (status=valid).
    vis_values = [
        v
        for fv in valid_reps
        if (v := _measured(fv.get("features", {}).get(_VISIBILITY_FIELD))) is not None
    ]
    if vis_values:
        components["tracking_quality"] = max(0.0, min(1.0, statistics.fmean(vis_values)))
    else:
        reason_codes.append("tracking_quality_unavailable")

    # 3. tempo_consistency -- 1 - coefficient of variation of tempo ratio across
    #    valid reps. Consistency is a within-session measure that needs NO
    #    reference band, so it is contract-safe. Requires >=2 measured reps.
    tempo_values = [
        v
        for fv in valid_reps
        if (v := _measured(fv.get("features", {}).get(_TEMPO_FIELD))) is not None
    ]
    if len(tempo_values) >= 2:
        mean_t = statistics.fmean(tempo_values)
        if mean_t > 0:
            cv = statistics.pstdev(tempo_values) / mean_t
            components["tempo_consistency"] = max(0.0, 1.0 - min(1.0, cv))
        else:
            reason_codes.append("tempo_consistency_unavailable_zero_mean")
    else:
        reason_codes.append("tempo_consistency_unavailable_single_rep")

    # Weighted blend, renormalized over the components we could actually measure.
    avail_w = {k: _WEIGHTS[k] for k in components}
    total_w = sum(avail_w.values())
    value = (
        100.0 * sum(avail_w[k] * components[k] for k in components) / total_w
        if total_w
        else None
    )

    # Status is valid only when every component was measurable; degraded otherwise.
    status = "valid" if len(components) == len(_WEIGHTS) else "degraded"

    return FormIndex(
        value=round(value, 1) if value is not None else None,
        unit=FORM_INDEX_UNIT,
        status=status,
        reason_codes=reason_codes,
        components={k: round(v, 4) for k, v in components.items()},
    )
