"""Realtime-vs-canonical parity probe (ADR 0004).

Compares a ``realtime_preview`` feature-vector dict against a
``canonical_backend`` feature-vector dict and returns the ``parity_probe``
block that is embedded in the v1 response envelope.

Design invariants
-----------------
* Only fields where **both** sides carry ``status="valid"`` and a non-None
  numeric value are included in the comparison.  Degraded / unknown fields
  are skipped -- the same policy as the ADR 0002 offline gate.
* If fewer than :data:`MIN_COMPARABLE_FIELDS` pairs survive the filter, the
  probe returns ``status="insufficient_data"`` rather than a potentially
  misleading statistic.
* Tolerance thresholds default to p90 <= 0.15 (absolute units) and
  max <= 0.50. These are conservative for v0: we have no calibrated
  reference yet, so the probe reports agreement / disagreement without
  claiming what the "correct" delta should be.
* The function is pure and deterministic -- no I/O, no side effects --
  so it can be tested without a running server.
"""
from __future__ import annotations

from typing import Any

import numpy as np

#: Minimum number of valid comparable field pairs to emit a numeric result.
MIN_COMPARABLE_FIELDS: int = 2

#: p90 absolute-delta threshold for ``within_tolerance``.
DEFAULT_P90_TOLERANCE: float = 0.15

#: Maximum absolute-delta threshold for ``within_tolerance``.
DEFAULT_MAX_TOLERANCE: float = 0.50


def compare_feature_vectors(
    realtime: dict[str, dict[str, Any]],
    canonical: dict[str, dict[str, Any]],
    *,
    p90_tolerance: float = DEFAULT_P90_TOLERANCE,
    max_tolerance: float = DEFAULT_MAX_TOLERANCE,
) -> dict[str, Any]:
    """Compare a realtime ghost feature dict against a canonical one.

    Parameters
    ----------
    realtime:
        ``{field_name: {"value": float|None, "status": str, ...}}``
        emitted by the browser-side rep counter (analysis_mode=realtime_preview).
    canonical:
        Same shape, emitted by the server-side gym pipeline
        (analysis_mode=canonical_backend).
    p90_tolerance:
        Absolute-delta threshold at the 90th percentile for
        ``within_tolerance`` classification.
    max_tolerance:
        Absolute-delta threshold at the maximum for
        ``within_tolerance`` classification.

    Returns
    -------
    dict matching :class:`app.api.v1.schema.ParityProbeModel`::

        {
            "fields_compared": ["rep_duration_s", "eccentric_duration_s"],
            "max_abs_delta": 0.08,
            "p90_abs_delta": 0.05,
            "status": "within_tolerance",
        }
    """
    deltas: list[float] = []
    compared_fields: list[str] = []

    for field, rt_entry in realtime.items():
        if field not in canonical:
            continue
        can_entry = canonical[field]
        rt_val = rt_entry.get("value")
        can_val = can_entry.get("value")
        # Skip missing values.
        if rt_val is None or can_val is None:
            continue
        # Skip non-valid status on either side.
        if rt_entry.get("status") != "valid" or can_entry.get("status") != "valid":
            continue
        deltas.append(abs(float(rt_val) - float(can_val)))
        compared_fields.append(field)

    compared_fields = sorted(compared_fields)

    if len(compared_fields) < MIN_COMPARABLE_FIELDS:
        return {
            "fields_compared": compared_fields,
            "max_abs_delta": 0.0,
            "p90_abs_delta": 0.0,
            "status": "insufficient_data",
        }

    arr = np.array(deltas, dtype=float)
    max_delta = float(arr.max())
    p90_delta = float(np.percentile(arr, 90))

    within = p90_delta <= p90_tolerance and max_delta <= max_tolerance
    status = "within_tolerance" if within else "outside_tolerance"

    return {
        "fields_compared": compared_fields,
        "max_abs_delta": round(max_delta, 6),
        "p90_abs_delta": round(p90_delta, 6),
        "status": status,
    }


def probe_reps(
    realtime_vectors: list[dict[str, Any]],
    canonical_vectors: list[dict[str, Any]],
    *,
    p90_tolerance: float = DEFAULT_P90_TOLERANCE,
    max_tolerance: float = DEFAULT_MAX_TOLERANCE,
) -> dict[str, Any]:
    """Aggregate parity probe across multiple rep vectors.

    Matches reps by ``rep_index``.  Only rep pairs present on both sides are
    included.  All per-field deltas are pooled before computing the p90 / max
    statistics, which gives a single aggregate ``parity_probe`` block for the
    full clip rather than one block per rep.

    Parameters
    ----------
    realtime_vectors:
        List of ``{"rep_index": int, "features": {field: FieldValueModel-like}}``
        from the realtime path.
    canonical_vectors:
        Same shape from the canonical backend path.

    Returns
    -------
    dict matching :class:`app.api.v1.schema.ParityProbeModel`.
    """
    # Index canonical by rep_index for O(1) lookup.
    can_by_idx: dict[int, dict[str, Any]] = {
        v["rep_index"]: v["features"] for v in canonical_vectors
    }

    all_deltas: list[float] = []
    field_set: set[str] = set()

    for rt_rep in realtime_vectors:
        idx = rt_rep["rep_index"]
        if idx not in can_by_idx:
            continue
        rt_feats = rt_rep.get("features", {})
        can_feats = can_by_idx[idx]

        for field, rt_entry in rt_feats.items():
            if field not in can_feats:
                continue
            can_entry = can_feats[field]
            rt_val = rt_entry.get("value")
            can_val = can_entry.get("value")
            if rt_val is None or can_val is None:
                continue
            if rt_entry.get("status") != "valid" or can_entry.get("status") != "valid":
                continue
            all_deltas.append(abs(float(rt_val) - float(can_val)))
            field_set.add(field)

    compared_fields = sorted(field_set)

    if len(compared_fields) < MIN_COMPARABLE_FIELDS:
        return {
            "fields_compared": compared_fields,
            "max_abs_delta": 0.0,
            "p90_abs_delta": 0.0,
            "status": "insufficient_data",
        }

    arr = np.array(all_deltas, dtype=float)
    max_delta = float(arr.max())
    p90_delta = float(np.percentile(arr, 90))

    within = p90_delta <= p90_tolerance and max_delta <= max_tolerance
    status = "within_tolerance" if within else "outside_tolerance"

    return {
        "fields_compared": compared_fields,
        "max_abs_delta": round(max_delta, 6),
        "p90_abs_delta": round(p90_delta, 6),
        "status": status,
    }
