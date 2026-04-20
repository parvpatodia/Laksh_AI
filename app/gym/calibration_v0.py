"""Versioned reference-range calibration config v0 (GOALS.md Milestone 1, bullet 4).

GOALS.md calibration policy (verbatim): *"No new silent hardcoded ideal angles
in code; reference ranges come from documented sources ... live in versioned
config committed with the eval run that justified them"*.

This module is the **config-driven replacement** for hardcoded ideal bands.
The source of truth is the JSON file at ``evaluation/gym_calibration_v0.json``
(versioned, reviewable, SHA-pinnable); this Python module is a typed loader +
validator + a pure ``apply_calibration`` function.

v0 honesty contract
-------------------
Because no labeled ground-truth subset exists yet (that arrives with
Milestone 2), every entry in the shipped v0 config MUST carry:

* ``evidence_status = "uncalibrated_v0"``
* ``reference_ranges = {}``  (empty)
* ``evidence_source = None``

An entry with ``evidence_status = "cited"`` is only accepted when it also
supplies ``evidence_source`` (e.g. a scorecard path + row hash). This is
enforced by :meth:`CalibrationEntry.__post_init__` so a future edit cannot
silently regress the policy. The validator is the **policy in code**.

``apply_calibration`` returns per-field reference statuses:

* ``no_reference_yet`` — config has no band for this field (v0 default)
* ``unavailable``     — the underlying :class:`FieldValue` is unknown/degraded
* ``within_reference``  — value inside the cited band
* ``outside_reference`` — value outside the cited band

Downstream UI / scorecards SHOULD surface ``no_reference_yet`` as explicit
"uncalibrated — awaiting labeled subset" text rather than hide the field.
That is the whole point: the system should be honest about what it hasn't
measured yet.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.gym.exercises_v0 import EXERCISES_V0

CALIBRATION_V0_SCHEMA_VERSION = "1.0.0"
# v0.2.0 (2026-04-19): populated literature-cited reference_ranges for every
# exercise (NSCA Essentials of Strength Training & Conditioning 4e (2016) +
# ACSM Guidelines 11e (2022) tempo prescriptions, plus per-exercise ROM
# literature). Added entry for `dumbbell_bicep_curl` (registry v0.2.0).
# Synthetic cohort percentile bands shipped alongside as a separate
# validation artifact (see evaluation/synthetic_cohort_v0/) — those are
# NOT used as the cited evidence_source; literature is.
CALIBRATION_V0_MANIFEST_VERSION = "v0.2.0"

# Allowed values for the per-entry ``evidence_status`` field. New statuses
# require a schema bump + a downstream scorecard update.
ALLOWED_EVIDENCE_STATUSES: frozenset[str] = frozenset(
    {"uncalibrated_v0", "cited"}
)

# Fields eligible to carry reference ranges. MUST stay in sync with field
# names emitted by :mod:`app.gym.rep_features`. Listed explicitly (not
# introspected) so an accidental rename in ``rep_features.py`` trips a test.
COMPARABLE_FIELDS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "rep_duration_s",
        "eccentric_duration_s",
        "concentric_duration_s",
        "tempo_ratio_ecc_over_con",
        "signal_amplitude",
        "primary_joints_min_visibility",
        "primary_joints_missing_frac",
    }
)


@dataclass(frozen=True)
class CalibrationEntry:
    """One exercise's reference-range entry. Immutable.

    ``reference_ranges`` is a mapping ``field_name -> (lo, hi)`` (inclusive
    low, inclusive high). An empty mapping means "no reference yet".
    """

    exercise_id: str
    comparable_fields: tuple[str, ...]
    reference_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    evidence_status: str = "uncalibrated_v0"
    evidence_source: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.exercise_id not in EXERCISES_V0:
            raise ValueError(
                f"unknown exercise_id {self.exercise_id!r}; must be in EXERCISES_V0"
            )
        if self.evidence_status not in ALLOWED_EVIDENCE_STATUSES:
            raise ValueError(
                f"{self.exercise_id}: evidence_status {self.evidence_status!r} "
                f"not in {sorted(ALLOWED_EVIDENCE_STATUSES)}"
            )
        for fname in self.comparable_fields:
            if fname not in COMPARABLE_FIELDS_ALLOWLIST:
                raise ValueError(
                    f"{self.exercise_id}: comparable_fields entry {fname!r} "
                    f"not in {sorted(COMPARABLE_FIELDS_ALLOWLIST)}"
                )
        if len(set(self.comparable_fields)) != len(self.comparable_fields):
            raise ValueError(
                f"{self.exercise_id}: comparable_fields must be unique"
            )
        for fname, rng in self.reference_ranges.items():
            if fname not in COMPARABLE_FIELDS_ALLOWLIST:
                raise ValueError(
                    f"{self.exercise_id}: reference_ranges key {fname!r} "
                    f"not in {sorted(COMPARABLE_FIELDS_ALLOWLIST)}"
                )
            if fname not in self.comparable_fields:
                raise ValueError(
                    f"{self.exercise_id}: reference_ranges field {fname!r} "
                    "must also be listed in comparable_fields"
                )
            if not isinstance(rng, tuple) or len(rng) != 2:
                raise ValueError(
                    f"{self.exercise_id}: reference_ranges[{fname!r}] must be a (lo, hi) tuple"
                )
            lo, hi = rng
            if not (isinstance(lo, int | float) and isinstance(hi, int | float)):
                raise ValueError(
                    f"{self.exercise_id}: reference_ranges[{fname!r}] must be numeric"
                )
            if not (lo < hi):
                raise ValueError(
                    f"{self.exercise_id}: reference_ranges[{fname!r}] requires lo < hi, got ({lo}, {hi})"
                )
        # Policy guards — the hard part of the calibration contract.
        if self.evidence_status == "uncalibrated_v0":
            if self.reference_ranges:
                raise ValueError(
                    f"{self.exercise_id}: evidence_status 'uncalibrated_v0' "
                    "forbids non-empty reference_ranges (GOALS.md calibration policy)"
                )
            if self.evidence_source is not None:
                raise ValueError(
                    f"{self.exercise_id}: 'uncalibrated_v0' must leave evidence_source=None"
                )
        if self.evidence_status == "cited":
            if not self.evidence_source or not self.evidence_source.strip():
                raise ValueError(
                    f"{self.exercise_id}: evidence_status 'cited' requires a "
                    "non-empty evidence_source (scorecard path / row hash)"
                )
            if not self.reference_ranges:
                raise ValueError(
                    f"{self.exercise_id}: 'cited' must carry at least one reference_range"
                )


@dataclass(frozen=True)
class CalibrationManifest:
    """Parsed, validated calibration config. Immutable.

    The full manifest level carries schema/manifest versions and an optional
    top-level ``evidence_source`` (e.g. the scorecard this config was derived
    from). The per-entry ``evidence_source`` takes precedence when set.
    """

    schema_version: str
    manifest_version: str
    evidence_source: str | None
    entries: dict[str, CalibrationEntry]

    def get(self, exercise_id: str) -> CalibrationEntry | None:
        return self.entries.get(exercise_id)


# ---------- loading / validation ------------------------------------------


def _entry_from_dict(d: Mapping[str, Any]) -> CalibrationEntry:
    """Parse one entry dict into a :class:`CalibrationEntry`.

    Accepts JSON-friendly types (list for tuples). Validation via
    ``__post_init__`` then enforces the policy contract.
    """
    ranges_raw = d.get("reference_ranges") or {}
    if not isinstance(ranges_raw, Mapping):
        raise ValueError(
            f"entry {d.get('exercise_id')!r}: reference_ranges must be a mapping"
        )
    ranges: dict[str, tuple[float, float]] = {}
    for k, v in ranges_raw.items():
        if not isinstance(v, list | tuple) or len(v) != 2:
            raise ValueError(
                f"entry {d.get('exercise_id')!r}: reference_ranges[{k!r}] must be [lo, hi]"
            )
        ranges[k] = (float(v[0]), float(v[1]))
    fields_raw = d.get("comparable_fields") or []
    if not isinstance(fields_raw, list | tuple):
        raise ValueError(
            f"entry {d.get('exercise_id')!r}: comparable_fields must be a list"
        )
    return CalibrationEntry(
        exercise_id=str(d["exercise_id"]),
        comparable_fields=tuple(str(x) for x in fields_raw),
        reference_ranges=ranges,
        evidence_status=str(d.get("evidence_status", "uncalibrated_v0")),
        evidence_source=(
            None if d.get("evidence_source") in (None, "") else str(d["evidence_source"])
        ),
        notes=str(d.get("notes") or ""),
    )


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != CALIBRATION_V0_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: got {payload.get('schema_version')!r}, "
            f"expected {CALIBRATION_V0_SCHEMA_VERSION!r}"
        )
    if not isinstance(payload.get("exercises"), list):
        raise ValueError("payload must contain an 'exercises' list")


def load_calibration_v0_from_dict(payload: Mapping[str, Any]) -> CalibrationManifest:
    """Build a :class:`CalibrationManifest` from an already-parsed dict."""
    _validate_payload(payload)
    entries: dict[str, CalibrationEntry] = {}
    for raw in payload["exercises"]:
        entry = _entry_from_dict(raw)
        if entry.exercise_id in entries:
            raise ValueError(f"duplicate exercise_id {entry.exercise_id!r}")
        entries[entry.exercise_id] = entry
    # Coverage guard: config MUST list every frozen v0 exercise. Missing
    # entries would silently ship as "no calibration known" with no way to
    # tell the difference from "deliberately empty" — force explicit listing.
    missing = sorted(set(EXERCISES_V0.keys()) - set(entries.keys()))
    if missing:
        raise ValueError(
            f"calibration config missing entries for {missing}; every exercise "
            "in EXERCISES_V0 must be listed (use uncalibrated_v0 if no evidence)"
        )
    extra = sorted(set(entries.keys()) - set(EXERCISES_V0.keys()))
    if extra:
        raise ValueError(
            f"calibration config has unknown exercise_ids {extra}; "
            "every id must be in EXERCISES_V0"
        )
    top_src = payload.get("evidence_source")
    return CalibrationManifest(
        schema_version=str(payload["schema_version"]),
        manifest_version=str(payload.get("manifest_version") or CALIBRATION_V0_MANIFEST_VERSION),
        evidence_source=None if top_src in (None, "") else str(top_src),
        entries=entries,
    )


def load_calibration_v0(path: Path) -> CalibrationManifest:
    """Load + validate a calibration JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return load_calibration_v0_from_dict(raw)


# ---------- apply_calibration ---------------------------------------------


def apply_calibration(
    entry: CalibrationEntry,
    feature_vector: Any,
) -> dict[str, dict[str, Any]]:
    """Return per-field reference status given a :class:`RepFeatureVector`.

    Output shape::

        {
          "rep_duration_s": {
            "status": "no_reference_yet" | "unavailable"
                      | "within_reference" | "outside_reference",
            "range": [lo, hi] | None,
            "value": float | None,
            "evidence_status": "uncalibrated_v0" | "cited",
            "evidence_source": str | None,
          },
          ...
        }

    The function never raises on missing fields — it emits ``unavailable``
    so downstream reporting stays honest without try/except.
    """
    out: dict[str, dict[str, Any]] = {}
    features = getattr(feature_vector, "features", None)
    if features is None and isinstance(feature_vector, Mapping):
        features = feature_vector.get("features")
    features = features or {}
    for fname in entry.comparable_fields:
        fv = features.get(fname)
        value = getattr(fv, "value", None) if fv is not None else None
        status = getattr(fv, "status", None) if fv is not None else None
        # If the measurement itself is unknown, skip range comparison.
        if value is None or status in (None, "unknown"):
            out[fname] = {
                "status": "unavailable",
                "range": None,
                "value": None if value is None else float(value),
                "evidence_status": entry.evidence_status,
                "evidence_source": entry.evidence_source,
            }
            continue
        rng = entry.reference_ranges.get(fname)
        if rng is None:
            out[fname] = {
                "status": "no_reference_yet",
                "range": None,
                "value": float(value),
                "evidence_status": entry.evidence_status,
                "evidence_source": entry.evidence_source,
            }
            continue
        lo, hi = rng
        inside = lo <= float(value) <= hi
        out[fname] = {
            "status": "within_reference" if inside else "outside_reference",
            "range": [lo, hi],
            "value": float(value),
            "evidence_status": entry.evidence_status,
            "evidence_source": entry.evidence_source,
        }
    return out


# ---------- manifest serialisation + SHA ----------------------------------


def _entry_to_dict(e: CalibrationEntry) -> dict[str, Any]:
    return {
        "exercise_id": e.exercise_id,
        "comparable_fields": list(e.comparable_fields),
        # Serialise tuples as [lo, hi] so JSON round-trips cleanly.
        "reference_ranges": {k: [v[0], v[1]] for k, v in e.reference_ranges.items()},
        "evidence_status": e.evidence_status,
        "evidence_source": e.evidence_source,
        "notes": e.notes,
    }


def manifest_to_dict(m: CalibrationManifest) -> dict[str, Any]:
    """Deterministic JSON-friendly representation of a manifest."""
    return {
        "schema_version": m.schema_version,
        "manifest_version": m.manifest_version,
        "evidence_source": m.evidence_source,
        "exercises": [_entry_to_dict(m.entries[k]) for k in sorted(m.entries.keys())],
    }


def compute_manifest_sha256(m: CalibrationManifest) -> str:
    """SHA-256 over canonical JSON of the manifest (sort_keys for stability)."""
    payload = json.dumps(manifest_to_dict(m), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
