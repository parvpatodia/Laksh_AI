"""Persistence record models and the envelope -> record builder.

A :class:`SessionRecord` is what gets persisted per analysis: the provenance
(incl. ``git_commit_sha``), rep counts, the leaderboard ``form_index``, and a
``fingerprint`` of the MEASURED per-rep features. The builder is the single
place that turns a v1 response envelope into that record.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.persistence.scoring import compute_form_index

#: Per-rep scalar features captured in the biomechanical fingerprint. These are
#: the gym measurement-spine fields; only those with status=valid are stored.
_FINGERPRINT_FEATURE_KEYS = (
    "rep_duration_s",
    "eccentric_duration_s",
    "concentric_duration_s",
    "tempo_ratio_ecc_over_con",
    "signal_amplitude",
    "primary_joints_min_visibility",
)


class RepFingerprint(BaseModel):
    """Measured (status=valid) scalar features for one rep."""

    model_config = ConfigDict(extra="forbid")

    rep_index: int
    rep_status: str
    measured: dict[str, float] = Field(default_factory=dict)


class SessionRecord(BaseModel):
    """One persisted analysis session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    created_at: str
    sport_id: str
    exercise_id: str
    display_name: str = "anon"
    git_commit_sha: Optional[str] = None
    pose_baseline_version: str = ""
    model: str = ""
    source: str = ""
    fps: float = 0.0
    n_frames: int = 0
    n_reps: int = 0
    n_valid_reps: int = 0
    form_index: Optional[float] = None
    form_index_status: str = "unknown"
    form_index_reason_codes: list[str] = Field(default_factory=list)
    form_index_components: dict[str, float] = Field(default_factory=dict)
    fingerprint: list[RepFingerprint] = Field(default_factory=list)


class LeaderboardEntry(BaseModel):
    """One ranked row served by ``GET /v1/leaderboard``."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    session_id: str
    display_name: str
    exercise_id: str
    form_index: float
    form_index_status: str
    n_valid_reps: int
    n_reps: int
    created_at: str
    git_commit_sha: Optional[str] = None


def _measured_values(features: dict[str, Any]) -> dict[str, float]:
    """Pick out status=valid numeric feature values for the fingerprint."""
    out: dict[str, float] = {}
    for key in _FINGERPRINT_FEATURE_KEYS:
        f = features.get(key)
        if f and f.get("status") == "valid" and isinstance(f.get("value"), (int, float)):
            out[key] = float(f["value"])
    return out


def build_session_record(
    envelope: dict[str, Any],
    display_name: str = "anon",
    session_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> SessionRecord:
    """Turn a v1 analyze response envelope into a persistable record.

    Args:
        envelope: an ``AnalyzeResponseModel``-shaped dict.
        display_name: leaderboard display name; falls back to ``"anon"``.
        session_id / created_at: injectable for deterministic tests.
    """
    fvs: list[dict[str, Any]] = envelope.get("feature_vectors", []) or []
    prov: dict[str, Any] = envelope.get("provenance", {}) or {}
    fi = compute_form_index(fvs)

    fingerprint = [
        RepFingerprint(
            rep_index=fv.get("rep_index", i),
            rep_status=fv.get("rep_status", "unknown"),
            measured=_measured_values(fv.get("features", {}) or {}),
        )
        for i, fv in enumerate(fvs)
    ]

    return SessionRecord(
        session_id=session_id or uuid.uuid4().hex,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        sport_id=envelope.get("sport_id", "gym"),
        exercise_id=envelope.get("exercise_id", ""),
        display_name=(display_name or "anon").strip() or "anon",
        git_commit_sha=prov.get("git_commit_sha"),
        pose_baseline_version=prov.get("pose_baseline_version", ""),
        model=prov.get("model", ""),
        source=envelope.get("source", ""),
        fps=float(envelope.get("fps", 0.0) or 0.0),
        n_frames=int(envelope.get("n_frames", 0) or 0),
        n_reps=len(fvs),
        n_valid_reps=sum(1 for fv in fvs if fv.get("rep_status") == "valid"),
        form_index=fi.value,
        form_index_status=fi.status,
        form_index_reason_codes=fi.reason_codes,
        form_index_components=fi.components,
        fingerprint=fingerprint,
    )
