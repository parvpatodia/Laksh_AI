"""Pydantic v2 request/response models for the v1 HTTP surface.

The response contract:

* Every measured number carries an explicit ``status`` and ``reason_codes``
  list -- this is the "measurement spine" surfaced over HTTP.
* ``provenance`` ties the result to the exact taxonomy + calibration
  manifest + pose baseline the server was running. A client that cares
  about reproducibility can pin these SHAs.
* ``analysis_mode`` distinguishes ``canonical_backend`` (full pipeline
  run on the server) from ``realtime_preview`` (ghost metrics from the
  browser). Day-1 only emits ``canonical_backend``.
* ``parity_probe`` is optional and only populated once a realtime
  preview has also been recorded for the same capture (Day-2).

Schema version is tracked separately from the gym library version so
the API and the library can evolve at independent cadences.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

#: Bumped when field names or semantics change at the v1 HTTP layer.
#: (Independent of :data:`app.gym.pipeline.GYM_PIPELINE_SCHEMA_VERSION`.)
V1_RESPONSE_SCHEMA_VERSION = "2.0.0"

# Status literals must match the gym measurement spine.
FieldStatus = Literal["valid", "degraded", "unknown"]
RepStatus = Literal["valid", "degraded", "unknown"]
CalibrationFieldStatus = Literal[
    "no_reference_yet",
    "unavailable",
    "within_reference",
    "outside_reference",
]
AnalysisMode = Literal["canonical_backend", "realtime_preview"]
SourceLiteral = Literal["frames_json", "video", "webcam_capture"]
SportId = Literal["basketball", "gym"]


class FieldValueModel(BaseModel):
    """One measured field."""

    model_config = ConfigDict(extra="forbid")

    value: Optional[float] = None
    unit: str
    status: FieldStatus
    reason_codes: list[str] = Field(default_factory=list)


class RepFeaturesModel(BaseModel):
    """Per-rep feature block."""

    model_config = ConfigDict(extra="allow")  # allow future fields

    # We don't freeze keys here: gym uses 7 fields, basketball will use 8.
    # The values are all FieldValueModel.


class RepVectorModel(BaseModel):
    """One rep, with metadata + features."""

    model_config = ConfigDict(extra="forbid")

    rep_index: int
    start_frame: int
    end_frame: int
    peak_frame: int
    rep_status: RepStatus
    features: dict[str, FieldValueModel]


class RepSpanModel(BaseModel):
    """One rep span as emitted by the segmenter."""

    model_config = ConfigDict(extra="forbid")

    start_frame: int
    end_frame: int
    peak_frame: int
    status: RepStatus
    reason_codes: list[str] = Field(default_factory=list)


class SegmentBlockModel(BaseModel):
    """Segmenter output block (matches :class:`app.gym.rep_segmenter.SegmentResult`)."""

    model_config = ConfigDict(extra="allow")

    schema_version: str
    exercise_id: str
    rep_signal_type: str
    rep_signal_joint: Optional[str] = None
    n_frames: int
    fps: float
    status: RepStatus
    reason_codes: list[str] = Field(default_factory=list)
    reps: list[RepSpanModel] = Field(default_factory=list)


class CalibrationFieldModel(BaseModel):
    """Per-field reference-range block."""

    model_config = ConfigDict(extra="forbid")

    status: CalibrationFieldStatus
    range: Optional[list[float]] = None  # [lo, hi] or null
    value: Optional[float] = None
    evidence_status: str  # "uncalibrated_v0" | "cited"
    evidence_source: Optional[str] = None


class CalibrationPerRepModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rep_index: int
    fields: dict[str, CalibrationFieldModel]


class CalibrationBlockModel(BaseModel):
    """Calibration block as emitted by :func:`app.gym.pipeline.analyze_gym_clip`."""

    model_config = ConfigDict(extra="forbid")

    exercise_id: str
    evidence_status: str
    evidence_source: Optional[str] = None
    comparable_fields: list[str] = Field(default_factory=list)
    per_rep: list[CalibrationPerRepModel] = Field(default_factory=list)


class ProvenanceModel(BaseModel):
    """Everything a client needs to reproduce the result offline."""

    model_config = ConfigDict(extra="forbid")

    git_commit_sha: Optional[str] = None
    pose_baseline_version: str
    exercise_manifest_sha: str
    calibration_manifest_sha: str
    calibration_manifest_version: str
    model: str  # e.g. "mediapipe_pose_landmarker_heavy" or "none_frames_json"


class ParityProbeModel(BaseModel):
    """Realtime-vs-canonical parity block (populated when both paths ran)."""

    model_config = ConfigDict(extra="forbid")

    fields_compared: list[str]
    max_abs_delta: float
    p90_abs_delta: float
    status: Literal["within_tolerance", "outside_tolerance", "insufficient_data"]


class AnalyzeResponseModel(BaseModel):
    """The v1 analyze response (same for every sport)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = V1_RESPONSE_SCHEMA_VERSION
    sport_id: SportId
    exercise_id: str
    source: SourceLiteral
    fps: float
    n_frames: int
    analysis_mode: AnalysisMode
    provenance: ProvenanceModel
    segment: SegmentBlockModel
    feature_vectors: list[RepVectorModel]
    calibration: CalibrationBlockModel
    parity_probe: Optional[ParityProbeModel] = None


# ---------- request models -----------------------------------------------


class JointObsModel(BaseModel):
    """One joint in one frame, permissive to match upstream pose output."""

    model_config = ConfigDict(extra="allow")

    x: float
    y: float
    z: Optional[float] = 0.0
    visibility: Optional[float] = 1.0


class AnalyzeGymRequest(BaseModel):
    """Body for ``POST /v1/analyze/gym``.

    ``frames`` matches the CLI ``--frames-json`` format so the same
    fixture (:file:`evaluation/fixtures/demo_squat_frames.json`) works
    over HTTP without conversion.
    """

    model_config = ConfigDict(extra="forbid")

    exercise_id: str = Field(..., description="Key in app.gym.exercises_v0.EXERCISES_V0")
    fps: float = Field(..., gt=0.0)
    frames: list[Optional[dict[str, Any]]] = Field(
        ...,
        description=(
            "Per-frame joint dicts. Each frame is either null or "
            "{joint_name: {x, y, z?, visibility?}}."
        ),
    )
    display_name: Optional[str] = Field(
        None, description="Optional leaderboard display name for this session."
    )


class SportInfoModel(BaseModel):
    """One sport row in ``GET /v1/sports``."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    available: bool
    exercises: list[str] = Field(default_factory=list)  # populated for gym


class HealthResponseModel(BaseModel):
    """``GET /v1/health`` payload."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    v1_schema_version: str
    provenance: ProvenanceModel
