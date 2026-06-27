"""Library-level gym clip analysis pipeline.

This module is the single source of truth for running a captured set of
canonical-joint frames through the gym measurement spine:

    canonical_frames
        |
        v  extract_rep_signal  (per-exercise 1D signal)
    rep signal + missingness mask
        |
        v  segment_reps        (scipy.signal.find_peaks)
    SegmentResult  (RepSpan[] tagged valid | degraded | unknown)
        |
        v  feature_vectors_from_segment
    RepFeatureVector[]  (7 FieldValues per rep)
        |
        v  apply_calibration   (honest uncalibrated_v0 or cited bands)
    per-rep calibration block

Both ``scripts/analyze_gym_clip.py`` (CLI) and ``app.api.v1.analyze``
(HTTP) import :func:`analyze_gym_clip` so there is exactly one
orchestration path. Keeping it a pure function (no I/O beyond reading
the calibration JSON) means the HTTP endpoint does not need a
subprocess and remains deterministic in tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.gym.calibration_v0 import (
    CalibrationManifest,
    apply_calibration,
    load_calibration_v0,
)
from app.gym.exercises_v0 import get_exercise, validate_exercise_id
from app.gym.rep_features import (
    RepFeaturesConfig,
    extract_rep_signal,
    feature_vectors_from_segment,
)
from app.gym.rep_segmenter import SegmenterConfig, segment_reps

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Schema version emitted in :func:`analyze_gym_clip` results.
#: Bump when field names or units at the top level change.
GYM_PIPELINE_SCHEMA_VERSION = "1.0.0"

DEFAULT_CALIBRATION_CONFIG = REPO_ROOT / "evaluation" / "gym_calibration_v0.json"


class UnknownExerciseError(ValueError):
    """Raised when ``exercise_id`` is not in the frozen v0 taxonomy."""


def _manifest_for(
    calibration_path: Path | None,
    cached: CalibrationManifest | None,
) -> CalibrationManifest:
    """Return the cached manifest if provided; else load from disk."""
    if cached is not None:
        return cached
    path = calibration_path or DEFAULT_CALIBRATION_CONFIG
    return load_calibration_v0(path)


def analyze_gym_clip(
    *,
    exercise_id: str,
    fps: float,
    canonical_frames: list,
    source: str = "frames_json",
    calibration_path: Path | None = None,
    calibration_manifest: CalibrationManifest | None = None,
    seg_config: SegmenterConfig | None = None,
    feat_config: RepFeaturesConfig | None = None,
) -> dict[str, Any]:
    """Run the full gym measurement spine on ``canonical_frames``.

    Parameters
    ----------
    exercise_id:
        Must be a key in :mod:`app.gym.exercises_v0.EXERCISES_V0`.
    fps:
        Sampling frame rate of ``canonical_frames``. Must be > 0.
    canonical_frames:
        List of per-frame joint dicts (or ``None``) as produced by
        :func:`app.gym.pose_adapter.frames_json_to_canonical_frames`
        or :func:`app.gym.pose_adapter.extract_canonical_frames`.
    source:
        Provenance tag -- one of ``"frames_json"``, ``"video"``,
        ``"webcam_capture"``. Surfaced verbatim in the result.
    calibration_path:
        Optional override for the JSON manifest. Ignored when
        ``calibration_manifest`` is supplied.
    calibration_manifest:
        Optional pre-loaded manifest (useful for HTTP handlers that
        want to cache the manifest across requests).
    seg_config / feat_config:
        Optional overrides for segmenter / feature-extractor knobs.

    Returns
    -------
    dict
        The result JSON matching the v1 gym schema (:data:`GYM_PIPELINE_SCHEMA_VERSION`).

    Raises
    ------
    UnknownExerciseError
        ``exercise_id`` is not in the frozen v0 taxonomy.
    ValueError
        ``fps`` is not positive.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    err = validate_exercise_id(exercise_id)
    if err:
        raise UnknownExerciseError(err)
    exercise = get_exercise(exercise_id)
    if exercise is None:
        raise UnknownExerciseError(
            f"reserved token {exercise_id!r} is not analysable"
        )

    signal, _miss = extract_rep_signal(canonical_frames, exercise)
    seg_result = segment_reps(
        signal=signal,
        fps=fps,
        exercise=exercise,
        config=seg_config,
    )
    feature_vectors = feature_vectors_from_segment(
        segment=seg_result,
        canonical_frames=canonical_frames,
        exercise=exercise,
        config=feat_config,
    )

    manifest = _manifest_for(calibration_path, calibration_manifest)
    cal_entry = manifest.get(exercise_id)

    per_rep_cal: list[dict[str, Any]] = []
    if cal_entry is not None:
        for fv in feature_vectors:
            per_rep_cal.append(
                {
                    "rep_index": fv.rep_index,
                    "fields": apply_calibration(cal_entry, fv),
                }
            )
    cal_block: dict[str, Any] = {
        "exercise_id": exercise_id,
        "evidence_status": cal_entry.evidence_status if cal_entry else "no_config",
        "evidence_source": cal_entry.evidence_source if cal_entry else None,
        "comparable_fields": list(cal_entry.comparable_fields) if cal_entry else [],
        "per_rep": per_rep_cal,
    }

    return {
        "schema_version": GYM_PIPELINE_SCHEMA_VERSION,
        "exercise_id": exercise_id,
        "source": source,
        "fps": fps,
        "n_frames": len(canonical_frames),
        "segment": seg_result.to_dict(),
        "feature_vectors": [fv.to_dict() for fv in feature_vectors],
        "calibration": cal_block,
    }
