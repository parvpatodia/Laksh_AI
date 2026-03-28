"""
Versioned, on-disk calibration for gym pose baseline heuristics.

Thresholds live in evaluation/gym_pose_calibration.json so product and research can
adjust gates without code edits; provenance records which file (and hash) was used.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


DEFAULT_CALIBRATION_PATH = _repo_root() / "evaluation" / "gym_pose_calibration.json"


@dataclass(frozen=True)
class GymPoseUsableGate:
    """Provisional 'usable for kinematics v0' gate; not a clinical or mocap standard."""

    min_detection_rate: float
    min_visibility_core_when_detected: float
    min_n_frames: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "min_detection_rate": self.min_detection_rate,
            "min_visibility_core_when_detected": self.min_visibility_core_when_detected,
            "min_n_frames": self.min_n_frames,
        }


_BUILTIN = GymPoseUsableGate(
    min_detection_rate=0.25,
    min_visibility_core_when_detected=0.35,
    min_n_frames=15,
)


def _validate_gate_values(d: dict[str, Any]) -> GymPoseUsableGate:
    md = float(d["min_detection_rate"])
    mv = float(d["min_visibility_core_when_detected"])
    mn = int(d["min_n_frames"])
    if not (0.0 <= md <= 1.0):
        raise ValueError(f"min_detection_rate must be in [0,1], got {md}")
    if not (0.0 <= mv <= 1.0):
        raise ValueError(f"min_visibility_core_when_detected must be in [0,1], got {mv}")
    if mn < 1:
        raise ValueError(f"min_n_frames must be >= 1, got {mn}")
    return GymPoseUsableGate(min_detection_rate=md, min_visibility_core_when_detected=mv, min_n_frames=mn)


def _path_for_record(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(_repo_root().resolve()))
    except ValueError:
        return str(p.resolve())


def load_gym_pose_usable_gate(
    path: Path | None = None,
) -> tuple[GymPoseUsableGate, dict[str, Any]]:
    """
    Load gate from JSON; fall back to built-in defaults if file is missing or invalid.

    Returns:
        (gate, calibration_record) — calibration_record is merged into pose provenance.
    """
    p = path or DEFAULT_CALIBRATION_PATH
    if not p.is_file():
        logger.info("No %s; using built-in pose_usable_gate defaults", p)
        return _BUILTIN, {
            "calibration_source": "builtin_defaults",
            "calibration_path": None,
            "calibration_file_sha256": None,
            "calibration_schema_version": None,
        }

    raw = p.read_text(encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s — using builtin gate", p, e)
        return _BUILTIN, {
            "calibration_source": "builtin_defaults_fallback_invalid_json",
            "calibration_path": _path_for_record(p),
            "calibration_file_sha256": digest,
            "calibration_schema_version": None,
        }

    if not isinstance(data, dict):
        logger.error("Calibration root must be a JSON object, got %s — using builtin gate", type(data).__name__)
        return _BUILTIN, {
            "calibration_source": "builtin_defaults_fallback_invalid_root_type",
            "calibration_path": _path_for_record(p),
            "calibration_file_sha256": digest,
            "calibration_schema_version": None,
        }

    try:
        section = data.get("pose_usable_heuristic") or {}
        gate = _validate_gate_values(
            {
                "min_detection_rate": section.get("min_detection_rate", _BUILTIN.min_detection_rate),
                "min_visibility_core_when_detected": section.get(
                    "min_visibility_core_when_detected",
                    _BUILTIN.min_visibility_core_when_detected,
                ),
                "min_n_frames": section.get("min_n_frames", _BUILTIN.min_n_frames),
            }
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.error("Invalid calibration in %s: %s — using builtin gate", p, e)
        return _BUILTIN, {
            "calibration_source": "builtin_defaults_fallback_invalid_schema",
            "calibration_path": _path_for_record(p),
            "calibration_file_sha256": digest,
            "calibration_schema_version": data.get("schema_version"),
        }

    rel = _path_for_record(p)
    return gate, {
        "calibration_source": rel,
        "calibration_path": rel,
        "calibration_file_sha256": digest,
        "calibration_schema_version": data.get("schema_version"),
    }
