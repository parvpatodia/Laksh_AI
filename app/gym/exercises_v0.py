"""Frozen exercise v0 registry (GOALS.md Milestone 1, bullet 1).

Structural metadata only — no "ideal angle" ranges, no biomechanical priors.
Reference ranges live in separate versioned config tied to eval evidence
(GOALS.md calibration policy: "No new silent hardcoded ideal angles in code;
reference ranges come from documented sources ... live in versioned config
committed with the eval run that justified them").

Purpose of this registry:
  1. Anchor ``exercise_id`` values used in ``evaluation/gym_manifest.csv``
     and future rubric / split-check artifacts.
  2. Give rep-segmentation + per-rep-feature code a typed vocabulary for
     "which joint carries the rep signal on this movement" without the
     segmenter having to enumerate biomech constants.
  3. Provide a SHA-hashable manifest so the scorecard header can pin the
     exact taxonomy version used at eval time.

Joint names come from :class:`app.pose.canonical.CanonicalJointName` to keep
the vocabulary consistent across pose + gym modules.

Any taxonomy change (add / remove / rename) requires bumping
``EXERCISE_V0_MANIFEST_VERSION`` and re-running
``scripts/freeze_exercise_v0.py`` so ``evaluation/exercise_v0_manifest.json``
stays in sync with the in-code source of truth.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from app.pose.canonical import CanonicalJointName

EXERCISE_V0_SCHEMA_VERSION = "1.0.0"
EXERCISE_V0_MANIFEST_VERSION = "v0.1.0"

ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    {
        "squat",
        "hinge",
        "horizontal_push",
        "vertical_push",
        "horizontal_pull",
        "vertical_pull",
        "lunge",
        "isometric_core",
        "carry",
    }
)

ALLOWED_CAMERA_VIEWS: frozenset[str] = frozenset(
    {"side", "front", "dtl", "45_front_offset"}
)

ALLOWED_FRAMINGS: frozenset[str] = frozenset(
    {"full_body", "upper_body", "lower_body"}
)

# How a rep segmenter should look at this movement.
#   cyclic_angle    -> track an interior joint angle (e.g. knee flexion) as a 1D signal
#   cyclic_vertical -> track a joint's y-coord (e.g. hip vertical for squat bottom)
#   duration        -> isometric hold, no rep cycles, emit duration + stability instead
#   gait_cadence    -> locomotion, emit step cadence + symmetry, not "reps"
ALLOWED_REP_SIGNAL_TYPES: frozenset[str] = frozenset(
    {"cyclic_angle", "cyclic_vertical", "duration", "gait_cadence"}
)


@dataclass(frozen=True)
class ExerciseV0:
    """One exercise in the frozen v0 taxonomy. Immutable.

    Fields are deliberately observational (what to watch, how to frame) and
    carry no biomechanical target ranges.
    """

    exercise_id: str
    display_name: str
    category: str
    camera_view_hint: str
    rep_signal_type: str
    rep_signal_joint: str | None  # canonical joint name, or None for duration/gait
    primary_joints: tuple[str, ...]
    framing: str
    camera_instruction: str

    def __post_init__(self) -> None:
        if self.category not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"{self.exercise_id}: category {self.category!r} not in ALLOWED_CATEGORIES"
            )
        if self.camera_view_hint not in ALLOWED_CAMERA_VIEWS:
            raise ValueError(
                f"{self.exercise_id}: camera_view_hint {self.camera_view_hint!r} "
                "not in ALLOWED_CAMERA_VIEWS"
            )
        if self.rep_signal_type not in ALLOWED_REP_SIGNAL_TYPES:
            raise ValueError(
                f"{self.exercise_id}: rep_signal_type {self.rep_signal_type!r} "
                "not in ALLOWED_REP_SIGNAL_TYPES"
            )
        if self.framing not in ALLOWED_FRAMINGS:
            raise ValueError(
                f"{self.exercise_id}: framing {self.framing!r} not in ALLOWED_FRAMINGS"
            )
        # Non-cyclic movements must not name a rep_signal_joint — the segmenter
        # would misinterpret it. Cyclic movements must name one.
        cyclic = self.rep_signal_type in {"cyclic_angle", "cyclic_vertical"}
        if cyclic and self.rep_signal_joint is None:
            raise ValueError(
                f"{self.exercise_id}: cyclic rep_signal_type requires rep_signal_joint"
            )
        if not cyclic and self.rep_signal_joint is not None:
            raise ValueError(
                f"{self.exercise_id}: non-cyclic rep_signal_type must not set rep_signal_joint"
            )
        if not self.primary_joints:
            raise ValueError(f"{self.exercise_id}: primary_joints must be non-empty")
        canonical = {j.value for j in CanonicalJointName}
        for j in self.primary_joints:
            if j not in canonical:
                raise ValueError(
                    f"{self.exercise_id}: primary_joints entry {j!r} not a canonical joint name"
                )
        if self.rep_signal_joint is not None and self.rep_signal_joint not in canonical:
            raise ValueError(
                f"{self.exercise_id}: rep_signal_joint {self.rep_signal_joint!r} "
                "not a canonical joint name"
            )


def _cj(name: CanonicalJointName) -> str:
    """Canonical joint enum -> string (stable across schema versions)."""
    return name.value


_HIPS_KNEES_ANKLES: tuple[str, ...] = (
    _cj(CanonicalJointName.LEFT_HIP),
    _cj(CanonicalJointName.RIGHT_HIP),
    _cj(CanonicalJointName.LEFT_KNEE),
    _cj(CanonicalJointName.RIGHT_KNEE),
    _cj(CanonicalJointName.LEFT_ANKLE),
    _cj(CanonicalJointName.RIGHT_ANKLE),
)

_SHOULDERS_ELBOWS_WRISTS: tuple[str, ...] = (
    _cj(CanonicalJointName.LEFT_SHOULDER),
    _cj(CanonicalJointName.RIGHT_SHOULDER),
    _cj(CanonicalJointName.LEFT_ELBOW),
    _cj(CanonicalJointName.RIGHT_ELBOW),
    _cj(CanonicalJointName.LEFT_WRIST),
    _cj(CanonicalJointName.RIGHT_WRIST),
)


# Ordered tuple so the frozen JSON manifest is deterministic by insertion order.
# Twelve compound movements covering squat / hinge / push / pull / lunge /
# isometric / carry. Within the 8–15 range GOALS.md Milestone 1 calls out.
_EXERCISE_LIST: tuple[ExerciseV0, ...] = (
    ExerciseV0(
        exercise_id="back_squat",
        display_name="Back Squat",
        category="squat",
        camera_view_hint="side",
        rep_signal_type="cyclic_vertical",
        rep_signal_joint=_cj(CanonicalJointName.RIGHT_HIP),
        primary_joints=_HIPS_KNEES_ANKLES,
        framing="full_body",
        camera_instruction="Place phone ~2m to your side, hip-height, lens level. Whole body in frame.",
    ),
    ExerciseV0(
        exercise_id="front_squat",
        display_name="Front Squat",
        category="squat",
        camera_view_hint="side",
        rep_signal_type="cyclic_vertical",
        rep_signal_joint=_cj(CanonicalJointName.RIGHT_HIP),
        primary_joints=_HIPS_KNEES_ANKLES
        + (
            _cj(CanonicalJointName.LEFT_SHOULDER),
            _cj(CanonicalJointName.RIGHT_SHOULDER),
        ),
        framing="full_body",
        camera_instruction="Side view, hip-height, whole body visible; keep bar path in-frame.",
    ),
    ExerciseV0(
        exercise_id="conventional_deadlift",
        display_name="Conventional Deadlift",
        category="hinge",
        camera_view_hint="side",
        rep_signal_type="cyclic_vertical",
        rep_signal_joint=_cj(CanonicalJointName.RIGHT_HIP),
        primary_joints=_HIPS_KNEES_ANKLES
        + (
            _cj(CanonicalJointName.LEFT_SHOULDER),
            _cj(CanonicalJointName.RIGHT_SHOULDER),
        ),
        framing="full_body",
        camera_instruction="Side view, knee-to-hip height, bar and feet both visible at lockout.",
    ),
    ExerciseV0(
        exercise_id="romanian_deadlift",
        display_name="Romanian Deadlift",
        category="hinge",
        camera_view_hint="side",
        rep_signal_type="cyclic_angle",
        rep_signal_joint=_cj(CanonicalJointName.RIGHT_HIP),
        primary_joints=_HIPS_KNEES_ANKLES
        + (
            _cj(CanonicalJointName.LEFT_SHOULDER),
            _cj(CanonicalJointName.RIGHT_SHOULDER),
        ),
        framing="full_body",
        camera_instruction="Side view, hip-height; capture torso angle change through the hinge.",
    ),
    ExerciseV0(
        exercise_id="bench_press",
        display_name="Barbell Bench Press",
        category="horizontal_push",
        camera_view_hint="side",
        rep_signal_type="cyclic_angle",
        rep_signal_joint=_cj(CanonicalJointName.RIGHT_ELBOW),
        primary_joints=_SHOULDERS_ELBOWS_WRISTS,
        framing="upper_body",
        camera_instruction="Side view, bar-height, bench and bar visible end-to-end through the rep.",
    ),
    ExerciseV0(
        exercise_id="overhead_press",
        display_name="Standing Overhead Press",
        category="vertical_push",
        camera_view_hint="side",
        rep_signal_type="cyclic_angle",
        rep_signal_joint=_cj(CanonicalJointName.RIGHT_ELBOW),
        primary_joints=_SHOULDERS_ELBOWS_WRISTS,
        framing="upper_body",
        camera_instruction="Side view, chest-height, head to hips in frame; show full lockout above head.",
    ),
    ExerciseV0(
        exercise_id="barbell_row",
        display_name="Bent-Over Barbell Row",
        category="horizontal_pull",
        camera_view_hint="side",
        rep_signal_type="cyclic_angle",
        rep_signal_joint=_cj(CanonicalJointName.RIGHT_ELBOW),
        primary_joints=_SHOULDERS_ELBOWS_WRISTS
        + (_cj(CanonicalJointName.LEFT_HIP), _cj(CanonicalJointName.RIGHT_HIP)),
        framing="full_body",
        camera_instruction="Side view, hip-height; torso hinge and bar path both in frame.",
    ),
    ExerciseV0(
        exercise_id="pull_up",
        display_name="Pull-Up",
        category="vertical_pull",
        camera_view_hint="front",
        rep_signal_type="cyclic_angle",
        rep_signal_joint=_cj(CanonicalJointName.RIGHT_ELBOW),
        primary_joints=_SHOULDERS_ELBOWS_WRISTS,
        framing="upper_body",
        camera_instruction="Front view, bar slightly above head height; whole torso + bar visible.",
    ),
    ExerciseV0(
        exercise_id="push_up",
        display_name="Push-Up",
        category="horizontal_push",
        camera_view_hint="side",
        rep_signal_type="cyclic_angle",
        rep_signal_joint=_cj(CanonicalJointName.RIGHT_ELBOW),
        primary_joints=_SHOULDERS_ELBOWS_WRISTS
        + (_cj(CanonicalJointName.LEFT_HIP), _cj(CanonicalJointName.RIGHT_HIP)),
        framing="full_body",
        camera_instruction="Side view, ground-level, shoulder-to-ankle in frame for a straight body check.",
    ),
    ExerciseV0(
        exercise_id="walking_lunge",
        display_name="Walking Lunge",
        category="lunge",
        camera_view_hint="side",
        rep_signal_type="cyclic_vertical",
        rep_signal_joint=_cj(CanonicalJointName.RIGHT_KNEE),
        primary_joints=_HIPS_KNEES_ANKLES,
        framing="full_body",
        camera_instruction="Side view, hip-height, pan if needed so full body stays in frame across steps.",
    ),
    ExerciseV0(
        exercise_id="plank",
        display_name="Plank (hold)",
        category="isometric_core",
        camera_view_hint="side",
        rep_signal_type="duration",
        rep_signal_joint=None,
        primary_joints=(
            _cj(CanonicalJointName.LEFT_SHOULDER),
            _cj(CanonicalJointName.RIGHT_SHOULDER),
            _cj(CanonicalJointName.LEFT_HIP),
            _cj(CanonicalJointName.RIGHT_HIP),
            _cj(CanonicalJointName.LEFT_ANKLE),
            _cj(CanonicalJointName.RIGHT_ANKLE),
        ),
        framing="full_body",
        camera_instruction="Side view, ground-level; shoulder-to-ankle in frame for line-of-body check.",
    ),
    ExerciseV0(
        exercise_id="farmer_carry",
        display_name="Farmer Carry",
        category="carry",
        camera_view_hint="side",
        rep_signal_type="gait_cadence",
        rep_signal_joint=None,
        primary_joints=(
            _cj(CanonicalJointName.LEFT_SHOULDER),
            _cj(CanonicalJointName.RIGHT_SHOULDER),
            _cj(CanonicalJointName.LEFT_HIP),
            _cj(CanonicalJointName.RIGHT_HIP),
            _cj(CanonicalJointName.LEFT_ANKLE),
            _cj(CanonicalJointName.RIGHT_ANKLE),
        ),
        framing="full_body",
        camera_instruction="Side view, hip-height, pan to follow; capture at least 6 steady strides.",
    ),
)


def _build_registry() -> dict[str, ExerciseV0]:
    out: dict[str, ExerciseV0] = {}
    for ex in _EXERCISE_LIST:
        if ex.exercise_id in out:
            raise ValueError(f"duplicate exercise_id {ex.exercise_id!r} in v0 registry")
        out[ex.exercise_id] = ex
    return out


EXERCISES_V0: dict[str, ExerciseV0] = _build_registry()

# Reserved tokens that may appear in ``exercise_id`` columns to mean
# "not a single frozen movement" (e.g. mixed-exercise clip in a hard subset).
# Treated as valid by :func:`validate_exercise_id` but NOT part of the frozen
# registry — they carry no schema.
RESERVED_EXERCISE_TOKENS: frozenset[str] = frozenset({"mixed", "unknown"})


def get_exercise(exercise_id: str) -> ExerciseV0 | None:
    """Return the registry entry or ``None`` if unknown."""
    return EXERCISES_V0.get(exercise_id)


def list_exercise_ids() -> list[str]:
    """Sorted list of frozen exercise IDs (excludes reserved tokens)."""
    return sorted(EXERCISES_V0.keys())


def validate_exercise_id(exercise_id: str | None) -> str | None:
    """Return an error message if ``exercise_id`` is not recognised; else None.

    Empty / missing values are valid (manifest allows unset for clips pending
    review); reserved tokens (``mixed``, ``unknown``) are valid.
    """
    if exercise_id is None:
        return None
    s = exercise_id.strip()
    if s == "":
        return None
    if s in RESERVED_EXERCISE_TOKENS:
        return None
    if s in EXERCISES_V0:
        return None
    return (
        f"unknown exercise_id {exercise_id!r}; expected one of "
        f"{list_exercise_ids() + sorted(RESERVED_EXERCISE_TOKENS)} or empty"
    )


def _exercise_to_dict(ex: ExerciseV0) -> dict[str, Any]:
    d = asdict(ex)
    d["primary_joints"] = list(ex.primary_joints)
    return d


def to_manifest_dict() -> dict[str, Any]:
    """Serialisable frozen manifest (ordering is registry insertion order)."""
    return {
        "schema_version": EXERCISE_V0_SCHEMA_VERSION,
        "manifest_version": EXERCISE_V0_MANIFEST_VERSION,
        "allowed_categories": sorted(ALLOWED_CATEGORIES),
        "allowed_camera_views": sorted(ALLOWED_CAMERA_VIEWS),
        "allowed_framings": sorted(ALLOWED_FRAMINGS),
        "allowed_rep_signal_types": sorted(ALLOWED_REP_SIGNAL_TYPES),
        "reserved_exercise_tokens": sorted(RESERVED_EXERCISE_TOKENS),
        "exercises": [_exercise_to_dict(ex) for ex in _EXERCISE_LIST],
    }


def compute_manifest_sha256() -> str:
    """SHA-256 over the canonical-JSON serialisation of the manifest.

    Sort keys so the hash is stable against dict-insertion-order changes in
    future Python versions; the ``exercises`` list order is preserved (it is
    the source of truth for ordering).
    """
    payload = json.dumps(to_manifest_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
