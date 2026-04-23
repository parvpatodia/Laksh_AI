"""Per-rep feature vector with per-field valid / degraded / unknown semantics.

GOALS.md Milestone 1 bullet 3: "Per-rep feature vector defined (joint angles,
velocities, depth proxies, symmetry) -- each field has valid / degraded /
unknown semantics like today's ``metric_status`` pattern."

Scope (v0, deliberate cut)
--------------------------
Features are **observational**: durations, amplitudes, visibility fractions.
No ideal-range comparisons, no coaching verdicts, no "good rep / bad rep"
scoring. Ideal ranges come from the upcoming versioned calibration config
(Milestone 1 bullet 4) tied to eval evidence -- NOT from hardcoded Python
literals (GOALS.md calibration policy).

Inputs
------
* A :class:`~app.gym.rep_segmenter.RepSpan` produced by the segmenter.
* The sequence of canonical-joint frames used during segmentation. Each
  entry is either ``None`` (frame has no pose) or a mapping
  ``{joint_name: JointObservation-or-dict}``. ``joint_name`` may be a
  :class:`~app.pose.canonical.CanonicalJointName` or its string value;
  observations may be a :class:`~app.pose.canonical.JointObservation` or a
  plain ``{x, y, z, visibility}`` dict. This tolerance lets us consume both
  in-memory pose outputs and deserialised JSON dumps without a conversion
  step at every call site.
* The :class:`~app.gym.exercises_v0.ExerciseV0` that drove the segmenter.
* The sampling ``fps``.

Output contract
---------------
One :class:`RepFeatureVector` per rep. Each of the seven features is a
:class:`FieldValue` carrying value, unit, status, and reason_codes.

Stability: schema pinned by :data:`REP_FEATURES_SCHEMA_VERSION`. Any change
to field names, unit labels, or status taxonomy requires a bump.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from app.gym.exercises_v0 import ExerciseV0
from app.gym.rep_segmenter import RepSpan, SegmentResult

REP_FEATURES_SCHEMA_VERSION = "1.0.0"

# Angle triplets for cyclic_angle exercises keyed by rep_signal_joint.
# Matches v0 registry: only right_elbow and right_hip appear there.
_ANGLE_TRIPLETS: dict[str, tuple[str, str, str]] = {
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
}


@dataclass(frozen=True)
class RepFeaturesConfig:
    """Thresholds shipped alongside every :class:`RepFeatureVector`.

    Constants that decide status are NOT hidden -- they ride with the result
    so a coaching response or downstream eval can attribute every flag.
    """

    # Minimum eccentric/concentric phase duration to be called valid.
    min_phase_s: float = 0.1
    # Below this mean visibility the primary-joint field is 'degraded'.
    visibility_degraded_threshold: float = 0.5
    # Above this missing fraction in a rep window the rep is 'degraded'.
    missing_frac_degraded_threshold: float = 0.25
    # Below these signal amplitudes we call the amplitude 'degraded'
    # (not 'invalid'; some reps really are shallow).
    min_amplitude_deg: float = 5.0
    min_amplitude_normalized: float = 0.02


@dataclass(frozen=True)
class BicepCurlRomGateConfig:
    """Physiologically-grounded ROM gate for dumbbell bicep curl.

    Thresholds come from Norkin & White, *Measurement of Joint Motion:
    A Guide to Goniometry*, 5th ed., Ch. 4 (elbow flexion-extension), and
    published EMG-bicep-curl studies. They are **not** hidden: this config
    rides with the per-rep feature so any coaching or eval downstream can
    attribute every pass/fail decision.

    C1 (three-gate angle cycle) -- interior elbow angle over the rep:
        * ``start_extension_deg_min``: 150 deg at rep start
          (functional extension; anatomical full extension ~180 is not
          required on a weighted curl -- Norkin & White call 150-180 the
          functional-extension band).
        * ``peak_flexion_deg_max``: 60 deg at rep peak
          (published EMG studies use 60 not 45 because dumbbell load
          prevents most trainees from clinical full flexion of 45).
        * ``end_extension_deg_min``: 150 deg at rep end
          (return-to-extension gate; symmetric with start).

    C1 partial regime (flagged, surfaced gray, NOT counted):
        * ``peak_partial_deg_max``: 90 deg -- a half-curl whose peak
          reached 90 but failed the strict three-gate cycle.

    C2 (secondary wrist-y descent) -- scale-invariant:
        * ``wrist_y_descent_ratio``: 0.4 -- the wrist's rest-to-trough
          vertical travel must be at least 40% of the rep's mean
          shoulder-to-elbow pixel distance. Normalising by a body-scale
          measurement removes height / framing dependence.

    Consensus rule:
        * ``valid`` = C1 full AND C2 pass.
        * ``partial`` = exactly one of {C1 full, C2} passes, OR C1 reaches
          only the partial regime (peak <= 90 but not <= 60).
        * ``dropped`` = neither passes.

    Partial reps are surfaced but NOT counted in the headline rep count.
    Dropped reps are NOT surfaced (no phantom reps).
    """

    start_extension_deg_min: float = 150.0
    end_extension_deg_min: float = 150.0
    peak_flexion_deg_max: float = 60.0
    peak_partial_deg_max: float = 90.0
    wrist_y_descent_ratio: float = 0.4


@dataclass(frozen=True)
class FieldValue:
    """One measured field in a :class:`RepFeatureVector`."""

    value: float | None
    unit: str
    status: str  # "valid" | "degraded" | "unknown"
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepFeatureVector:
    """Immutable per-rep feature output."""

    schema_version: str
    exercise_id: str
    rep_index: int
    start_frame: int
    end_frame: int
    peak_frame: int
    rep_status: str  # from the RepSpan that drove this computation
    features: dict[str, FieldValue] = field(default_factory=dict)
    config: RepFeaturesConfig = field(default_factory=RepFeaturesConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict already converts FieldValue -> dict and tuples -> lists.
        return d


# ----- joint/frame normalisation helpers -----------------------------------


def _joint_name_str(key: Any) -> str:
    """Normalise dict / enum / string key -> canonical joint name string."""
    if isinstance(key, str):
        return key
    # StrEnum stringifies to the .value via __str__, but be explicit:
    value = getattr(key, "value", None)
    return value if isinstance(value, str) else str(key)


def _get_joint(
    frame: Mapping[Any, Any] | None, joint_name: str
) -> tuple[float, float, float] | None:
    """Return ``(x, y, visibility)`` for ``joint_name`` in ``frame``, or None.

    Accepts frames whose keys are string names OR CanonicalJointName enums,
    and whose values are JointObservation dataclasses OR plain dicts.
    """
    if frame is None:
        return None
    # Direct hit on string key first (cheap path).
    obs = frame.get(joint_name)  # type: ignore[arg-type]
    if obs is None:
        # Fall back: linear scan over keys normalising to str.
        for k, v in frame.items():
            if _joint_name_str(k) == joint_name:
                obs = v
                break
    if obs is None:
        return None
    x = getattr(obs, "x", None)
    y = getattr(obs, "y", None)
    vis = getattr(obs, "visibility", None)
    if x is None and isinstance(obs, Mapping):
        x = obs.get("x")
        y = obs.get("y")
        vis = obs.get("visibility")
    if x is None or y is None:
        return None
    try:
        return float(x), float(y), float(vis if vis is not None else 1.0)
    except (TypeError, ValueError):
        return None


def _interior_angle_deg(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float | None:
    """Interior angle at b (in degrees) formed by rays b->a and b->c.

    Returns None when either vector is near-zero.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    n_ba = math.hypot(ba[0], ba[1])
    n_bc = math.hypot(bc[0], bc[1])
    if n_ba < 1e-9 or n_bc < 1e-9:
        return None
    cos = (ba[0] * bc[0] + ba[1] * bc[1]) / (n_ba * n_bc)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


# ----- signal extraction ---------------------------------------------------


def extract_rep_signal(
    canonical_frames: Sequence[Mapping[Any, Any] | None],
    exercise: ExerciseV0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 1D rep-signal the segmenter would consume, plus a missingness mask.

    Dispatch:
        * ``cyclic_vertical`` -> y-coord of ``rep_signal_joint``
        * ``cyclic_angle``    -> interior angle (deg) at ``rep_signal_joint``
                                 using the standard adjacent-joint triplet
        * ``duration``        -> mean y of (shoulder, hip, ankle) midline
                                 per frame, used as a stability proxy
        * ``gait_cadence``    -> signed vertical offset between ankles
                                 (left.y - right.y), oscillates with stride
    """
    n = len(canonical_frames)
    signal = np.full(n, np.nan, dtype=np.float64)
    miss = np.ones(n, dtype=bool)
    rst = exercise.rep_signal_type

    if rst == "cyclic_vertical" and exercise.rep_signal_joint is not None:
        for i, fr in enumerate(canonical_frames):
            j = _get_joint(fr, exercise.rep_signal_joint)
            if j is not None:
                signal[i] = j[1]
                miss[i] = False
    elif rst == "cyclic_angle" and exercise.rep_signal_joint is not None:
        triplet = _ANGLE_TRIPLETS.get(exercise.rep_signal_joint)
        if triplet is None:
            # Unknown joint -> whole signal is missing; caller will see
            # flat_signal / no_reps_detected from the segmenter.
            return signal, miss
        a_name, b_name, c_name = triplet
        for i, fr in enumerate(canonical_frames):
            a = _get_joint(fr, a_name)
            b = _get_joint(fr, b_name)
            c = _get_joint(fr, c_name)
            if a is None or b is None or c is None:
                continue
            ang = _interior_angle_deg((a[0], a[1]), (b[0], b[1]), (c[0], c[1]))
            if ang is None:
                continue
            signal[i] = ang
            miss[i] = False
    elif rst == "duration":
        for i, fr in enumerate(canonical_frames):
            sh = _get_joint(fr, "right_shoulder") or _get_joint(fr, "left_shoulder")
            hp = _get_joint(fr, "right_hip") or _get_joint(fr, "left_hip")
            an = _get_joint(fr, "right_ankle") or _get_joint(fr, "left_ankle")
            if sh is None or hp is None or an is None:
                continue
            signal[i] = (sh[1] + hp[1] + an[1]) / 3.0
            miss[i] = False
    elif rst == "gait_cadence":
        for i, fr in enumerate(canonical_frames):
            la = _get_joint(fr, "left_ankle")
            ra = _get_joint(fr, "right_ankle")
            if la is None or ra is None:
                continue
            signal[i] = la[1] - ra[1]
            miss[i] = False

    return signal, miss


# ----- feature computation -------------------------------------------------


def _duration_features(
    rep: RepSpan, fps: float, cfg: RepFeaturesConfig
) -> dict[str, FieldValue]:
    length_frames = rep.end_frame - rep.start_frame + 1
    total_s = max(0.0, length_frames / fps)
    # eccentric = start -> peak, concentric = peak -> end. If the peak sits
    # at a boundary we degrade the corresponding phase.
    ecc_frames = rep.peak_frame - rep.start_frame
    con_frames = rep.end_frame - rep.peak_frame
    ecc_s = ecc_frames / fps
    con_s = con_frames / fps

    def _phase(value: float, code_prefix: str) -> FieldValue:
        if value <= 0:
            return FieldValue(None, "s", "unknown", (f"{code_prefix}_missing",))
        if value < cfg.min_phase_s:
            return FieldValue(value, "s", "degraded", (f"{code_prefix}_too_short",))
        return FieldValue(value, "s", "valid", ())

    duration = FieldValue(total_s, "s", "valid" if total_s > 0 else "unknown", ())
    ecc = _phase(ecc_s, "eccentric")
    con = _phase(con_s, "concentric")
    if ecc.value is not None and con.value is not None and con.value > 0:
        tempo = FieldValue(
            ecc.value / con.value,
            "ratio",
            "valid" if ecc.status == "valid" and con.status == "valid" else "degraded",
            (),
        )
    else:
        tempo = FieldValue(None, "ratio", "unknown", ("phase_missing",))
    return {
        "rep_duration_s": duration,
        "eccentric_duration_s": ecc,
        "concentric_duration_s": con,
        "tempo_ratio_ecc_over_con": tempo,
    }


def _amplitude_feature(
    signal: np.ndarray,
    miss: np.ndarray,
    rep: RepSpan,
    exercise: ExerciseV0,
    cfg: RepFeaturesConfig,
) -> FieldValue:
    window = signal[rep.start_frame : rep.end_frame + 1]
    window_miss = miss[rep.start_frame : rep.end_frame + 1]
    usable = window[~window_miss]
    usable = usable[np.isfinite(usable)]
    rst = exercise.rep_signal_type
    if rst == "cyclic_angle":
        unit = "deg"
        min_amp = cfg.min_amplitude_deg
    elif rst == "cyclic_vertical":
        unit = "normalized_y"
        min_amp = cfg.min_amplitude_normalized
    elif rst == "gait_cadence":
        unit = "normalized_y"
        min_amp = cfg.min_amplitude_normalized
    else:  # duration hold -> amplitude is a stability proxy
        unit = "normalized_y"
        min_amp = cfg.min_amplitude_normalized
    if usable.size < 2:
        return FieldValue(None, unit, "unknown", ("no_usable_samples",))
    amp = float(np.max(usable) - np.min(usable))
    if rst == "duration":
        # For a hold, high amplitude = instability, not range. Flip semantics:
        # very small amplitude is 'valid stable'; large amplitude is degraded.
        if amp <= min_amp:
            return FieldValue(amp, unit, "valid", ("stable_hold",))
        return FieldValue(amp, unit, "degraded", ("unstable_hold",))
    if amp < min_amp:
        return FieldValue(amp, unit, "degraded", ("low_amplitude",))
    return FieldValue(amp, unit, "valid", ())


def _visibility_and_missing_features(
    canonical_frames: Sequence[Mapping[Any, Any] | None],
    rep: RepSpan,
    exercise: ExerciseV0,
    cfg: RepFeaturesConfig,
) -> tuple[FieldValue, FieldValue]:
    start, end = rep.start_frame, rep.end_frame
    window_frames = canonical_frames[start : end + 1]
    n_window = len(window_frames)
    if n_window == 0:
        return (
            FieldValue(None, "visibility", "unknown", ("no_frames",)),
            FieldValue(None, "frac", "unknown", ("no_frames",)),
        )
    # Missingness = fraction of frames where ANY primary joint is missing.
    n_missing = 0
    # Per-joint mean visibility (over frames where the joint is present).
    per_joint_means: list[float] = []
    for joint_name in exercise.primary_joints:
        present_vis: list[float] = []
        for fr in window_frames:
            j = _get_joint(fr, joint_name)
            if j is None:
                continue
            present_vis.append(j[2])
        if present_vis:
            per_joint_means.append(float(np.mean(present_vis)))
    for fr in window_frames:
        if fr is None:
            n_missing += 1
            continue
        if any(_get_joint(fr, j) is None for j in exercise.primary_joints):
            n_missing += 1
    missing_frac = n_missing / n_window
    if per_joint_means:
        min_vis = float(min(per_joint_means))
        if min_vis < cfg.visibility_degraded_threshold:
            vis_field = FieldValue(min_vis, "visibility", "degraded", ("low_visibility",))
        else:
            vis_field = FieldValue(min_vis, "visibility", "valid", ())
    else:
        vis_field = FieldValue(None, "visibility", "unknown", ("no_joint_observations",))
    if missing_frac > cfg.missing_frac_degraded_threshold:
        miss_field = FieldValue(
            missing_frac, "frac", "degraded", ("high_missingness",)
        )
    else:
        miss_field = FieldValue(missing_frac, "frac", "valid", ())
    return vis_field, miss_field


def evaluate_bicep_curl_rom_gate(
    rep: RepSpan,
    canonical_frames: Sequence[Mapping[Any, Any] | None],
    exercise: ExerciseV0,
    gate_cfg: BicepCurlRomGateConfig | None = None,
) -> FieldValue:
    """Evaluate Norkin & White-cited ROM gate for one bicep-curl rep.

    Pure function. Returns a :class:`FieldValue` whose ``status`` is one of:

    * ``"valid"``   -- C1 full AND C2 pass (full curl by both signals).
    * ``"partial"`` -- exactly one of {C1 full, C2} passes, OR C1 reaches
      the partial regime (peak 60 < angle <= 90).
    * ``"dropped"`` -- neither C1 nor C2 passes.
    * ``"unknown"`` -- landmarks too sparse to evaluate (no frames, no
      angle samples at start/peak/end, or wrist/shoulder-elbow missing
      from the whole window).

    ``reason_codes`` name every gate that failed so the caller can surface
    them in the UI (e.g. ``("start_not_extended", "wrist_amplitude_low")``).

    The returned ``value`` carries no physical meaning on its own --
    consumers read ``status`` + ``reason_codes``. We store ``None`` for
    ``value`` to avoid tempting a downstream user to plot it as a scalar.
    """
    if exercise.exercise_id != "dumbbell_bicep_curl":
        # Defensive: caller should not invoke this for other exercises.
        return FieldValue(None, "curl_rom_gate", "unknown", ("wrong_exercise",))
    cfg = gate_cfg or BicepCurlRomGateConfig()

    a_name, b_name, c_name = _ANGLE_TRIPLETS["right_elbow"]
    # Fallback to left side if right is occluded -- match the browser
    # mirror (repCounter.ts) which picks the best-visibility side.
    def _angle_at(frame_idx: int) -> float | None:
        if frame_idx < 0 or frame_idx >= len(canonical_frames):
            return None
        fr = canonical_frames[frame_idx]
        for triplet in (
            _ANGLE_TRIPLETS["right_elbow"],
            _ANGLE_TRIPLETS["left_elbow"],
        ):
            a = _get_joint(fr, triplet[0])
            b = _get_joint(fr, triplet[1])
            c = _get_joint(fr, triplet[2])
            if a is None or b is None or c is None:
                continue
            ang = _interior_angle_deg(
                (a[0], a[1]), (b[0], b[1]), (c[0], c[1])
            )
            if ang is not None:
                return ang
        return None

    start_angle = _angle_at(rep.start_frame)
    peak_angle = _angle_at(rep.peak_frame)
    end_angle = _angle_at(rep.end_frame)

    reason_codes: list[str] = []

    # C1 -- three-gate angle cycle.
    c1_start_ok = start_angle is not None and start_angle >= cfg.start_extension_deg_min
    c1_peak_full = peak_angle is not None and peak_angle <= cfg.peak_flexion_deg_max
    c1_peak_partial = peak_angle is not None and peak_angle <= cfg.peak_partial_deg_max
    c1_end_ok = end_angle is not None and end_angle >= cfg.end_extension_deg_min
    c1_full_pass = c1_start_ok and c1_peak_full and c1_end_ok

    if not c1_start_ok:
        reason_codes.append("start_not_extended")
    if not c1_peak_full:
        reason_codes.append("peak_not_flexed")
    if not c1_end_ok:
        reason_codes.append("end_not_extended")

    # C2 -- wrist-y descent normalised by shoulder-elbow pixel length.
    # Pixel-y grows downward; the wrist goes UP during a curl, so pixel-y
    # DECREASES. The relevant amplitude is rest_y - min_y over the rep
    # window, normalised by the rep's mean shoulder-to-elbow length.
    wrist_ys: list[float] = []
    se_lengths: list[float] = []
    start = max(0, rep.start_frame)
    end = min(len(canonical_frames) - 1, rep.end_frame)
    for i in range(start, end + 1):
        fr = canonical_frames[i]
        # Try right side first, fall back to left.
        for side in ("right", "left"):
            sh = _get_joint(fr, f"{side}_shoulder")
            el = _get_joint(fr, f"{side}_elbow")
            wr = _get_joint(fr, f"{side}_wrist")
            if sh is not None and el is not None and wr is not None:
                wrist_ys.append(wr[1])
                se_lengths.append(
                    math.hypot(sh[0] - el[0], sh[1] - el[1])
                )
                break

    c2_pass = False
    wrist_amplitude: float | None = None
    if wrist_ys and se_lengths:
        mean_se = float(np.mean(se_lengths))
        if mean_se >= 1e-6:
            # "Descent" = rest-wrist-y minus min-wrist-y (positive when the
            # wrist rises in world coords). Use the rep's FIRST sample as the
            # rest reference since the rep starts at an extension boundary.
            rest_y = wrist_ys[0]
            min_y = float(np.min(wrist_ys))
            descent = rest_y - min_y
            wrist_amplitude = descent / mean_se
            c2_pass = wrist_amplitude >= cfg.wrist_y_descent_ratio

    if not c2_pass:
        reason_codes.append("wrist_amplitude_low")

    # Unknown state: we could not evaluate either gate at all.
    angles_unknown = (
        start_angle is None and peak_angle is None and end_angle is None
    )
    if angles_unknown and wrist_amplitude is None:
        return FieldValue(
            None, "curl_rom_gate", "unknown", ("no_joint_observations",)
        )

    # Consensus status.
    if c1_full_pass and c2_pass:
        status = "valid"
        reason_codes = []  # clean: no failures to surface
    elif (c1_full_pass and not c2_pass) or (c2_pass and not c1_full_pass):
        status = "partial"
        reason_codes.append("single_signal_rom")
    elif c1_peak_partial:
        # Partial regime: peak reached <=90 deg but failed the strict
        # three-gate cycle and C2 also failed.
        status = "partial"
        reason_codes.append("partial_rom")
    else:
        status = "dropped"
        reason_codes.append("twitch")

    return FieldValue(None, "curl_rom_gate", status, tuple(reason_codes))


def compute_rep_features(
    rep: RepSpan,
    rep_index: int,
    canonical_frames: Sequence[Mapping[Any, Any] | None],
    exercise: ExerciseV0,
    fps: float,
    config: RepFeaturesConfig | None = None,
    curl_rom_config: BicepCurlRomGateConfig | None = None,
) -> RepFeatureVector:
    """Compute per-rep feature vector for ``rep``. Pure function.

    For ``dumbbell_bicep_curl`` the result additionally carries a
    ``curl_rom_gate`` field whose status is the Norkin & White-cited
    consensus of C1 (three-gate angle cycle) and C2 (normalised wrist-y
    descent). See :func:`evaluate_bicep_curl_rom_gate`.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if rep.start_frame < 0 or rep.end_frame < rep.start_frame:
        raise ValueError(f"invalid RepSpan frames: {rep.start_frame}..{rep.end_frame}")
    cfg = config or RepFeaturesConfig()
    features = _duration_features(rep, fps, cfg)
    signal, miss = extract_rep_signal(canonical_frames, exercise)
    features["signal_amplitude"] = _amplitude_feature(signal, miss, rep, exercise, cfg)
    vis, missing = _visibility_and_missing_features(canonical_frames, rep, exercise, cfg)
    features["primary_joints_min_visibility"] = vis
    features["primary_joints_missing_frac"] = missing
    if exercise.exercise_id == "dumbbell_bicep_curl":
        features["curl_rom_gate"] = evaluate_bicep_curl_rom_gate(
            rep, canonical_frames, exercise, curl_rom_config
        )
    return RepFeatureVector(
        schema_version=REP_FEATURES_SCHEMA_VERSION,
        exercise_id=exercise.exercise_id,
        rep_index=rep_index,
        start_frame=rep.start_frame,
        end_frame=rep.end_frame,
        peak_frame=rep.peak_frame,
        rep_status=rep.status,
        features=features,
        config=cfg,
    )


def feature_vectors_from_segment(
    segment: SegmentResult,
    canonical_frames: Sequence[Mapping[Any, Any] | None],
    exercise: ExerciseV0,
    config: RepFeaturesConfig | None = None,
) -> list[RepFeatureVector]:
    """Convenience: compute features for every rep in a :class:`SegmentResult`."""
    return [
        compute_rep_features(rep, i, canonical_frames, exercise, segment.fps, config)
        for i, rep in enumerate(segment.reps)
    ]
