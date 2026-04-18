/**
 * Hardcoded failure-mode demo fixtures for the "Failure modes & honesty" section.
 *
 * Each entry is shaped like an AnalyzeResponse so FailureModeCards can render
 * the same status chips and field rows used for real results. The three modes
 * cover the most common real-world failure scenarios.
 */

import type { AnalyzeResponse, CalibrationBlock, RepVector } from "@/lib/api";

/** A single failure-mode entry with display metadata and a partial result. */
export interface FailureMode {
  /** Stable identifier used as React key and CSS anchor. */
  id: string;
  /** Short human label shown in the card header. */
  label: string;
  /** One-sentence explanation of what triggered this mode. */
  description: string;
  /** Emoji icon for the card header. */
  icon: string;
  /** Realistic partial AnalyzeResponse demonstrating the failure. */
  result: Partial<AnalyzeResponse> & {
    feature_vectors: RepVector[];
    calibration: CalibrationBlock;
  };
}

/** Shared calibration block used by all fixtures (no reference data yet). */
const UNCALIBRATED_CALIB: CalibrationBlock = {
  exercise_id: "back_squat",
  evidence_status: "uncalibrated_v0",
  evidence_source: null,
  comparable_fields: [],
  per_rep: [],
};

/** Mode 1: joints become occluded mid-rep (e.g. towel, camera angle). */
const JOINT_OCCLUSION: FailureMode = {
  id: "joint_occlusion",
  label: "Occluded joints",
  description:
    "Key joints disappeared mid-rep because the athlete moved out of frame or was partially hidden.",
  icon: "~",
  result: {
    schema_version: "1.0.0",
    sport_id: "gym",
    exercise_id: "back_squat",
    source: "video",
    fps: 30,
    n_frames: 90,
    analysis_mode: "canonical_backend",
    segment: { status: "degraded", reason_codes: ["visibility_below_threshold"] },
    feature_vectors: [
      {
        rep_index: 0,
        start_frame: 0,
        end_frame: 89,
        peak_frame: 44,
        rep_status: "degraded",
        features: {
          rep_duration_s: {
            value: 2.9,
            unit: "s",
            status: "degraded",
            reason_codes: ["visibility_below_threshold"],
          },
          eccentric_phase_s: {
            value: null,
            unit: "s",
            status: "degraded",
            reason_codes: ["visibility_below_threshold"],
          },
          concentric_phase_s: {
            value: null,
            unit: "s",
            status: "degraded",
            reason_codes: ["visibility_below_threshold"],
          },
          tempo_ratio: {
            value: null,
            unit: "",
            status: "degraded",
            reason_codes: ["visibility_below_threshold"],
          },
          signal_amplitude: {
            value: 0.21,
            unit: "",
            status: "degraded",
            reason_codes: ["visibility_below_threshold"],
          },
          min_visibility: {
            value: 0.18,
            unit: "",
            status: "degraded",
            reason_codes: ["visibility_below_threshold"],
          },
        },
      },
    ],
    calibration: UNCALIBRATED_CALIB,
    parity_probe: null,
  },
};

/** Mode 2: camera too far away -- no reps detected at all. */
const NO_REPS_DETECTED: FailureMode = {
  id: "no_reps_detected",
  label: "No reps detected",
  description:
    "Camera was too far away or the signal had insufficient variance to identify any rep boundaries.",
  icon: "?",
  result: {
    schema_version: "1.0.0",
    sport_id: "gym",
    exercise_id: "back_squat",
    source: "video",
    fps: 30,
    n_frames: 120,
    analysis_mode: "canonical_backend",
    segment: {
      status: "unknown",
      reason_codes: ["insufficient_signal_variance"],
    },
    feature_vectors: [],
    calibration: UNCALIBRATED_CALIB,
    parity_probe: null,
  },
};

/** Mode 3: two athletes in frame -- system cannot assign landmarks to one person. */
const MULTI_PERSON: FailureMode = {
  id: "multi_person",
  label: "Multiple people in frame",
  description:
    "Two people were detected simultaneously; landmark ownership is ambiguous so both reps are flagged degraded.",
  icon: "!",
  result: {
    schema_version: "1.0.0",
    sport_id: "gym",
    exercise_id: "back_squat",
    source: "video",
    fps: 30,
    n_frames: 180,
    analysis_mode: "canonical_backend",
    segment: { status: "degraded", reason_codes: ["multi_person_ambiguity"] },
    feature_vectors: [
      {
        rep_index: 0,
        start_frame: 0,
        end_frame: 89,
        peak_frame: 44,
        rep_status: "degraded",
        features: {
          rep_duration_s: {
            value: 3.1,
            unit: "s",
            status: "degraded",
            reason_codes: ["multi_person_ambiguity"],
          },
          eccentric_phase_s: {
            value: null,
            unit: "s",
            status: "degraded",
            reason_codes: ["multi_person_ambiguity"],
          },
          concentric_phase_s: {
            value: null,
            unit: "s",
            status: "degraded",
            reason_codes: ["multi_person_ambiguity"],
          },
          min_visibility: {
            value: 0.44,
            unit: "",
            status: "degraded",
            reason_codes: ["multi_person_ambiguity"],
          },
        },
      },
      {
        rep_index: 1,
        start_frame: 90,
        end_frame: 179,
        peak_frame: 134,
        rep_status: "degraded",
        features: {
          rep_duration_s: {
            value: 3.0,
            unit: "s",
            status: "degraded",
            reason_codes: ["multi_person_ambiguity"],
          },
          eccentric_phase_s: {
            value: null,
            unit: "s",
            status: "degraded",
            reason_codes: ["multi_person_ambiguity"],
          },
          concentric_phase_s: {
            value: null,
            unit: "s",
            status: "degraded",
            reason_codes: ["multi_person_ambiguity"],
          },
          min_visibility: {
            value: 0.41,
            unit: "",
            status: "degraded",
            reason_codes: ["multi_person_ambiguity"],
          },
        },
      },
    ],
    calibration: UNCALIBRATED_CALIB,
    parity_probe: null,
  },
};

/** All three failure modes in display order. */
export const FAILURE_MODES: FailureMode[] = [
  JOINT_OCCLUSION,
  NO_REPS_DETECTED,
  MULTI_PERSON,
];
