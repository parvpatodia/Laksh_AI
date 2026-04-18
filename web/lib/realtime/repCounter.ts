/**
 * Browser-side lightweight rep counter for ghost metrics (realtime_preview).
 *
 * This is NOT a port of the Python gym pipeline.  It is intentionally simple:
 * derive one scalar signal per exercise from the MediaPipe landmark stream,
 * detect peaks/troughs by sign-change of the smoothed first derivative, and
 * report per-rep duration + phase.  The ADR 0004 parity probe will quantify
 * how close these estimates are to the canonical backend results.
 *
 * Landmark indices follow MediaPipe Pose (COCO-17 superset, 33 points):
 *   https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
 *
 * All signals are normalised to [0, 1] (fraction of frame height/width) so
 * the rep counter never needs calibration: it counts cycles regardless of
 * the absolute joint position.
 *
 * reason_codes on every field include "realtime_preview" so the UI can
 * label ghost metrics distinctly from canonical results.
 */

import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

// ---------------------------------------------------------------------------
// Landmark index constants (MediaPipe Pose 33-point model)
// ---------------------------------------------------------------------------

const LM = {
  LEFT_HIP:    23,
  RIGHT_HIP:   24,
  LEFT_KNEE:   25,
  RIGHT_KNEE:  26,
  LEFT_ANKLE:  27,
  RIGHT_ANKLE: 28,
  LEFT_SHOULDER:  11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW:  13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST:  15,
  RIGHT_WRIST: 16,
};

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/** 2D angle at *b* formed by points *a*-*b*-*c* in [0, 180] degrees. */
function angle2d(
  a: NormalizedLandmark,
  b: NormalizedLandmark,
  c: NormalizedLandmark,
): number {
  const ax = a.x - b.x, ay = a.y - b.y;
  const cx = c.x - b.x, cy = c.y - b.y;
  const dot = ax * cx + ay * cy;
  const mag = Math.sqrt((ax * ax + ay * ay) * (cx * cx + cy * cy));
  if (mag < 1e-6) return 0;
  return (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
}

/** Average y of two landmarks (both must exist). */
function avgY(a: NormalizedLandmark, b: NormalizedLandmark): number {
  return (a.y + b.y) / 2;
}

/** Minimum visibility of a set of landmarks. */
function minVis(...lms: NormalizedLandmark[]): number {
  return Math.min(...lms.map((l) => l.visibility ?? 0));
}

// ---------------------------------------------------------------------------
// Signal extraction per exercise / sport
// ---------------------------------------------------------------------------

export type SportId = "basketball" | "gym";

/**
 * Extract the primary rep signal scalar for the given exercise.
 *
 * Returns ``null`` when landmarks are too occluded to produce a reliable
 * signal (min visibility < 0.4).
 */
export function extractSignal(
  landmarks: NormalizedLandmark[],
  exerciseId: string,
): number | null {
  const get = (idx: number) => landmarks[idx];

  switch (exerciseId) {
    // ---- Squat variants: hip y tracks depth --------------------------------
    case "back_squat":
    case "front_squat":
    case "lunge": {
      const lh = get(LM.LEFT_HIP), rh = get(LM.RIGHT_HIP);
      if (!lh || !rh || minVis(lh, rh) < 0.4) return null;
      // Higher y = lower hip (image coords) -- invert so "deep squat" = low signal.
      return 1 - avgY(lh, rh);
    }

    // ---- Hip hinge: hip y + knee extension ---------------------------------
    case "deadlift":
    case "romanian_deadlift":
    case "hip_thrust": {
      const lh = get(LM.LEFT_HIP), rh = get(LM.RIGHT_HIP);
      if (!lh || !rh || minVis(lh, rh) < 0.4) return null;
      return 1 - avgY(lh, rh);
    }

    // ---- Upper-body push: elbow angle --------------------------------------
    case "bench_press":
    case "overhead_press":
    case "tricep_pushdown": {
      const ls = get(LM.LEFT_SHOULDER), le = get(LM.LEFT_ELBOW), lw = get(LM.LEFT_WRIST);
      const rs = get(LM.RIGHT_SHOULDER), re = get(LM.RIGHT_ELBOW), rw = get(LM.RIGHT_WRIST);
      const leftOk = ls && le && lw && minVis(ls, le, lw) >= 0.4;
      const rightOk = rs && re && rw && minVis(rs, re, rw) >= 0.4;
      if (!leftOk && !rightOk) return null;
      const angles: number[] = [];
      if (leftOk)  angles.push(angle2d(ls!, le!, lw!));
      if (rightOk) angles.push(angle2d(rs!, re!, rw!));
      // Normalise 0-180 deg -> 0-1.
      return angles.reduce((a, b) => a + b, 0) / angles.length / 180;
    }

    // ---- Pull + curl: elbow angle ------------------------------------------
    case "barbell_row":
    case "pull_up":
    case "dumbbell_curl": {
      const ls = get(LM.LEFT_SHOULDER), le = get(LM.LEFT_ELBOW), lw = get(LM.LEFT_WRIST);
      const rs = get(LM.RIGHT_SHOULDER), re = get(LM.RIGHT_ELBOW), rw = get(LM.RIGHT_WRIST);
      const leftOk = ls && le && lw && minVis(ls, le, lw) >= 0.4;
      const rightOk = rs && re && rw && minVis(rs, re, rw) >= 0.4;
      if (!leftOk && !rightOk) return null;
      const angles: number[] = [];
      if (leftOk)  angles.push(angle2d(ls!, le!, lw!));
      if (rightOk) angles.push(angle2d(rs!, re!, rw!));
      return angles.reduce((a, b) => a + b, 0) / angles.length / 180;
    }

    // ---- Basketball: wrist y (release height) ------------------------------
    case "basketball":
    case "jump_shot": {
      const lw = get(LM.LEFT_WRIST), rw = get(LM.RIGHT_WRIST);
      if (!lw || !rw || minVis(lw, rw) < 0.4) return null;
      // Low y = high wrist in image coords: invert so "arms raised" = high signal.
      return 1 - avgY(lw, rw);
    }

    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// EMA smoother
// ---------------------------------------------------------------------------

/** Exponential moving average state. */
export interface EmaState {
  value: number;
  alpha: number;
}

export function makeEma(alpha = 0.2): EmaState {
  return { value: 0, alpha };
}

export function updateEma(state: EmaState, sample: number): number {
  state.value = state.alpha * sample + (1 - state.alpha) * state.value;
  return state.value;
}

// ---------------------------------------------------------------------------
// Rep detection
// ---------------------------------------------------------------------------

export type Phase = "rest" | "eccentric" | "concentric";

/** Ghost metric FieldValue mirroring the Python FieldValueModel. */
export interface GhostField {
  value: number | null;
  unit: string;
  status: "valid" | "degraded" | "unknown";
  reason_codes: string[];
}

/** Ghost metrics emitted per rep. */
export interface GhostRepMetrics {
  rep_index: number;
  start_ts: number;
  end_ts: number;
  rep_duration_s: GhostField;
  eccentric_duration_s: GhostField;
  concentric_duration_s: GhostField;
  tempo_ratio_ecc_over_con: GhostField;
  min_visibility: GhostField;
}

/** Internal per-rep tracking. */
interface RepInProgress {
  rep_index: number;
  start_ts: number;
  peak_ts: number | null;
  min_vis: number;
  phase: Phase;
}

/** State managed externally (pass to feedFrame, get back updated state). */
export interface RepCounterState {
  ema: EmaState;
  prevSmoothed: number;
  prevDelta: number;
  currentRep: RepInProgress | null;
  completedReps: GhostRepMetrics[];
  repCount: number;
  currentPhase: Phase;
  currentSignal: number | null;
}

export function makeRepCounterState(): RepCounterState {
  return {
    ema: makeEma(0.25),
    prevSmoothed: 0,
    prevDelta: 0,
    currentRep: null,
    completedReps: [],
    repCount: 0,
    currentPhase: "rest",
    currentSignal: null,
  };
}

const DELTA_THRESHOLD = 0.005; // minimum smoothed delta to declare motion

/** Build a GhostField. */
function field(
  value: number | null,
  unit: string,
  status: GhostField["status"],
): GhostField {
  return {
    value,
    unit,
    status,
    reason_codes: ["realtime_preview"],
  };
}

/**
 * Feed one landmark frame into the rep counter.
 *
 * Mutates *state* in place and returns any newly completed rep (or null).
 */
export function feedFrame(
  state: RepCounterState,
  landmarks: NormalizedLandmark[],
  exerciseId: string,
  timestampMs: number,
): GhostRepMetrics | null {
  const raw = extractSignal(landmarks, exerciseId);
  if (raw === null) {
    state.currentSignal = null;
    return null;
  }

  const smoothed = updateEma(state.ema, raw);
  const delta = smoothed - state.prevSmoothed;
  state.currentSignal = smoothed;

  // Phase detection by sign change of delta.
  let newPhase: Phase = "rest";
  if (delta > DELTA_THRESHOLD) newPhase = "concentric";
  else if (delta < -DELTA_THRESHOLD) newPhase = "eccentric";

  // Compute min visibility across key landmarks.
  const vis = Math.min(
    ...landmarks
      .filter((_, i) => [LM.LEFT_HIP, LM.RIGHT_HIP, LM.LEFT_KNEE, LM.RIGHT_KNEE,
                          LM.LEFT_ELBOW, LM.RIGHT_ELBOW].includes(i))
      .map((l) => l.visibility ?? 0),
  );

  // Rep boundary: sign flip of delta (peak or trough).
  const signFlip =
    (state.prevDelta > DELTA_THRESHOLD && delta < -DELTA_THRESHOLD) ||
    (state.prevDelta < -DELTA_THRESHOLD && delta > DELTA_THRESHOLD);

  let completedRep: GhostRepMetrics | null = null;

  if (signFlip) {
    if (state.currentRep === null) {
      // Start new rep.
      state.currentRep = {
        rep_index: state.repCount,
        start_ts: timestampMs,
        peak_ts: timestampMs,
        min_vis: vis,
        phase: newPhase,
      };
    } else {
      // Complete current rep on second sign flip.
      const rep = state.currentRep;
      rep.peak_ts = rep.peak_ts ?? timestampMs;
      const durS = (timestampMs - rep.start_ts) / 1000;
      const eccS = (rep.peak_ts - rep.start_ts) / 1000;
      const conS = (timestampMs - rep.peak_ts) / 1000;
      const ratio = conS > 0.01 ? eccS / conS : null;
      const repVis = Math.min(rep.min_vis, vis);
      const visStatus: GhostField["status"] = repVis >= 0.6 ? "valid" : repVis >= 0.4 ? "degraded" : "unknown";

      completedRep = {
        rep_index: rep.rep_index,
        start_ts: rep.start_ts,
        end_ts: timestampMs,
        rep_duration_s:          field(parseFloat(durS.toFixed(3)), "s", durS > 0.5 && durS < 15 ? visStatus : "degraded"),
        eccentric_duration_s:    field(parseFloat(eccS.toFixed(3)), "s", eccS > 0.1 ? visStatus : "degraded"),
        concentric_duration_s:   field(parseFloat(conS.toFixed(3)), "s", conS > 0.1 ? visStatus : "degraded"),
        tempo_ratio_ecc_over_con: field(ratio !== null ? parseFloat(ratio.toFixed(2)) : null, "ratio", ratio !== null ? visStatus : "unknown"),
        min_visibility:          field(parseFloat(repVis.toFixed(3)), "fraction", visStatus),
      };

      state.repCount++;
      state.completedReps.push(completedRep);

      // Start next rep immediately.
      state.currentRep = {
        rep_index: state.repCount,
        start_ts: timestampMs,
        peak_ts: null,
        min_vis: vis,
        phase: newPhase,
      };
    }
  } else if (state.currentRep) {
    // Track min visibility within rep.
    state.currentRep.min_vis = Math.min(state.currentRep.min_vis, vis);
    // Record first peak within rep.
    if (state.currentRep.peak_ts === null && newPhase !== state.currentRep.phase) {
      state.currentRep.peak_ts = timestampMs;
      state.currentRep.phase = newPhase;
    }
  }

  state.prevSmoothed = smoothed;
  state.prevDelta = delta;
  state.currentPhase = newPhase;

  return completedRep;
}
