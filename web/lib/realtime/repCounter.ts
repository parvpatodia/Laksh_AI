/**
 * Browser-side ghost rep counter (realtime_preview).
 *
 * Design contract
 * ---------------
 * This is NOT a port of the Python `app/gym/rep_segmenter.py`.  It is a
 * deliberately simple, low-latency, stateful pass over the MediaPipe Pose
 * landmark stream that emits one `GhostRepMetrics` object per detected rep
 * AT THE INSTANT THE REP COMPLETES.  The canonical pipeline runs on the
 * recorded clip after the user stops.
 *
 * Honesty contract
 * ----------------
 * We refuse to count anything that does not pass four quality gates that
 * mirror the canonical `SegmenterConfig` defaults:
 *
 *   1. Warm-up frames before counting starts (lets the user get into
 *      position; rejects "lowering into push-up plank" as a rep).
 *   2. Minimum rep duration  (matches `SegmenterConfig.min_rep_s` = 0.4 s).
 *   3. Minimum rep amplitude (mirrors `prominence_frac * signal_range`
 *      from the canonical peak detector — rejects camera jitter as reps).
 *   4. Minimum visibility on the rep-defining joints during the rep
 *      (mirrors `max_missingness_per_span` in canonical).
 *
 * If a candidate rep fails any gate, it is DISCARDED — the user-visible
 * rep counter does not increment.  The counter is therefore always a
 * lower bound on the user's true rep count, never an upper bound.  We
 * decided that under-counting by one is a far better failure mode for
 * a research-showcase demo than the inverse (the symptom the user
 * reported: 35 reps for ~5 actual push-ups, caused by no quality gates).
 *
 * Per-exercise signal selection mirrors `app/gym/exercises_v0.py`
 * `rep_signal_joint` / `rep_signal_type`, so that the parity probe
 * (`app/parity/realtime.py`) compares like-with-like.  A mismatch here
 * silently scores parity against the wrong feature and is a demo-killer.
 */

import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

// ---------------------------------------------------------------------------
// Landmark index constants (MediaPipe Pose 33-point model)
// https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
// ---------------------------------------------------------------------------

const LM = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
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

function avgY(a: NormalizedLandmark, b: NormalizedLandmark): number {
  return (a.y + b.y) / 2;
}

function minVis(...lms: NormalizedLandmark[]): number {
  return Math.min(...lms.map((l) => l.visibility ?? 0));
}

// ---------------------------------------------------------------------------
// Per-exercise rep configuration.
//
// Single source of truth for: signal extractor, peak/trough direction,
// minimum amplitude, minimum rep duration.  The rest of the file is
// generic over this table.
//
// Backend reference: app/gym/exercises_v0.py + app/gym/rep_segmenter.py
// ---------------------------------------------------------------------------

export type SportId = "basketball" | "gym";

type SignalKind =
  | "vertical_hip"   // 1 - avg(hip_y); trough at lowest body
  | "vertical_knee"  // 1 - right_knee_y; trough at lunge bottom
  | "vertical_wrist" // 1 - avg(wrist_y); peak at release
  | "elbow_angle"    // angle(shoulder, elbow, wrist) / 180; trough at max flexion
  | "hip_angle";     // angle(shoulder, hip, knee) / 180; trough at max hinge

type RepPeakDirection = "peak" | "trough";

interface ExerciseRepConfig {
  signalKind: SignalKind;
  peakDirection: RepPeakDirection;
  /**
   * Minimum amplitude (max(signal) − min(signal)) within a single rep
   * window for the rep to be counted.  Units are normalised signal
   * units (signal is always in [0, 1]).
   *
   * For elbow_angle/hip_angle this is a fraction of 180° — 0.10 = 18°
   * which is the smallest range a human can reliably do as a "rep".
   * For vertical_* this is a fraction of frame height — 0.06 = 6% which
   * is a few inches at typical webcam framing.
   */
  minAmplitude: number;
  /**
   * Minimum rep duration in seconds.  Mirrors
   * `SegmenterConfig.min_rep_s = 0.4` from the canonical backend.
   */
  minRepS: number;
}

/**
 * Per-exercise configuration table.  Every key here MUST exist in
 * `app/gym/exercises_v0.py` (or be a basketball alias) — otherwise the
 * UI will accept a selection the backend rejects with 400
 * UnknownExerciseError.  Verified against `_EXERCISE_LIST` 2026-04-19.
 *
 * --- Threshold divergence from backend (intentional) ---
 *
 * The backend `SegmenterConfig` defaults are:
 *   min_rep_s = 0.4,  max_rep_s = 8.0,  prominence_frac = 0.15
 *
 * Frontend `minRepS` values (0.5–0.7) are INTENTIONALLY STRICTER than
 * the backend's 0.4 s floor.  Rationale: the frontend runs on noisy
 * 30 fps webcam input where sub-0.5 s "reps" are almost always tracking
 * jitter; the backend processes the same video with heavy-model
 * MediaPipe at known FPS and can safely accept shorter cycles.
 *
 * Frontend `minAmplitude` values are tuned per-exercise because each
 * signal type has a different physical range.  The backend uses a
 * single `prominence_frac = 0.15 × signal_range` which auto-adapts.
 *
 * The `validateRep` max duration (see below) is set to 8 s to match
 * the backend's `max_rep_s = 8.0`.
 *
 * Cross-reference: app/gym/rep_segmenter.py:SegmenterConfig
 *
 * --- Why some peakDirections are "trough" ---
 *
 * The frontend signal extractor INVERTS image-y (returns `1 - y`) so
 * "deeper" / "lower physically" is a SMALLER number.  The canonical
 * backend's `find_peaks(+signal)` for cyclic_vertical detects the
 * highest image-y, which corresponds to a trough in the frontend
 * signal.  Likewise for cyclic_angle: backend `find_peaks(-signal)`
 * detects the smallest joint angle, which is also a trough in the
 * frontend's `angle/180` representation.  Hence almost every gym
 * exercise is "trough".  Basketball is the exception — release at
 * max wrist height is a peak of `1 - wrist_y`.
 */
const EXERCISE_CONFIGS: Record<string, ExerciseRepConfig> = {
  // cyclic_vertical (backend rep_signal_joint = right_hip)
  back_squat: { signalKind: "vertical_hip", peakDirection: "trough", minAmplitude: 0.06, minRepS: 0.6 },
  front_squat: { signalKind: "vertical_hip", peakDirection: "trough", minAmplitude: 0.06, minRepS: 0.6 },
  conventional_deadlift: { signalKind: "vertical_hip", peakDirection: "trough", minAmplitude: 0.06, minRepS: 0.7 },
  // cyclic_vertical (backend rep_signal_joint = right_knee)
  walking_lunge: { signalKind: "vertical_knee", peakDirection: "trough", minAmplitude: 0.05, minRepS: 0.6 },
  // cyclic_angle (backend rep_signal_joint = right_elbow)
  bench_press: { signalKind: "elbow_angle", peakDirection: "trough", minAmplitude: 0.12, minRepS: 0.5 },
  overhead_press: { signalKind: "elbow_angle", peakDirection: "trough", minAmplitude: 0.15, minRepS: 0.5 },
  barbell_row: { signalKind: "elbow_angle", peakDirection: "trough", minAmplitude: 0.10, minRepS: 0.5 },
  pull_up: { signalKind: "elbow_angle", peakDirection: "trough", minAmplitude: 0.20, minRepS: 0.7 },
  push_up: { signalKind: "elbow_angle", peakDirection: "trough", minAmplitude: 0.15, minRepS: 0.6 },
  // Demo-driven addition v0.2.0: single-DB elbow flexion. Same triplet as
  // bench/OHP. minAmplitude=0.18 (≈32° elbow change) is the smallest "real"
  // half-curl we want to count; full ROM curls are ~110-150°.
  dumbbell_bicep_curl: { signalKind: "elbow_angle", peakDirection: "trough", minAmplitude: 0.18, minRepS: 0.5 },
  // cyclic_angle (backend rep_signal_joint = right_hip)
  romanian_deadlift: { signalKind: "hip_angle", peakDirection: "trough", minAmplitude: 0.10, minRepS: 0.7 },
  // basketball: realtime-only release counter; "rep" = one shot release
  basketball: { signalKind: "vertical_wrist", peakDirection: "peak", minAmplitude: 0.10, minRepS: 0.6 },
  jump_shot: { signalKind: "vertical_wrist", peakDirection: "peak", minAmplitude: 0.10, minRepS: 0.6 },
};

/** Public: list of exercises supported by the realtime ghost counter.
 * Used by the picker so we never offer an exercise we cannot count. */
export function isRealtimeSupported(exerciseId: string): boolean {
  return exerciseId in EXERCISE_CONFIGS;
}

// ---------------------------------------------------------------------------
// Signal extraction
// ---------------------------------------------------------------------------

const SIGNAL_MIN_VISIBILITY = 0.3;

/**
 * Extract the primary 1-D rep signal for the given exercise.
 *
 * Returns ``null`` when the required landmarks are too occluded
 * (visibility < SIGNAL_MIN_VISIBILITY).  The caller treats null as a
 * dropout and resets the gap timer; sustained dropout past
 * SIGNAL_GAP_RESET_MS triggers a state machine reset.
 */
export function extractSignal(
  landmarks: NormalizedLandmark[],
  exerciseId: string,
): number | null {
  const cfg = EXERCISE_CONFIGS[exerciseId];
  if (!cfg) return null;
  const get = (idx: number) => landmarks[idx];

  switch (cfg.signalKind) {
    case "vertical_hip": {
      const lh = get(LM.LEFT_HIP), rh = get(LM.RIGHT_HIP);
      if (!lh || !rh || minVis(lh, rh) < SIGNAL_MIN_VISIBILITY) return null;
      return 1 - avgY(lh, rh);
    }
    case "vertical_knee": {
      // Match backend `rep_signal_joint = right_knee` for walking_lunge.
      // Falls back to left_knee if right is occluded so the user can stand
      // in either direction relative to the camera.
      const rk = get(LM.RIGHT_KNEE), lk = get(LM.LEFT_KNEE);
      const rOk = rk && (rk.visibility ?? 0) >= SIGNAL_MIN_VISIBILITY;
      const lOk = lk && (lk.visibility ?? 0) >= SIGNAL_MIN_VISIBILITY;
      if (!rOk && !lOk) return null;
      const k = rOk ? rk! : lk!;
      return 1 - k.y;
    }
    case "vertical_wrist": {
      // Basketball release: maximum wrist height (top of follow-through).
      // We use the dominant wrist (whichever is higher physically /
      // smaller image-y) so left- and right-handed shooters both work.
      const lw = get(LM.LEFT_WRIST), rw = get(LM.RIGHT_WRIST);
      const lOk = lw && (lw.visibility ?? 0) >= SIGNAL_MIN_VISIBILITY;
      const rOk = rw && (rw.visibility ?? 0) >= SIGNAL_MIN_VISIBILITY;
      if (!lOk && !rOk) return null;
      let y: number;
      if (lOk && rOk) y = Math.min(lw!.y, rw!.y);
      else if (lOk) y = lw!.y;
      else y = rw!.y;
      return 1 - y;
    }
    case "elbow_angle": {
      // Mirror backend rep_features triplet: shoulder-elbow-wrist.  Use
      // the side with better visibility (the user's "camera-facing" arm).
      const ls = get(LM.LEFT_SHOULDER), le = get(LM.LEFT_ELBOW), lw = get(LM.LEFT_WRIST);
      const rs = get(LM.RIGHT_SHOULDER), re = get(LM.RIGHT_ELBOW), rw = get(LM.RIGHT_WRIST);
      const leftVis = ls && le && lw ? minVis(ls, le, lw) : 0;
      const rightVis = rs && re && rw ? minVis(rs, re, rw) : 0;
      if (Math.max(leftVis, rightVis) < SIGNAL_MIN_VISIBILITY) return null;
      const angle = leftVis >= rightVis
        ? angle2d(ls!, le!, lw!)
        : angle2d(rs!, re!, rw!);
      return angle / 180;
    }
    case "hip_angle": {
      // Romanian deadlift: hinge angle at the hip (shoulder-hip-knee
      // triplet) mirrors backend `rep_signal_joint = right_hip` with
      // _ANGLE_TRIPLETS["right_hip"] = (right_shoulder, right_hip,
      // right_knee).  At standing this is ~180° (trunk vertical, thigh
      // vertical) → ~1.0; at the bottom of an RDL the trunk hinges to
      // ~90° while the leg stays straight → angle ~90° → ~0.5.  Trough
      // detection therefore catches max hinge.
      const ls = get(LM.LEFT_SHOULDER), lh = get(LM.LEFT_HIP), lk = get(LM.LEFT_KNEE);
      const rs = get(LM.RIGHT_SHOULDER), rh = get(LM.RIGHT_HIP), rk = get(LM.RIGHT_KNEE);
      const leftVis = ls && lh && lk ? minVis(ls, lh, lk) : 0;
      const rightVis = rs && rh && rk ? minVis(rs, rh, rk) : 0;
      if (Math.max(leftVis, rightVis) < SIGNAL_MIN_VISIBILITY) return null;
      const angle = leftVis >= rightVis
        ? angle2d(ls!, lh!, lk!)
        : angle2d(rs!, rh!, rk!);
      return angle / 180;
    }
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// EMA smoother
// ---------------------------------------------------------------------------

export interface EmaState { value: number; alpha: number; }

export function makeEma(alpha = 0.25): EmaState {
  return { value: 0, alpha };
}

export function updateEma(state: EmaState, sample: number): number {
  state.value = state.alpha * sample + (1 - state.alpha) * state.value;
  return state.value;
}

// ---------------------------------------------------------------------------
// Public output types
// ---------------------------------------------------------------------------

export type Phase = "rest" | "eccentric" | "concentric";

export interface GhostField {
  value: number | null;
  unit: string;
  status: "valid" | "degraded" | "unknown";
  reason_codes: string[];
}

export interface GhostRepMetrics {
  rep_index: number;
  start_ts: number;
  end_ts: number;
  rep_duration_s: GhostField;
  eccentric_duration_s: GhostField;
  concentric_duration_s: GhostField;
  tempo_ratio_ecc_over_con: GhostField;
  signal_amplitude: GhostField;
  min_visibility: GhostField;
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

interface RepInProgress {
  rep_index: number;
  /** Boundary timestamp at which this rep started (start of eccentric). */
  start_ts: number;
  /** Timestamp at which the rep-defining extremum was crossed. */
  peak_ts: number | null;
  /** Tracked min/max of the smoothed signal across the rep window. */
  signal_min: number;
  signal_max: number;
  /** Min visibility on the canonical primary joints during this rep. */
  min_vis: number;
}

export type LatchedDirection = "up" | "down" | "none";

export interface RepCounterState {
  ema: EmaState;
  prevSmoothed: number;
  /** Schmitt-trigger output: stays committed through dead-zone crossings. */
  latchedDirection: LatchedDirection;
  currentRep: RepInProgress | null;
  completedReps: GhostRepMetrics[];
  /** Total reps that PASSED all quality gates. This is what the user sees. */
  repCount: number;
  currentPhase: Phase;
  currentSignal: number | null;
  lastSignalTs: number | null;
  /** Number of consecutive frames where the signal was usable AND
   * primary-joint visibility was high.  Counting starts once this
   * exceeds WARMUP_FRAMES — prevents "getting into push-up position"
   * from being mistaken for a rep. */
  warmupFrames: number;
}

export function makeRepCounterState(): RepCounterState {
  return {
    ema: makeEma(0.25),
    prevSmoothed: 0,
    latchedDirection: "none",
    currentRep: null,
    completedReps: [],
    repCount: 0,
    currentPhase: "rest",
    currentSignal: null,
    lastSignalTs: null,
    warmupFrames: 0,
  };
}

// ---------------------------------------------------------------------------
// Tunable thresholds (Schmitt-trigger style hysteresis + quality gates)
// ---------------------------------------------------------------------------

/** |smoothed delta| > ENTER commits to a new direction; below that the
 * latched direction is preserved.  Empirically tuned at 30 fps with EMA
 * α = 0.25.  Smaller numbers → more sensitive to noise → over-counting. */
const DELTA_THRESHOLD_ENTER = 0.005;
const DELTA_THRESHOLD_EXIT = 0.002;

/** Sustained signal dropout that triggers a state-machine reset
 * (clears latched direction and any in-progress rep). */
const SIGNAL_GAP_RESET_MS = 500;

/** Number of consecutive good-quality frames required before counting
 * begins.  ~0.5 s at 30 fps.  Lets the user get into position and stops
 * us counting "lowering into plank" as a push-up. */
const WARMUP_FRAMES = 15;

/** Minimum primary-joint visibility for a frame to count toward warm-up
 * AND to be eligible to start a rep. */
const PRIMARY_JOINT_VIS_GATE = 0.4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function field(
  value: number | null,
  unit: string,
  status: GhostField["status"],
): GhostField {
  return { value, unit, status, reason_codes: ["realtime_preview"] };
}

/** Fail-validate a candidate rep against the four quality gates. */
function validateRep(
  rep: RepInProgress,
  endTs: number,
  cfg: ExerciseRepConfig,
): { ok: true } | { ok: false; reason: string } {
  const durS = (endTs - rep.start_ts) / 1000;
  const amp = rep.signal_max - rep.signal_min;
  if (durS < cfg.minRepS) return { ok: false, reason: "too_short" };
  if (durS > 8) return { ok: false, reason: "too_long" };
  if (amp < cfg.minAmplitude) return { ok: false, reason: "low_amplitude" };
  if (rep.min_vis < PRIMARY_JOINT_VIS_GATE) return { ok: false, reason: "low_visibility" };
  return { ok: true };
}

function buildCompletedRep(
  rep: RepInProgress,
  endTs: number,
  endVis: number,
): GhostRepMetrics {
  const peakTs = rep.peak_ts as number;
  const durS = (endTs - rep.start_ts) / 1000;
  const eccS = (peakTs - rep.start_ts) / 1000;
  const conS = (endTs - peakTs) / 1000;
  const ratio = conS > 0.05 ? eccS / conS : null;
  const repVis = Math.min(rep.min_vis, endVis);
  const amplitude = rep.signal_max - rep.signal_min;

  // A rep that PASSED gating is conservatively "valid" on duration /
  // amplitude.  Visibility status reflects per-rep min visibility.
  const visStatus: GhostField["status"] =
    repVis >= 0.6 ? "valid" : repVis >= 0.4 ? "degraded" : "unknown";

  return {
    rep_index: rep.rep_index,
    start_ts: rep.start_ts,
    end_ts: endTs,
    rep_duration_s: field(parseFloat(durS.toFixed(3)), "s", "valid"),
    eccentric_duration_s: field(parseFloat(eccS.toFixed(3)), "s",
      eccS >= 0.1 ? "valid" : "degraded"),
    concentric_duration_s: field(parseFloat(conS.toFixed(3)), "s",
      conS >= 0.1 ? "valid" : "degraded"),
    tempo_ratio_ecc_over_con: field(ratio !== null ? parseFloat(ratio.toFixed(2)) : null,
      "ratio", ratio !== null ? "valid" : "unknown"),
    signal_amplitude: field(parseFloat(amplitude.toFixed(3)), "norm", "valid"),
    min_visibility: field(parseFloat(repVis.toFixed(3)), "fraction", visStatus),
  };
}

// ---------------------------------------------------------------------------
// Main entry point: feedFrame
// ---------------------------------------------------------------------------

/**
 * Advance the rep counter by one frame.  Returns the newly completed rep
 * (with all metrics) iff a rep just completed, else null.
 *
 * State machine:
 *
 *   1. extractSignal → null on poor visibility → record dropout.
 *   2. Gap > SIGNAL_GAP_RESET_MS since last good signal → state reset.
 *   3. Smooth (EMA) → compute delta.
 *   4. Hysteresis direction:  |delta| > ENTER commits "up" or "down";
 *      otherwise latchedDirection is preserved.
 *   5. Warm-up: only after WARMUP_FRAMES of good visibility do we accept
 *      transitions.  Otherwise we just track signal and return.
 *   6. Direction transition vs. latched direction:
 *        - START crossing  → if currentRep already had a peak, validate
 *          + maybe emit; then open a new currentRep.
 *        - PEAK crossing   → mark peak_ts on currentRep (provisional).
 *   7. Track signal min/max/min_vis throughout the rep window so we can
 *      validate amplitude + visibility at completion.
 *
 * The user-visible repCount only increments inside the validation step
 * — a rep that fails any quality gate is silently discarded.
 */
export function feedFrame(
  state: RepCounterState,
  landmarks: NormalizedLandmark[],
  exerciseId: string,
  timestampMs: number,
): GhostRepMetrics | null {
  const cfg = EXERCISE_CONFIGS[exerciseId];
  if (!cfg) {
    state.currentSignal = null;
    return null;
  }

  const raw = extractSignal(landmarks, exerciseId);
  if (raw === null) {
    state.currentSignal = null;
    // Don't update lastSignalTs so the next valid frame sees the gap.
    return null;
  }

  // ---- Visibility on the canonical core joints (used by the gate
  // AND surfaced as min_visibility on the emitted rep). ---------------
  const vis = Math.min(
    ...landmarks
      .filter((_, i) => [
        LM.LEFT_HIP, LM.RIGHT_HIP,
        LM.LEFT_KNEE, LM.RIGHT_KNEE,
        LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
        LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
      ].includes(i))
      .map((l) => l.visibility ?? 0),
  );

  // ---- 1. Gap-reset on returning signal -------------------------------
  const gap =
    state.lastSignalTs === null ? Infinity : timestampMs - state.lastSignalTs;
  const isFreshStart = gap > SIGNAL_GAP_RESET_MS;
  if (isFreshStart) {
    state.ema.value = raw;
    state.prevSmoothed = raw;
    state.latchedDirection = "none";
    state.warmupFrames = 0;
    // Drop any half-formed rep that was interrupted by the dropout.
    state.currentRep = null;
  }

  // ---- 2. Smooth + delta ---------------------------------------------
  const smoothed = updateEma(state.ema, raw);
  const delta = smoothed - state.prevSmoothed;
  state.currentSignal = smoothed;
  state.lastSignalTs = timestampMs;

  // ---- 3. Direction with hysteresis ----------------------------------
  let direction: LatchedDirection = state.latchedDirection;
  if (delta > DELTA_THRESHOLD_ENTER) direction = "up";
  else if (delta < -DELTA_THRESHOLD_ENTER) direction = "down";

  // ---- 4. Display phase (independent of latched direction) ----------
  let newPhase: Phase = state.currentPhase;
  if (delta > DELTA_THRESHOLD_ENTER) newPhase = "concentric";
  else if (delta < -DELTA_THRESHOLD_ENTER) newPhase = "eccentric";
  else if (Math.abs(delta) < DELTA_THRESHOLD_EXIT) newPhase = "rest";

  // ---- 5. Warm-up gate ------------------------------------------------
  // Count consecutive good-vis frames.  Only after WARMUP_FRAMES do we
  // allow transitions to start/end reps.
  if (vis >= PRIMARY_JOINT_VIS_GATE) {
    state.warmupFrames = Math.min(state.warmupFrames + 1, WARMUP_FRAMES + 5);
  } else {
    state.warmupFrames = 0;
    // Visibility just dropped — abort any in-progress rep so we don't
    // emit a half-tracked metric whose visibility data is stale.
    state.currentRep = null;
  }
  const inWarmup = state.warmupFrames < WARMUP_FRAMES;

  // ---- 6. Track signal min/max + visibility on the active rep --------
  if (state.currentRep !== null) {
    if (smoothed < state.currentRep.signal_min) state.currentRep.signal_min = smoothed;
    if (smoothed > state.currentRep.signal_max) state.currentRep.signal_max = smoothed;
    if (vis < state.currentRep.min_vis) state.currentRep.min_vis = vis;
  }

  // ---- 7. Boundary state machine -------------------------------------
  let completedRep: GhostRepMetrics | null = null;

  if (!isFreshStart && !inWarmup) {
    // The "start direction" is the direction the signal moves in during
    // the eccentric phase: toward the work apex.
    const startDirection: LatchedDirection = cfg.peakDirection === "peak" ? "up" : "down";

    const isFirstCommit =
      state.latchedDirection === "none" &&
      direction === startDirection &&
      state.currentRep === null;

    const directionChanged =
      state.latchedDirection !== "none" &&
      direction !== "none" &&
      direction !== state.latchedDirection;

    if (isFirstCommit) {
      state.currentRep = openRep(state.repCount, timestampMs, smoothed, vis);
    } else if (directionChanged) {
      const peakDir = cfg.peakDirection;

      const wasStartCrossing =
        (peakDir === "peak" && state.latchedDirection === "down" && direction === "up") ||
        (peakDir === "trough" && state.latchedDirection === "up" && direction === "down");

      const wasPeakCrossing =
        (peakDir === "peak" && state.latchedDirection === "up" && direction === "down") ||
        (peakDir === "trough" && state.latchedDirection === "down" && direction === "up");

      if (wasStartCrossing) {
        // Close out the previous rep iff it had a peak AND passes gates.
        if (state.currentRep !== null && state.currentRep.peak_ts !== null) {
          const verdict = validateRep(state.currentRep, timestampMs, cfg);
          if (verdict.ok) {
            completedRep = buildCompletedRep(state.currentRep, timestampMs, vis);
            state.completedReps.push(completedRep);
            state.repCount++;
          }
          // else: silently discarded.  No count, no emission.
        }
        state.currentRep = openRep(state.repCount, timestampMs, smoothed, vis);
      } else if (wasPeakCrossing) {
        if (state.currentRep !== null) {
          state.currentRep.peak_ts = timestampMs;
        }
      }
    }
  }

  // ---- 8. Latch direction + advance ----------------------------------
  if (direction !== "none") state.latchedDirection = direction;
  state.prevSmoothed = smoothed;
  state.currentPhase = newPhase;

  return completedRep;
}

function openRep(
  repIndex: number,
  startTs: number,
  signal: number,
  vis: number,
): RepInProgress {
  return {
    rep_index: repIndex,
    start_ts: startTs,
    peak_ts: null,
    signal_min: signal,
    signal_max: signal,
    min_vis: vis,
  };
}

// ---------------------------------------------------------------------------
// Public helpers for tests and UI
// ---------------------------------------------------------------------------

/** Get the per-exercise config (read-only) for use in UI hints. */
export function getExerciseConfig(exerciseId: string): ExerciseRepConfig | null {
  return EXERCISE_CONFIGS[exerciseId] ?? null;
}

// ---------------------------------------------------------------------------
// Wire-format helper for the parity probe (app/parity/realtime.py probe_reps).
//
// The canonical backend (app/gym/rep_features.py) emits:
//   * primary_joints_min_visibility   (visibility 0..1)
//   * signal_amplitude in DEGREES for cyclic_angle exercises
//   * signal_amplitude in NORMALIZED_Y for cyclic_vertical exercises
//
// The ghost counter internally uses the more-readable name `min_visibility`
// and tracks `signal_amplitude` in normalised [0,1] units across all
// exercises (because its 1-D signal is `angle/180` for cyclic_angle and
// `1 - y` for cyclic_vertical).  When the page uploads the ghost vector
// to the backend, we MUST translate both the field name AND the unit so
// the parity probe compares apples to apples.
//
// Bug history: shipping with mis-named `min_visibility` and unit-mismatched
// `signal_amplitude` made the parity_probe block silently degrade to
// `insufficient_data` (visibility) or `outside_tolerance` (amplitude with a
// 100x unit skew), even when the two pipelines actually agreed.
// ---------------------------------------------------------------------------

/** Wire-format ghost rep vector matching the v1 envelope contract. */
export interface GhostRepWireVector {
  rep_index: number;
  features: Record<string, GhostField>;
}

/** Translate one in-memory GhostRepMetrics → wire-format vector for the
 * parity probe.  Returns the fields canonical also emits, in the units
 * canonical also emits them.  Fields the canonical pipeline does not
 * compute are deliberately omitted (probe_reps would skip them anyway). */
export function toWireVector(
  rep: GhostRepMetrics,
  exerciseId: string,
): GhostRepWireVector {
  const cfg = EXERCISE_CONFIGS[exerciseId];
  const isCyclicAngle =
    cfg?.signalKind === "elbow_angle" || cfg?.signalKind === "hip_angle";

  // signal_amplitude unit conversion: ghost stores [0,1] regardless of
  // signal kind. Canonical reports degrees for cyclic_angle.
  const ampValue = rep.signal_amplitude.value;
  const wireAmpValue =
    ampValue === null
      ? null
      : isCyclicAngle
        ? parseFloat((ampValue * 180).toFixed(2))   // → degrees
        : parseFloat(ampValue.toFixed(4));           // → normalized_y
  const wireAmpUnit = isCyclicAngle ? "deg" : "normalized_y";

  return {
    rep_index: rep.rep_index,
    features: {
      rep_duration_s: rep.rep_duration_s,
      eccentric_duration_s: rep.eccentric_duration_s,
      concentric_duration_s: rep.concentric_duration_s,
      tempo_ratio_ecc_over_con: rep.tempo_ratio_ecc_over_con,
      signal_amplitude: {
        value: wireAmpValue,
        unit: wireAmpUnit,
        status: rep.signal_amplitude.status,
        reason_codes: rep.signal_amplitude.reason_codes,
      },
      // Rename to match canonical app.gym.rep_features field name so the
      // parity probe can pair them (it matches purely on field-name string).
      primary_joints_min_visibility: {
        value: rep.min_visibility.value,
        unit: "visibility",
        status: rep.min_visibility.status,
        reason_codes: rep.min_visibility.reason_codes,
      },
    },
  };
}
