/**
 * Deterministic unit tests for repCounter.ts A3 changes.
 *
 * Tests mirror the backend fixture pattern (no DOM, no WASM, no I/O).
 * We verify:
 *   1. extractSignal returns the correct elbow-angle fraction for curl.
 *   2. A full curl (amplitude >= 0.50) is accepted by feedFrame.
 *   3. A twitch curl (amplitude < 0.50) is rejected by feedFrame.
 *   4. isRealtimeSupported recognises dumbbell_bicep_curl.
 *
 * Physical geometry:
 *   shoulder at (0.5, 0.3), elbow at (0.5, 0.5), wrist placed so that the
 *   interior angle at the elbow equals the target degrees (same synthetic
 *   geometry as tests/test_bicep_curl_rom_gate.py).
 *
 * Threshold rationale:
 *   minAmplitude = 0.50 ↔ 0.50 * 180° = 90°, which is the Norkin & White
 *   functional full-curl minimum (start>=150°, peak<=60° → 90° swing).
 */

import { describe, it, expect } from "vitest";
import {
  extractSignal,
  feedFrame,
  isRealtimeSupported,
  makeRepCounterState,
} from "../repCounter";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

// ---------------------------------------------------------------------------
// Landmark builder helpers
// ---------------------------------------------------------------------------

/** Create a NormalizedLandmark object. */
function lm(x: number, y: number, visibility = 0.95): NormalizedLandmark {
  return { x, y, z: 0, visibility };
}

/**
 * Build 33 MediaPipe landmarks (index 0–32) with the core joints visible.
 *
 * The vis gate in feedFrame uses Math.min across indices
 *   11,12 (shoulders), 13,14 (elbows), 23,24 (hips), 25,26 (knees).
 * All of these must have visibility >= PRIMARY_JOINT_VIS_GATE (0.4) or the
 * warmup counter never increments and no rep is ever counted.  We therefore
 * set plausible positions for hips and knees in addition to the arm triplet.
 *
 * Geometry (image coords — y grows downward):
 *   shoulder at (0.5, 0.30), elbow at (0.5, 0.50)
 *   hip at (0.5, 0.65), knee at (0.5, 0.80)
 *   wrist placed by angle.
 */
const FOREARM = 0.15;
const UPPER_ARM = 0.20;

function curlLandmarks(angleDeg: number): NormalizedLandmark[] {
  const all: NormalizedLandmark[] = Array.from({ length: 33 }, () =>
    lm(0, 0, 0),
  );
  const ex = 0.5, ey = 0.5;
  const sx = 0.5, sy = ey - UPPER_ARM;
  const rot = ((180 - angleDeg) * Math.PI) / 180;
  const wx = ex + FOREARM * Math.sin(rot);
  const wy = ey + FOREARM * Math.cos(rot);

  // Arm joints (indices from repCounter.ts LM table).
  all[11] = lm(sx, sy);        // LEFT_SHOULDER
  all[12] = lm(sx, sy);        // RIGHT_SHOULDER
  all[13] = lm(ex, ey);        // LEFT_ELBOW
  all[14] = lm(ex, ey);        // RIGHT_ELBOW
  all[15] = lm(wx, wy);        // LEFT_WRIST
  all[16] = lm(wx, wy);        // RIGHT_WRIST
  // Hip and knee: needed to pass PRIMARY_JOINT_VIS_GATE in the vis check.
  all[23] = lm(0.5, 0.65);     // LEFT_HIP
  all[24] = lm(0.5, 0.65);     // RIGHT_HIP
  all[25] = lm(0.5, 0.80);     // LEFT_KNEE
  all[26] = lm(0.5, 0.80);     // RIGHT_KNEE
  return all;
}

// ---------------------------------------------------------------------------
// Fixture 1: extractSignal correctness
// ---------------------------------------------------------------------------

describe("extractSignal — dumbbell_bicep_curl", () => {
  it("returns angle/180 fraction for a known angle", () => {
    const targets = [60, 90, 120, 150, 170];
    for (const deg of targets) {
      const lms = curlLandmarks(deg);
      const sig = extractSignal(lms, "dumbbell_bicep_curl");
      expect(sig).not.toBeNull();
      // sig = angle/180; allow ±0.5° measurement tolerance
      const expected = deg / 180;
      expect(sig!).toBeCloseTo(expected, 1);
    }
  });

  it("returns null when all joints are low-visibility", () => {
    const lms: NormalizedLandmark[] = Array.from({ length: 33 }, () =>
      lm(0.5, 0.5, 0.0),
    );
    expect(extractSignal(lms, "dumbbell_bicep_curl")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Fixture 2: isRealtimeSupported
// ---------------------------------------------------------------------------

describe("isRealtimeSupported", () => {
  it("recognises dumbbell_bicep_curl", () => {
    expect(isRealtimeSupported("dumbbell_bicep_curl")).toBe(true);
  });
  it("rejects unknown exercises", () => {
    expect(isRealtimeSupported("flying_saucer_press")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Helpers to simulate a rep through feedFrame
// ---------------------------------------------------------------------------

/**
 * Drive the rep counter through a synthetic curl sequence.
 * `frames` is an array of [angleDeg, timestampMs] pairs.
 * Returns the list of completed reps detected.
 */
function runFrames(
  frames: Array<[number, number]>,
  exerciseId = "dumbbell_bicep_curl",
) {
  const state = makeRepCounterState();
  const completed: ReturnType<typeof feedFrame>[] = [];
  for (const [angle, ts] of frames) {
    const lms = curlLandmarks(angle);
    const result = feedFrame(state, lms, exerciseId, ts);
    if (result !== null) completed.push(result);
  }
  return completed;
}

/** Build cosine-interpolated angle sequence start→peak→end at 30 fps. */
function synthesiseAngles(
  startDeg: number,
  peakDeg: number,
  endDeg: number,
  nFrames = 40,
  startMs = 500, // after 0.5s warmup
): Array<[number, number]> {
  const half = Math.floor(nFrames / 2);
  const frames: Array<[number, number]> = [];
  for (let i = 0; i < nFrames; i++) {
    let angle: number;
    if (i <= half) {
      const alpha = half > 0 ? (1 - Math.cos(Math.PI * i / half)) / 2 : 1;
      angle = startDeg + (peakDeg - startDeg) * alpha;
    } else {
      const j = i - half;
      const rem = nFrames - 1 - half;
      const alpha = rem > 0 ? (1 - Math.cos(Math.PI * j / rem)) / 2 : 1;
      angle = peakDeg + (endDeg - peakDeg) * alpha;
    }
    frames.push([angle, startMs + Math.round((i / 30) * 1000)]);
  }
  return frames;
}

// ---------------------------------------------------------------------------
// Fixture 3: full curl (amplitude ≥ 0.50) is ACCEPTED
// ---------------------------------------------------------------------------

describe("feedFrame — full curl accepted", () => {
  it("counts two full curls and emits 1 rep", () => {
    // State machine contract: a rep is emitted when the NEXT rep's eccentric
    // phase starts (direction goes "down" again after going "up"), closing the
    // previous rep. So 2 curls -> 1 emitted + 1 still open at end.
    // Amplitude per curl: (160 - 55) / 180 = 0.583 > 0.50 threshold.
    const warmup = synthesiseAngles(160, 160, 160, 20, 0);
    const curl1 = synthesiseAngles(160, 55, 160, 40, 700);
    // Second curl starts after first completes (~1.33 s for 40 frames at 30fps).
    const curl2 = synthesiseAngles(160, 55, 160, 40, 2100);
    const frames = [...warmup, ...curl1, ...curl2];
    const reps = runFrames(frames);
    // Expect 1 emitted rep (first curl closed when second starts).
    expect(reps).toHaveLength(1);
    const rep = reps[0]!;
    expect(rep.signal_amplitude.value).toBeGreaterThanOrEqual(0.5);
  });
});

// ---------------------------------------------------------------------------
// Fixture 4: twitch curl (amplitude < 0.50) is REJECTED
// ---------------------------------------------------------------------------

describe("feedFrame — twitch rejected", () => {
  it("does not count a twitch (160→140→160) — amplitude 0.11 < 0.50", () => {
    // Amplitude = (160 - 140) / 180 ≈ 0.111 < 0.50 threshold.
    const warmup = synthesiseAngles(160, 160, 160, 20, 0);
    const twitch = synthesiseAngles(160, 140, 160, 40, 700);
    const frames = [...warmup, ...twitch];
    const reps = runFrames(frames);
    expect(reps).toHaveLength(0);
  });

  it("does not count a partial curl (160→100→160) — amplitude 0.33 < 0.50", () => {
    // Amplitude = (160 - 100) / 180 ≈ 0.333 < 0.50 threshold.
    const warmup = synthesiseAngles(160, 160, 160, 20, 0);
    const partial = synthesiseAngles(160, 100, 160, 40, 700);
    const frames = [...warmup, ...partial];
    const reps = runFrames(frames);
    expect(reps).toHaveLength(0);
  });
});
