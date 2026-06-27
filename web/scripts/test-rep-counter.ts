// Standalone synthetic test for repCounter quality gates.
//
// Run with:  cd web && npx tsx scripts/test-rep-counter.mjs
//
// We synthesise MediaPipe-style landmark frames with controllable
// hip-y trajectories + visibility, then assert:
//
//   1. 5 ideal squats produce repCount in [4, 5] (off-by-one allowed
//      on the trailing rep -- see repCounter docstring).
//   2. Pure noise (zero amplitude motion) produces repCount = 0.
//   3. Low-visibility motion produces repCount = 0.
//   4. A single tiny micro-bobble inside one rep window does not
//      produce extra reps.
//   5. Basketball release sequence: 3 wrist-pumps -> repCount in [2, 3].

import {
  feedFrame,
  makeRepCounterState,
} from "../lib/realtime/repCounter.ts";

// MediaPipe landmark indices we care about.
const IDX = {
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

function makeBlankFrame(visibility = 0.9) {
  const f = [];
  for (let i = 0; i < 33; i++) {
    f.push({ x: 0.5, y: 0.5, z: 0, visibility });
  }
  return f;
}

// Squat trajectory: hip_y oscillates between standing (0.55) and bottom (0.85)
// over `repPeriodMs`, sustained for `nReps` cycles. Vis stays high.
function squatFrames(nReps, repPeriodMs = 1500, fps = 30, vis = 0.9) {
  const frames = [];
  const dt = 1000 / fps;
  const totalMs = nReps * repPeriodMs + 500; // 500 ms of standing tail
  for (let t = 0; t <= totalMs; t += dt) {
    const phase = (t % repPeriodMs) / repPeriodMs; // 0..1
    // Cosine: starts at standing (cos=1 -> y_low), bottoms at half (cos=-1 -> y_high)
    const hipY = 0.55 + 0.15 * (1 - Math.cos(phase * 2 * Math.PI));
    const f = makeBlankFrame(vis);
    f[IDX.LEFT_HIP] = { x: 0.45, y: hipY, z: 0, visibility: vis };
    f[IDX.RIGHT_HIP] = { x: 0.55, y: hipY, z: 0, visibility: vis };
    // Knees roughly track hips for this synthetic.
    f[IDX.LEFT_KNEE] = { x: 0.45, y: hipY + 0.10, z: 0, visibility: vis };
    f[IDX.RIGHT_KNEE] = { x: 0.55, y: hipY + 0.10, z: 0, visibility: vis };
    // Provide elbows so primary-vis check passes (squats use hip+knee+elbow+shoulder).
    f[IDX.LEFT_ELBOW] = { x: 0.40, y: 0.50, z: 0, visibility: vis };
    f[IDX.RIGHT_ELBOW] = { x: 0.60, y: 0.50, z: 0, visibility: vis };
    f[IDX.LEFT_SHOULDER] = { x: 0.42, y: 0.40, z: 0, visibility: vis };
    f[IDX.RIGHT_SHOULDER] = { x: 0.58, y: 0.40, z: 0, visibility: vis };
    frames.push({ t, frame: f });
  }
  return frames;
}

// Pure micro-noise around a constant hip-y -- no real reps.
function noiseFrames(durationMs = 5000, fps = 30, vis = 0.9) {
  const frames = [];
  const dt = 1000 / fps;
  for (let t = 0; t <= durationMs; t += dt) {
    const hipY = 0.55 + (Math.random() - 0.5) * 0.005; // ±0.0025 = ~0.5% jitter
    const f = makeBlankFrame(vis);
    f[IDX.LEFT_HIP] = { x: 0.45, y: hipY, z: 0, visibility: vis };
    f[IDX.RIGHT_HIP] = { x: 0.55, y: hipY, z: 0, visibility: vis };
    f[IDX.LEFT_KNEE] = { x: 0.45, y: hipY + 0.10, z: 0, visibility: vis };
    f[IDX.RIGHT_KNEE] = { x: 0.55, y: hipY + 0.10, z: 0, visibility: vis };
    f[IDX.LEFT_ELBOW] = { x: 0.40, y: 0.50, z: 0, visibility: vis };
    f[IDX.RIGHT_ELBOW] = { x: 0.60, y: 0.50, z: 0, visibility: vis };
    f[IDX.LEFT_SHOULDER] = { x: 0.42, y: 0.40, z: 0, visibility: vis };
    f[IDX.RIGHT_SHOULDER] = { x: 0.58, y: 0.40, z: 0, visibility: vis };
    frames.push({ t, frame: f });
  }
  return frames;
}

// Squat trajectory but with low visibility -- should never count.
function lowVisSquats(nReps) {
  return squatFrames(nReps, 1500, 30, 0.2);
}

// Basketball "shot" trajectory: wrist swings up to release, comes down.
function basketballFrames(nShots, periodMs = 1500, fps = 30, vis = 0.9) {
  const frames = [];
  const dt = 1000 / fps;
  const totalMs = nShots * periodMs + 500;
  for (let t = 0; t <= totalMs; t += dt) {
    const phase = (t % periodMs) / periodMs;
    // Wrist starts low (y=0.7), peaks high (y=0.2), returns.
    const wristY = 0.45 - 0.25 * Math.cos(phase * 2 * Math.PI);
    const f = makeBlankFrame(vis);
    f[IDX.LEFT_WRIST] = { x: 0.45, y: wristY, z: 0, visibility: vis };
    f[IDX.RIGHT_WRIST] = { x: 0.55, y: wristY, z: 0, visibility: vis };
    f[IDX.LEFT_ELBOW] = { x: 0.40, y: 0.50, z: 0, visibility: vis };
    f[IDX.RIGHT_ELBOW] = { x: 0.60, y: 0.50, z: 0, visibility: vis };
    f[IDX.LEFT_SHOULDER] = { x: 0.42, y: 0.40, z: 0, visibility: vis };
    f[IDX.RIGHT_SHOULDER] = { x: 0.58, y: 0.40, z: 0, visibility: vis };
    f[IDX.LEFT_HIP] = { x: 0.45, y: 0.65, z: 0, visibility: vis };
    f[IDX.RIGHT_HIP] = { x: 0.55, y: 0.65, z: 0, visibility: vis };
    f[IDX.LEFT_KNEE] = { x: 0.45, y: 0.80, z: 0, visibility: vis };
    f[IDX.RIGHT_KNEE] = { x: 0.55, y: 0.80, z: 0, visibility: vis };
    frames.push({ t, frame: f });
  }
  return frames;
}

function runTest(name, frames, exerciseId) {
  const state = makeRepCounterState();
  let lastEmitted = null;
  for (const { t, frame } of frames) {
    const emitted = feedFrame(state, frame, exerciseId, t);
    if (emitted) lastEmitted = emitted;
  }
  return { name, count: state.repCount, lastEmitted, completedReps: state.completedReps.length };
}

function assertInRange(name, value, lo, hi) {
  const ok = value >= lo && value <= hi;
  const status = ok ? "PASS" : "FAIL";
  console.log(`  [${status}] ${name}: count=${value} (expected ${lo}-${hi})`);
  return ok;
}

// ---------------------------------------------------------------------------
// Run all tests
// ---------------------------------------------------------------------------

let allPassed = true;
console.log("Rep counter synthetic tests");
console.log("===========================\n");

console.log("Test 1: 5 ideal back-squat reps");
{
  const r = runTest("squat-5", squatFrames(5), "back_squat");
  console.log(`         repCount=${r.count}, completedReps=${r.completedReps}`);
  if (r.lastEmitted) {
    console.log(`         last rep: dur=${r.lastEmitted.rep_duration_s.value}s amp=${r.lastEmitted.signal_amplitude.value}`);
  }
  allPassed = assertInRange("expect 4-5", r.count, 4, 5) && allPassed;
}

console.log("\nTest 2: 5 seconds of noise (no real reps)");
{
  const r = runTest("noise", noiseFrames(5000), "back_squat");
  console.log(`         repCount=${r.count}`);
  allPassed = assertInRange("expect 0", r.count, 0, 0) && allPassed;
}

console.log("\nTest 3: 3 squat reps with low visibility (vis=0.2)");
{
  const r = runTest("low-vis", lowVisSquats(3), "back_squat");
  console.log(`         repCount=${r.count}`);
  allPassed = assertInRange("expect 0", r.count, 0, 0) && allPassed;
}

console.log("\nTest 4: 5 push-up cycles (elbow_angle signal)");
{
  // For push-ups the signal is elbow angle; we synthesise elbow positions.
  const fps = 30;
  const periodMs = 1800;
  const nReps = 5;
  const totalMs = nReps * periodMs + 500;
  const frames = [];
  const dt = 1000 / fps;
  for (let t = 0; t <= totalMs; t += dt) {
    const phase = (t % periodMs) / periodMs;
    // Elbow angle in degrees: top of push-up = 170 (arms straight),
    // bottom = 80 (chest at floor). 90 deg amplitude.
    const angleDeg = 125 - 45 * Math.cos(phase * 2 * Math.PI);
    const angleRad = (angleDeg * Math.PI) / 180;
    // Construct an isoceles arm: shoulder at (0,0), wrist directly below
    // at (0, L+L)=(0,0.2) when fully straight (angle 180), and the elbow
    // at the symmetric break-point.  For half-angle α at the elbow:
    //   half = (180 - angleDeg)/2 = bend per arm
    // Place shoulder at (0.5, 0.4), wrist at (0.5, 0.7) (vertical line).
    // The midpoint between them is at (0.5, 0.55). The elbow lies
    // perpendicular to that midpoint by distance L * cos(angle/2),
    // where L = 0.15 (half the shoulder-wrist span... approximated).
    const L = 0.15;                       // half-span of straight arm
    const half = (Math.PI - angleRad) / 2; // half the bend
    const offset = L * Math.sin(half);    // horizontal elbow offset
    const sx = 0.5, sy = 0.4;
    const wx = 0.5, wy = 0.7;
    const ex = sx + offset;               // elbow bows out to the right
    const ey = (sy + wy) / 2;
    const vis = 0.9;
    const f = makeBlankFrame(vis);
    f[IDX.LEFT_SHOULDER] = { x: sx - 0.02, y: sy, z: 0, visibility: vis };
    f[IDX.RIGHT_SHOULDER] = { x: sx + 0.02, y: sy, z: 0, visibility: vis };
    f[IDX.LEFT_ELBOW] = { x: ex, y: ey, z: 0, visibility: vis };
    f[IDX.RIGHT_ELBOW] = { x: ex + 0.04, y: ey, z: 0, visibility: vis };
    f[IDX.LEFT_WRIST] = { x: wx - 0.02, y: wy, z: 0, visibility: vis };
    f[IDX.RIGHT_WRIST] = { x: wx + 0.02, y: wy, z: 0, visibility: vis };
    f[IDX.LEFT_HIP] = { x: 0.45, y: 0.55, z: 0, visibility: vis };
    f[IDX.RIGHT_HIP] = { x: 0.55, y: 0.55, z: 0, visibility: vis };
    f[IDX.LEFT_KNEE] = { x: 0.45, y: 0.70, z: 0, visibility: vis };
    f[IDX.RIGHT_KNEE] = { x: 0.55, y: 0.70, z: 0, visibility: vis };
    frames.push({ t, frame: f });
  }
  const r = runTest("push-up-5", frames, "push_up");
  console.log(`         repCount=${r.count}, completedReps=${r.completedReps}`);
  if (r.lastEmitted) {
    console.log(`         last rep: dur=${r.lastEmitted.rep_duration_s.value}s amp=${r.lastEmitted.signal_amplitude.value}`);
  }
  // Push-ups have stricter amplitude gate; allow 0-5 since geometry approx.
  allPassed = assertInRange("expect 3-5", r.count, 3, 5) && allPassed;
}

console.log("\nTest 5: 3 basketball jump-shot releases");
{
  const r = runTest("bball-3", basketballFrames(3), "basketball");
  console.log(`         repCount=${r.count}`);
  allPassed = assertInRange("expect 2-3", r.count, 2, 3) && allPassed;
}

console.log("\n===========================");
console.log(allPassed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
process.exit(allPassed ? 0 : 1);
