/**
 * Canvas skeleton drawing utilities for MediaPipe PoseLandmarker output.
 *
 * Keeps drawing logic separate from the React component so it can be
 * unit-tested and reused by any canvas consumer.
 */

import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import { PoseLandmarker } from "@mediapipe/tasks-vision";

/** Colours for the skeleton overlay (designed for dark background). */
const LANDMARK_COLOR = "rgba(14, 165, 233, 0.9)";   // brand-500
const CONNECTION_COLOR = "rgba(14, 165, 233, 0.45)"; // brand-500 dimmer
const LOW_VIS_COLOR = "rgba(239, 68, 68, 0.6)";      // red-500 for degraded

/** Minimum MediaPipe visibility to draw a landmark as solid. */
const VIS_THRESHOLD = 0.5;

/**
 * Draw PoseLandmarker results onto *ctx*.
 *
 * @param ctx   - 2D canvas context (canvas must match video dimensions)
 * @param landmarks - normalised landmark array from one detected pose
 * @param width  - canvas display width
 * @param height - canvas display height
 */
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
): void {
  ctx.clearRect(0, 0, width, height);

  // Draw connections first (under landmarks).
  // Connection is {start: number, end: number} in this version of tasks-vision.
  ctx.lineWidth = 2;
  for (const conn of PoseLandmarker.POSE_CONNECTIONS) {
    const start = landmarks[conn.start];
    const end = landmarks[conn.end];
    if (!start || !end) continue;
    const vis = Math.min(start.visibility ?? 0, end.visibility ?? 0);
    ctx.strokeStyle = vis >= VIS_THRESHOLD ? CONNECTION_COLOR : "rgba(255,255,255,0.1)";
    ctx.beginPath();
    ctx.moveTo(start.x * width, start.y * height);
    ctx.lineTo(end.x * width, end.y * height);
    ctx.stroke();
  }

  // Draw landmark dots.
  for (const lm of landmarks) {
    const vis = lm.visibility ?? 0;
    ctx.fillStyle = vis >= VIS_THRESHOLD ? LANDMARK_COLOR : LOW_VIS_COLOR;
    ctx.beginPath();
    ctx.arc(lm.x * width, lm.y * height, vis >= VIS_THRESHOLD ? 4 : 2, 0, Math.PI * 2);
    ctx.fill();
  }
}
