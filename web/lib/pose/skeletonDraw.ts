/**
 * Canvas skeleton drawing utilities for MediaPipe PoseLandmarker output.
 *
 * Keeps drawing logic separate from the React component so it can be
 * unit-tested and reused by any canvas consumer.
 *
 * VIS_THRESHOLD is set to 0.3 (not 0.5) because the LITE model in the
 * browser returns lower visibility scores than the HEAVY model used
 * server-side. 0.3 is the effective "confident" boundary for LITE.
 */

import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import { PoseLandmarker } from "@mediapipe/tasks-vision";

/** Colours for the skeleton overlay (designed for dark background). */
const LANDMARK_COLOR     = "rgba(14, 165, 233, 0.95)";  // brand-500 solid
const CONNECTION_COLOR   = "rgba(14, 165, 233, 0.55)";  // brand-500 dimmer
const LOW_VIS_COLOR      = "rgba(148, 163, 184, 0.4)";  // slate-400 muted (not red — less alarming)
const BBOX_COLOR         = "rgba(14, 165, 233, 0.55)";  // brand-500 for bounding box
const BBOX_CORNER_COLOR  = "rgba(14, 165, 233, 0.95)";  // brand-500 solid for corners

/**
 * Minimum MediaPipe LITE visibility to draw a landmark as solid.
 * 0.3 matches the LITE model's actual operating range.
 * The HEAVY model (server-side) uses 0.5 but LITE visibility is uncalibrated there.
 */
const VIS_THRESHOLD = 0.3;

/**
 * Draw PoseLandmarker results onto ctx.
 * Also draws a bounding box around the detected person so users can
 * see exactly what the model has locked onto.
 *
 * @param ctx       - 2D canvas context (canvas must match video dimensions)
 * @param landmarks - normalised landmark array from one detected pose
 * @param width     - canvas display width
 * @param height    - canvas display height
 */
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
): void {
  ctx.clearRect(0, 0, width, height);

  // -------------------------------------------------------------------
  // 1. Bounding box around the detected person
  // -------------------------------------------------------------------
  // Compute tight bbox from all landmarks with visibility above threshold.
  const visLandmarks = landmarks.filter((lm) => (lm.visibility ?? 0) >= VIS_THRESHOLD);
  if (visLandmarks.length >= 4) {
    const xs = visLandmarks.map((lm) => lm.x * width);
    const ys = visLandmarks.map((lm) => lm.y * height);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    // Add 5% padding on each side.
    const padX = (maxX - minX) * 0.12;
    const padY = (maxY - minY) * 0.12;
    const bx = Math.max(0, minX - padX);
    const by = Math.max(0, minY - padY);
    const bw = Math.min(width - bx, maxX - minX + padX * 2);
    const bh = Math.min(height - by, maxY - minY + padY * 2);

    // Dashed bounding box.
    ctx.save();
    ctx.strokeStyle = BBOX_COLOR;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(bx, by, bw, bh);
    ctx.restore();

    // Corner accent marks (L-shaped, 16px long).
    const cornerLen = 16;
    ctx.save();
    ctx.strokeStyle = BBOX_CORNER_COLOR;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([]);
    const corners: [number, number, number, number, number, number, number, number][] = [
      [bx, by, bx + cornerLen, by, bx, by, bx, by + cornerLen],
      [bx + bw - cornerLen, by, bx + bw, by, bx + bw, by, bx + bw, by + cornerLen],
      [bx, by + bh - cornerLen, bx, by + bh, bx, by + bh, bx + cornerLen, by + bh],
      [bx + bw - cornerLen, by + bh, bx + bw, by + bh, bx + bw, by + bh - cornerLen, bx + bw, by + bh],
    ];
    for (const [x1, y1, x2, y2, x3, y3, x4, y4] of corners) {
      ctx.beginPath();
      ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
      ctx.moveTo(x3, y3); ctx.lineTo(x4, y4);
      ctx.stroke();
    }
    ctx.restore();
  }

  // -------------------------------------------------------------------
  // 2. Connections (drawn under landmarks)
  // -------------------------------------------------------------------
  ctx.lineWidth = 2;
  for (const conn of PoseLandmarker.POSE_CONNECTIONS) {
    const start = landmarks[conn.start];
    const end = landmarks[conn.end];
    if (!start || !end) continue;
    const vis = Math.min(start.visibility ?? 0, end.visibility ?? 0);
    ctx.strokeStyle = vis >= VIS_THRESHOLD ? CONNECTION_COLOR : "rgba(255,255,255,0.08)";
    ctx.beginPath();
    ctx.moveTo(start.x * width, start.y * height);
    ctx.lineTo(end.x * width, end.y * height);
    ctx.stroke();
  }

  // -------------------------------------------------------------------
  // 3. Landmark dots
  // -------------------------------------------------------------------
  for (const lm of landmarks) {
    const vis = lm.visibility ?? 0;
    ctx.fillStyle = vis >= VIS_THRESHOLD ? LANDMARK_COLOR : LOW_VIS_COLOR;
    ctx.beginPath();
    ctx.arc(lm.x * width, lm.y * height, vis >= VIS_THRESHOLD ? 4 : 2, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * Compute the fraction of core landmarks (shoulders, elbows, wrists)
 * with visibility >= VIS_THRESHOLD.  Used by PoseCamera to show a
 * live detection quality indicator to the user.
 *
 * @returns 0.0–1.0 (1.0 = all core joints detected)
 */
export function coreDetectionFraction(landmarks: NormalizedLandmark[]): number {
  // Indices: 11=L_shoulder, 12=R_shoulder, 13=L_elbow, 14=R_elbow,
  //          15=L_wrist, 16=R_wrist
  const CORE = [11, 12, 13, 14, 15, 16];
  const detected = CORE.filter((i) => (landmarks[i]?.visibility ?? 0) >= VIS_THRESHOLD).length;
  return detected / CORE.length;
}
