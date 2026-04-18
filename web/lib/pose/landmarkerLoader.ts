/**
 * Lazy singleton loader for MediaPipe PoseLandmarker (LIVE_STREAM mode).
 *
 * Loading is deferred to first call so the ~3 MB model + WASM bundle does
 * not block the page render.  Subsequent calls return the same instance via
 * a cached Promise, so concurrent mount calls are safe.
 *
 * WASM and model are served from the jsDelivr CDN so they don't enter the
 * Next.js bundle (WASM via webpack is complex and adds ~2 MB to the chunk).
 * The Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy headers in
 * next.config.js enable SharedArrayBuffer for the WASM runtime.
 */

import {
  FilesetResolver,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";

const TASKS_VISION_VERSION = "0.10.34";
const WASM_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

let _promise: Promise<PoseLandmarker> | null = null;

/**
 * Return a PoseLandmarker ready for LIVE_STREAM detection.
 * Safe to call from multiple components simultaneously.
 */
export function loadPoseLandmarker(): Promise<PoseLandmarker> {
  if (_promise) return _promise;
  _promise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_CDN);
    return PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  })();
  return _promise;
}

/** Discard the cached instance (call on hot-reload in dev only). */
export function _resetLandmarkerCache(): void {
  _promise = null;
}
