/**
 * Lazy singleton loader for MediaPipe PoseLandmarker (VIDEO mode).
 *
 * Loading is deferred to first call so the ~3 MB model + WASM bundle does
 * not block the page render. Subsequent calls return the same instance via
 * a cached Promise, so concurrent mount calls are safe.
 *
 * WASM and model are served from the jsDelivr CDN so they don't enter the
 * Next.js bundle (WASM via webpack is complex and adds ~2 MB to the chunk).
 * The Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy headers in
 * next.config.js enable SharedArrayBuffer for the WASM runtime.
 *
 * Delegate choice: CPU (not GPU).
 * The GPU delegate is faster on high-end machines but unreliable on
 * integrated graphics / Chromium sandboxed GPU contexts (common on laptops).
 * When the GPU delegate silently falls back or misconfigures, landmark
 * quality degrades severely. CPU delegate is slower (~20-30 fps vs ~60 fps)
 * but deterministically correct on all hardware.
 *
 * Confidence thresholds: 0.3 (not 0.5).
 * The LITE model's per-landmark `visibility` field is poorly calibrated
 * compared to the HEAVY model used server-side. Visibility of 0.1-0.3 is
 * normal for clearly-visible joints under the LITE model. Using 0.3
 * detection/tracking thresholds gives reliable detection under indoor
 * lighting and typical webcam angles without increasing false positives.
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
 * Return a PoseLandmarker ready for VIDEO-mode detection.
 * Safe to call from multiple components simultaneously.
 */
export function loadPoseLandmarker(): Promise<PoseLandmarker> {
  if (_promise) return _promise;
  _promise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_CDN);
    return PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        // CPU: reliable across all laptop GPUs. GPU is faster but fails
        // silently on integrated graphics, degrading landmark quality.
        delegate: "CPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
      // 0.3 matches the LITE model's actual operating range.
      // The LITE model visibility field is uncalibrated at the 0.5 level;
      // 0.3 gives detection parity with HEAVY at 0.5 in practice.
      minPoseDetectionConfidence: 0.3,
      minPosePresenceConfidence: 0.3,
      minTrackingConfidence: 0.3,
      outputSegmentationMasks: false, // not needed; saves ~20% inference time
    });
  })();
  return _promise;
}

/** Discard the cached instance (call on hot-reload in dev only). */
export function _resetLandmarkerCache(): void {
  _promise = null;
}
