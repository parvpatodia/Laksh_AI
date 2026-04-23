"use client";

/**
 * PoseCamera: webcam feed + MediaPipe LIVE_STREAM skeleton overlay.
 *
 * Responsibilities
 * ----------------
 * 1. Request camera access (getUserMedia).
 * 2. Feed frames into PoseLandmarker (LIVE_STREAM, lite model).
 * 3. Draw skeleton on a canvas overlay at ~30 FPS via requestAnimationFrame.
 * 4. Call onLandmarks on every detected frame so parent can run ghost metrics.
 * 5. Expose startCapture / stopCapture for MediaRecorder clip recording.
 *    - Records the raw camera stream (no overlay burned in).
 *    - Returns a Blob to onCaptureComplete so the parent can POST it.
 *
 * The landmarker is a process-wide singleton (landmarkerLoader.ts) so
 * navigating between sports does not reload the 3 MB model.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import { loadPoseLandmarker } from "@/lib/pose/landmarkerLoader";
import { drawSkeleton } from "@/lib/pose/skeletonDraw";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PoseCameraProps {
  /** Show/hide the component. When false the camera is released. */
  active: boolean;
  /** Called each frame with normalised landmarks (33 points, COCO-17 order). */
  onLandmarks?: (landmarks: NormalizedLandmark[], timestampMs: number) => void;
  /** Called when a MediaRecorder capture finishes. */
  onCaptureComplete?: (blob: Blob, mimeType: string) => void;
  /** Called when camera permission is denied or an unrecoverable error occurs. */
  onError?: (message: string) => void;
  /**
   * Auto-stop recording after this many seconds. Defaults to 6.
   * Shorter clips = less MediaPipe work = faster analysis (~40s vs 3+ min).
   * A 6s clip at 30 fps gives ~180 frames — enough for 3-5 clean reps.
   */
  maxDurationS?: number;
}

type CameraState = "idle" | "requesting" | "ready" | "error";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function PoseCamera({
  active,
  onLandmarks,
  onCaptureComplete,
  onError,
  maxDurationS = 6,
}: PoseCameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const rafRef = useRef<number>(0);
  const landmarkerReadyRef = useRef(false);
  // Auto-stop timer and elapsed counter for the recording countdown.
  const autoStopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const recordingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [isRecording, setIsRecording] = useState(false);
  const [landmarkerLoaded, setLandmarkerLoaded] = useState(false);
  const [fps, setFps] = useState<number>(0);
  // Seconds elapsed since recording started (drives the countdown badge).
  const [recordingElapsed, setRecordingElapsed] = useState(0);

  // FPS counter (rolling average over 30 frames).
  const fpsBuffer = useRef<number[]>([]);
  const lastFrameTs = useRef<number>(0);

  // ---------------------------------------------------------------------------
  // Landmarker init (singleton, fires once per page lifetime)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    loadPoseLandmarker()
      .then(() => {
        landmarkerReadyRef.current = true;
        setLandmarkerLoaded(true);
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        onError?.(`MediaPipe load failed: ${msg}`);
      });
    // No cleanup: singleton lives for the page lifetime.
  }, [onError]);

  // ---------------------------------------------------------------------------
  // Camera lifecycle
  // ---------------------------------------------------------------------------
  const startCamera = useCallback(async () => {
    setCameraState("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraState("ready");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setCameraState("error");
      onError?.(msg.includes("Permission") ? "Camera permission denied." : `Camera error: ${msg}`);
    }
  }, [onError]);

  const stopCamera = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraState("idle");
    setIsRecording(false);
  }, []);

  useEffect(() => {
    if (active) {
      startCamera();
    } else {
      stopCamera();
    }
    return stopCamera;
  }, [active, startCamera, stopCamera]);

  // ---------------------------------------------------------------------------
  // Detection loop (requestAnimationFrame)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (cameraState !== "ready" || !landmarkerReadyRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let stopped = false;

    const loop = async () => {
      if (stopped || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      // Sync canvas size to video intrinsic dimensions.
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth || 1280;
        canvas.height = video.videoHeight || 720;
      }

      const now = performance.now();

      // FPS
      if (lastFrameTs.current > 0) {
        fpsBuffer.current.push(1000 / (now - lastFrameTs.current));
        if (fpsBuffer.current.length > 30) fpsBuffer.current.shift();
        const avg = fpsBuffer.current.reduce((a, b) => a + b, 0) / fpsBuffer.current.length;
        setFps(Math.round(avg));
      }
      lastFrameTs.current = now;

      // VIDEO mode: detectForVideo returns synchronously.
      const landmarker = await loadPoseLandmarker();
      const result = landmarker.detectForVideo(video, now);
      const pose = result.landmarks?.[0];
      if (!pose) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      } else {
        drawSkeleton(ctx, pose, canvas.width, canvas.height);
        onLandmarks?.(pose, now);
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => {
      stopped = true;
      cancelAnimationFrame(rafRef.current);
    };
  }, [cameraState, landmarkerLoaded, onLandmarks]);

  // ---------------------------------------------------------------------------
  // MediaRecorder
  // ---------------------------------------------------------------------------
  const stopCapture = useCallback(() => {
    // Clear auto-stop timer and elapsed counter before stopping.
    if (autoStopTimerRef.current !== null) {
      clearTimeout(autoStopTimerRef.current);
      autoStopTimerRef.current = null;
    }
    if (recordingIntervalRef.current !== null) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }
    recorderRef.current?.stop();
    recorderRef.current = null;
    setRecordingElapsed(0);
  }, []);

  const startCapture = useCallback(() => {
    if (!streamRef.current || isRecording) return;

    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : "video/webm";

    const recorder = new MediaRecorder(streamRef.current, { mimeType });
    const chunks: Blob[] = [];

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mimeType });
      onCaptureComplete?.(blob, mimeType);
      setIsRecording(false);
      setRecordingElapsed(0);
    };

    recorder.start(100); // 100ms timeslices for low-latency chunks
    recorderRef.current = recorder;
    setIsRecording(true);
    setRecordingElapsed(0);

    // Elapsed counter — updates every second for the countdown badge.
    recordingIntervalRef.current = setInterval(() => {
      setRecordingElapsed((prev) => prev + 1);
    }, 1000);

    // Auto-stop after maxDurationS to keep clips short and analysis fast.
    autoStopTimerRef.current = setTimeout(() => {
      recorderRef.current?.stop();
      recorderRef.current = null;
      if (recordingIntervalRef.current !== null) {
        clearInterval(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }
    }, maxDurationS * 1000);
  }, [isRecording, maxDurationS, onCaptureComplete]);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="flex flex-col gap-4">
      {/* Camera viewport.
          aspect-[3/4] (portrait-ish) instead of 16:9 so a standing full body
          fits without cropping. object-contain (not cover) so the entire
          camera frame is visible -- judges can see exactly what the model
          sees, including their feet.
          On large screens we let the box grow up to ~80vh so a full standing
          shot is comfortably visible from across a research-showcase booth. */}
      <div className="relative w-full mx-auto aspect-[3/4] sm:aspect-[4/5] lg:aspect-[16/10]
                      max-h-[80vh] rounded-2xl overflow-hidden border border-surface-700 bg-black">
        {/* Video */}
        <video
          ref={videoRef}
          className="w-full h-full object-contain"
          playsInline
          muted
          style={{ transform: "scaleX(-1)" }} // mirror for natural selfie view
        />

        {/* Skeleton overlay (object-contain on the video means the canvas
            must letterbox the same way; we still draw in video-pixel space
            inside the canvas, so just match container size). */}
        <canvas
          ref={canvasRef}
          className="pose-overlay object-contain"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Framing tip overlay (only while idle; disappears once camera ready) */}
        {cameraState === "idle" && (
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2
                          bg-black/60 text-slate-300 text-xs px-3 py-1.5 rounded-full
                          border border-surface-600 whitespace-nowrap">
            Stand 6&#8211;10 ft back, full body in frame, ball or dumbbell in hand
          </div>
        )}

        {/* Status overlays */}
        {cameraState === "idle" && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface-900/80">
            <div className="text-center text-slate-400">
              <p className="text-4xl mb-2">📷</p>
              <p className="text-sm">Click Start to begin</p>
            </div>
          </div>
        )}

        {cameraState === "requesting" && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface-900/80">
            <p className="text-sm text-slate-400 animate-pulse">Requesting camera…</p>
          </div>
        )}

        {cameraState === "error" && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface-900/90">
            <p className="text-sm text-rose-400">Camera unavailable</p>
          </div>
        )}

        {/* Loading badge */}
        {cameraState === "ready" && !landmarkerLoaded && (
          <div className="absolute top-3 left-3 chip-preview text-xs px-2 py-1 rounded">
            Loading pose model…
          </div>
        )}

        {/* FPS badge */}
        {cameraState === "ready" && landmarkerLoaded && (
          <div className="absolute top-3 left-3 bg-black/50 text-slate-400 text-xs px-2 py-1 rounded font-mono">
            {fps} fps
          </div>
        )}

        {/* Recording badge — shows live countdown to auto-stop */}
        {isRecording && (
          <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-rose-900/80 text-rose-300 text-xs px-2 py-1 rounded">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400 animate-pulse-slow" />
            REC · {Math.max(0, maxDurationS - recordingElapsed)}s
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        {!active ? null : cameraState === "idle" ? (
          <button
            onClick={startCamera}
            className="px-5 py-2.5 rounded-lg bg-brand-500 text-white text-sm font-medium
                       hover:bg-brand-600 transition-colors"
          >
            Start
          </button>
        ) : cameraState === "ready" && !isRecording ? (
          <button
            onClick={startCapture}
            disabled={!landmarkerLoaded}
            className="px-5 py-2.5 rounded-lg bg-rose-600 text-white text-sm font-medium
                       hover:bg-rose-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Record
          </button>
        ) : isRecording ? (
          <button
            onClick={stopCapture}
            className="px-5 py-2.5 rounded-lg bg-surface-700 text-slate-200 text-sm font-medium
                       hover:bg-surface-600 transition-colors border border-rose-700"
          >
            Stop & Analyse
          </button>
        ) : null}

        {cameraState === "ready" && (
          <button
            onClick={stopCamera}
            className="px-4 py-2.5 rounded-lg text-slate-400 text-sm hover:text-slate-200
                       transition-colors"
          >
            Stop camera
          </button>
        )}

        {cameraState === "ready" && (
          <span className="text-xs text-slate-600 ml-auto">
            {isRecording
              ? `Recording — auto-stops in ${Math.max(0, maxDurationS - recordingElapsed)}s`
              : landmarkerLoaded
                ? `Pose tracking active · will record up to ${maxDurationS}s`
                : "Loading pose model (~3 MB)…"}
          </span>
        )}
      </div>
    </div>
  );
}
