"use client";

/**
 * PoseCamera: webcam feed + MediaPipe LIVE_STREAM skeleton overlay.
 *
 * Responsibilities
 * 1. Request camera access (getUserMedia).
 * 2. Feed frames into PoseLandmarker (LIVE_STREAM, lite model).
 * 3. Draw skeleton on a canvas overlay at ~30 FPS via requestAnimationFrame.
 * 4. Call onLandmarks on every detected frame so the parent can run ghost metrics.
 * 5. Expose startCapture / stopCapture for MediaRecorder clip recording.
 *    Returns a Blob to onCaptureComplete so the parent can POST it.
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
import { drawSkeleton, coreDetectionFraction } from "@/lib/pose/skeletonDraw";

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
   * Auto-stop recording after this many seconds. Defaults to 15.
   * A 15 s clip at 30 fps gives ~450 frames - enough for multiple reps or shots.
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
  maxDurationS = 15,
}: PoseCameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const rafRef = useRef<number>(0);
  const landmarkerReadyRef = useRef(false);
  const autoStopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const recordingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [isRecording, setIsRecording] = useState(false);
  const [landmarkerLoaded, setLandmarkerLoaded] = useState(false);
  const [fps, setFps] = useState<number>(0);
  const [recordingElapsed, setRecordingElapsed] = useState(0);
  // Detection quality: 0.0-1.0 (fraction of core joints confident).
  // null = camera not yet ready. 0 = camera ready but no person in frame.
  const [detectionQuality, setDetectionQuality] = useState<number | null>(null);
  // How many consecutive frames had zero detection (drives "no person" warning).
  const noPoseFramesRef = useRef(0);
  const [showNoPoseWarning, setShowNoPoseWarning] = useState(false);

  const fpsBuffer = useRef<number[]>([]);
  const lastFrameTs = useRef<number>(0);

  // ---------------------------------------------------------------------------
  // Landmarker init
  // ---------------------------------------------------------------------------
  useEffect(() => {
    loadPoseLandmarker()
      .then(() => {
        landmarkerReadyRef.current = true;
        setLandmarkerLoaded(true);
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        onError?.(`Pose model failed to load: ${msg}`);
      });
  }, [onError]);

  // ---------------------------------------------------------------------------
  // Camera lifecycle
  // ---------------------------------------------------------------------------
  const startCamera = useCallback(async () => {
    setCameraState("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      // Request minimum zoom (widest FOV) if the device supports it.
      try {
        const track = stream.getVideoTracks()[0];
        if (track) {
          const caps = track.getCapabilities() as MediaTrackCapabilities & {
            zoom?: { min: number; max: number; step: number };
          };
          if (caps.zoom) {
            await track.applyConstraints({
              advanced: [{ zoom: caps.zoom.min } as MediaTrackConstraintSet],
            });
          }
        }
      } catch {
        // Zoom unsupported - proceed normally.
      }

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
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraState("idle");
    setIsRecording(false);
    setDetectionQuality(null);
    setShowNoPoseWarning(false);
    noPoseFramesRef.current = 0;
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
  // Detection loop
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

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth || 1280;
        canvas.height = video.videoHeight || 720;
      }

      const now = performance.now();

      // FPS counter
      if (lastFrameTs.current > 0) {
        fpsBuffer.current.push(1000 / (now - lastFrameTs.current));
        if (fpsBuffer.current.length > 30) fpsBuffer.current.shift();
        const avg = fpsBuffer.current.reduce((a, b) => a + b, 0) / fpsBuffer.current.length;
        setFps(Math.round(avg));
      }
      lastFrameTs.current = now;

      const landmarker = await loadPoseLandmarker();
      const result = landmarker.detectForVideo(video, now);
      const pose = result.landmarks?.[0];

      if (!pose) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        setDetectionQuality(0);
        // Show "no person" warning after 20 consecutive empty frames (~0.7 s).
        // This avoids flashing the warning for a single dropped frame.
        noPoseFramesRef.current += 1;
        if (noPoseFramesRef.current >= 20) setShowNoPoseWarning(true);
      } else {
        noPoseFramesRef.current = 0;
        setShowNoPoseWarning(false);
        drawSkeleton(ctx, pose, canvas.width, canvas.height);
        onLandmarks?.(pose, now);
        setDetectionQuality((prev) => {
          const q = coreDetectionFraction(pose);
          return prev === null ? q : prev * 0.8 + q * 0.2;
        });
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

    recorder.start(100);
    recorderRef.current = recorder;
    setIsRecording(true);
    setRecordingElapsed(0);

    recordingIntervalRef.current = setInterval(() => {
      setRecordingElapsed((prev) => prev + 1);
    }, 1000);

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
  const timeLeft = Math.max(0, maxDurationS - recordingElapsed);
  const pct = detectionQuality !== null ? Math.round(detectionQuality * 100) : null;

  const qualityColor = pct === null ? "" :
    pct === 0   ? "bg-rose-900/70 text-rose-300 border border-rose-700/40" :
    pct >= 80   ? "bg-emerald-900/60 text-emerald-300 border border-emerald-700/40" :
    pct >= 50   ? "bg-amber-900/60 text-amber-300 border border-amber-700/40" :
                  "bg-rose-900/60 text-rose-300 border border-rose-700/40";

  const qualityDot = pct === null ? "" :
    pct === 0   ? "bg-rose-400" :
    pct >= 80   ? "bg-emerald-400" :
    pct >= 50   ? "bg-amber-400" :
                  "bg-rose-400";

  return (
    <div className="flex flex-col gap-3">
      {/* Camera viewport */}
      <div className="relative w-full mx-auto aspect-[3/4] sm:aspect-[4/5] lg:aspect-[16/10]
                      max-h-[80vh] rounded-2xl overflow-hidden border border-surface-700 bg-black">
        {/* Video feed */}
        <video
          ref={videoRef}
          className="w-full h-full object-contain"
          playsInline
          muted
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Skeleton overlay */}
        <canvas
          ref={canvasRef}
          className="pose-overlay object-contain"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Idle state */}
        {cameraState === "idle" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-surface-900/90 gap-3">
            <div className="rounded-full border border-surface-600 bg-surface-800 p-5">
              <svg className="w-8 h-8 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round"
                      d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z" />
              </svg>
            </div>
            <p className="text-sm text-slate-500">Press Start to activate camera</p>
          </div>
        )}

        {/* Requesting */}
        {cameraState === "requesting" && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface-900/80">
            <p className="text-sm text-slate-400 animate-pulse">Requesting camera access...</p>
          </div>
        )}

        {/* Error */}
        {cameraState === "error" && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface-900/90">
            <p className="text-sm text-rose-400">Camera unavailable</p>
          </div>
        )}

        {/* Loading pose model */}
        {cameraState === "ready" && !landmarkerLoaded && (
          <div className="absolute top-3 left-3 text-xs px-2.5 py-1 rounded bg-surface-800/80 text-slate-400 border border-surface-600">
            Loading pose model...
          </div>
        )}

        {/* No person detected warning - overlaid in the centre of the frame */}
        {cameraState === "ready" && landmarkerLoaded && showNoPoseWarning && (
          <div className="absolute inset-0 flex items-end justify-center pb-16 pointer-events-none">
            <div className="flex items-center gap-2 bg-rose-950/90 border border-rose-700/60
                            text-rose-300 text-sm px-4 py-2.5 rounded-xl backdrop-blur-sm">
              <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round"
                      d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
              </svg>
              No person detected - step into frame
            </div>
          </div>
        )}

        {/* FPS + detection quality badge */}
        {cameraState === "ready" && landmarkerLoaded && (
          <div className="absolute top-3 left-3 flex items-center gap-1.5">
            <div className="bg-black/60 text-slate-400 text-xs px-2 py-1 rounded font-mono border border-white/5">
              {fps} fps
            </div>
            {pct !== null && (
              <div className={`text-xs px-2 py-1 rounded font-mono flex items-center gap-1.5 ${qualityColor}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${qualityDot} ${pct > 0 && pct < 50 ? "animate-pulse" : ""}`} />
                Pose {pct}%
              </div>
            )}
          </div>
        )}

        {/* Recording badge */}
        {isRecording && (
          <div className="absolute top-3 right-3 flex items-center gap-1.5
                          bg-rose-950/90 border border-rose-700/50 text-rose-300 text-xs px-2.5 py-1 rounded-lg">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400 animate-pulse" />
            REC {timeLeft}s
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
                       hover:bg-rose-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Record
          </button>
        ) : isRecording ? (
          <button
            onClick={stopCapture}
            className="px-5 py-2.5 rounded-lg border border-rose-700/60 bg-surface-700
                       text-slate-200 text-sm font-medium hover:bg-surface-600 transition-colors"
          >
            Stop and Analyse
          </button>
        ) : null}

        {cameraState === "ready" && (
          <button
            onClick={stopCamera}
            className="px-4 py-2.5 rounded-lg text-slate-400 text-sm hover:text-slate-200 transition-colors"
          >
            Stop camera
          </button>
        )}

        {cameraState === "ready" && (
          <span className="text-xs text-slate-600 ml-auto">
            {isRecording
              ? `Recording - stops in ${timeLeft}s`
              : landmarkerLoaded
                ? `Tracking active - records up to ${maxDurationS}s`
                : "Loading pose model (3 MB)..."}
          </span>
        )}
      </div>
    </div>
  );
}
