"use client";

/**
 * /basketball  -> capture page for basketball jump shot
 * /gym         -> exercise picker, then capture page
 *
 * Day 5: PoseCamera (LIVE_STREAM skeleton).
 * Day 6: repCounter + GhostMetricsPanel.
 * Day 7: clip upload -> /v1/analyze/gym/video -> CanonicalReport.
 * Day 8: parity probe surfaced.
 */

import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

import {
  feedFrame,
  makeRepCounterState,
  type GhostRepMetrics,
  type Phase,
  type RepCounterState,
} from "@/lib/realtime/repCounter";
import { analyzeGymVideo, type AnalyzeResponse } from "@/lib/api";
import GhostMetricsPanel from "@/components/GhostMetricsPanel";
import CanonicalReport from "@/components/CanonicalReport";

const PoseCamera = dynamic(() => import("@/components/PoseCamera"), {
  ssr: false,
  loading: () => (
    <div className="w-full aspect-video rounded-2xl border border-surface-700 bg-surface-900 flex items-center justify-center">
      <p className="text-sm text-slate-600">Loading camera…</p>
    </div>
  ),
});

const GYM_EXERCISES: { id: string; label: string }[] = [
  { id: "back_squat",        label: "Back Squat" },
  { id: "front_squat",       label: "Front Squat" },
  { id: "deadlift",          label: "Deadlift" },
  { id: "romanian_deadlift", label: "Romanian Deadlift" },
  { id: "bench_press",       label: "Bench Press" },
  { id: "overhead_press",    label: "Overhead Press" },
  { id: "barbell_row",       label: "Barbell Row" },
  { id: "pull_up",           label: "Pull-up" },
  { id: "dumbbell_curl",     label: "Dumbbell Curl" },
  { id: "tricep_pushdown",   label: "Tricep Pushdown" },
  { id: "lunge",             label: "Lunge" },
  { id: "hip_thrust",        label: "Hip Thrust" },
];

type UploadStatus = "idle" | "uploading" | "done" | "error";

function SportPageInner() {
  const params = useParams<{ sport: string }>();
  const searchParams = useSearchParams();
  const router = useRouter();

  const sport = params.sport as "basketball" | "gym";
  const exerciseId = searchParams.get("exercise") ?? (sport === "basketball" ? "basketball" : null);

  // Camera
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Capture
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);
  const [capturedMime, setCapturedMime] = useState("video/webm");

  // Upload / canonical result
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>("idle");
  const [uploadError, setUploadError] = useState<string | undefined>();
  const [canonicalResult, setCanonicalResult] = useState<AnalyzeResponse | null>(null);

  // Rep counter (ref so RAF mutations don't trigger re-renders)
  const repStateRef = useRef<RepCounterState>(makeRepCounterState());
  const [repCount, setRepCount] = useState(0);
  const [currentPhase, setCurrentPhase] = useState<Phase>("rest");
  const [currentSignal, setCurrentSignal] = useState<number | null>(null);
  const [lastRep, setLastRep] = useState<GhostRepMetrics | null>(null);

  const handleLandmarks = useCallback(
    (landmarks: NormalizedLandmark[], ts: number) => {
      if (!exerciseId) return;
      const completed = feedFrame(repStateRef.current, landmarks, exerciseId, ts);
      const s = repStateRef.current;
      setCurrentSignal(s.currentSignal);
      setCurrentPhase(s.currentPhase);
      if (s.repCount !== repCount) setRepCount(s.repCount);
      if (completed) setLastRep(completed);
    },
    [exerciseId, repCount],
  );

  const handleCaptureComplete = useCallback((blob: Blob, mime: string) => {
    setCapturedBlob(blob);
    setCapturedMime(mime);
    setCameraActive(false);
  }, []);

  const handleUpload = useCallback(async () => {
    if (!capturedBlob || !exerciseId) return;
    setUploadStatus("uploading");
    setUploadError(undefined);
    try {
      const result = await analyzeGymVideo(capturedBlob, exerciseId, capturedMime);
      setCanonicalResult(result);
      setUploadStatus("done");
    } catch (err: unknown) {
      setUploadStatus("error");
      setUploadError(err instanceof Error ? err.message : String(err));
    }
  }, [capturedBlob, capturedMime, exerciseId]);

  const resetAll = useCallback(() => {
    setCapturedBlob(null);
    setCanonicalResult(null);
    setUploadStatus("idle");
    setUploadError(undefined);
    setCameraError(null);
    repStateRef.current = makeRepCounterState();
    setRepCount(0);
    setCurrentPhase("rest");
    setCurrentSignal(null);
    setLastRep(null);
    setCameraActive(false);
  }, []);

  // Unknown sport
  if (sport !== "basketball" && sport !== "gym") {
    return (
      <div className="max-w-2xl mx-auto px-6 py-24 text-center">
        <p className="text-slate-400">Unknown sport: {sport}</p>
        <a href="/" className="mt-4 inline-block text-brand-500 hover:underline">Back to home</a>
      </div>
    );
  }

  // Gym exercise picker
  if (sport === "gym" && !exerciseId) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-12">
        <div className="mb-8">
          <a href="/" className="text-sm text-slate-500 hover:text-slate-300 transition-colors">← Home</a>
          <h1 className="text-3xl font-bold text-slate-100 mt-3 mb-1">Gym</h1>
          <p className="text-slate-400">Select an exercise to begin.</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {GYM_EXERCISES.map((ex) => (
            <button
              key={ex.id}
              onClick={() => router.push(`/gym?exercise=${ex.id}`)}
              className="rounded-xl border border-surface-700 bg-surface-800 px-4 py-3
                         text-sm font-medium text-slate-300 text-left
                         hover:border-brand-500/60 hover:bg-surface-700/50 hover:text-slate-100
                         transition-all duration-150"
            >
              {ex.label}
            </button>
          ))}
        </div>
      </div>
    );
  }

  const sportLabel = sport === "basketball" ? "Basketball" : "Gym";
  const exerciseLabel =
    sport === "gym"
      ? GYM_EXERCISES.find((e) => e.id === exerciseId)?.label ?? exerciseId
      : "Jump Shot";

  return (
    <div className="max-w-5xl mx-auto px-6 py-8">
      {/* Breadcrumb */}
      <div className="mb-6 flex items-center gap-2 text-sm text-slate-500">
        <a href="/" className="hover:text-slate-300 transition-colors">Home</a>
        <span>/</span>
        {sport === "gym" && exerciseId ? (
          <>
            <a href="/gym" className="hover:text-slate-300 transition-colors">Gym</a>
            <span>/</span>
            <span className="text-slate-300">{exerciseLabel}</span>
          </>
        ) : (
          <span className="text-slate-300">{sportLabel}</span>
        )}
      </div>

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">{exerciseLabel}</h1>
          <p className="text-sm text-slate-500 mt-0.5">{sportLabel} analysis</p>
        </div>
        <div className="flex items-center gap-2">
          {(capturedBlob || canonicalResult) && (
            <button onClick={resetAll} className="px-4 py-2 rounded-lg text-xs text-slate-500 hover:text-slate-300 transition-colors border border-surface-700">
              Start over
            </button>
          )}
          {!capturedBlob && !canonicalResult && (
            <button
              onClick={() => { setCameraError(null); setCameraActive((v) => !v); }}
              className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-colors
                ${cameraActive
                  ? "bg-surface-700 text-slate-300 hover:bg-surface-600 border border-surface-600"
                  : "bg-brand-500 text-white hover:bg-brand-600"}`}
            >
              {cameraActive ? "Stop camera" : "Start camera"}
            </button>
          )}
        </div>
      </div>

      {cameraError && (
        <div className="mb-4 rounded-lg border border-rose-700/50 bg-rose-900/20 px-4 py-3 text-sm text-rose-300">
          {cameraError}
        </div>
      )}

      {/* Camera or captured-clip placeholder */}
      {!capturedBlob && !canonicalResult ? (
        <PoseCamera
          active={cameraActive}
          onLandmarks={handleLandmarks}
          onCaptureComplete={handleCaptureComplete}
          onError={setCameraError}
        />
      ) : !canonicalResult ? (
        <div className="w-full aspect-video rounded-2xl border border-emerald-700/40 bg-surface-800
                        flex items-center justify-center mb-2">
          <div className="text-center">
            <p className="text-4xl mb-2">🎬</p>
            <p className="text-sm text-emerald-300 font-medium">
              Clip ready — {(capturedBlob!.size / 1024).toFixed(0)} KB
            </p>
            <p className="text-xs text-slate-600 mt-1">
              Click &ldquo;Analyse clip&rdquo; below to run the canonical pipeline
            </p>
          </div>
        </div>
      ) : null}

      {/* Metrics panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        <GhostMetricsPanel
          repCount={repCount}
          currentPhase={currentPhase}
          currentSignal={currentSignal}
          lastRep={lastRep}
          active={cameraActive || !!canonicalResult}
        />
        <CanonicalReport
          result={canonicalResult}
          uploadState={{ status: uploadStatus, error: uploadError }}
          capturedBlob={capturedBlob}
          exerciseId={exerciseId}
          onUpload={handleUpload}
        />
      </div>
    </div>
  );
}

export default function SportPage() {
  return (
    <Suspense>
      <SportPageInner />
    </Suspense>
  );
}
