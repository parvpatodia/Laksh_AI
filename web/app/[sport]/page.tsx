"use client";

/**
 * /basketball   -> capture + analysis page for jump shot
 * /gym          -> exercise picker, then capture + analysis
 *
 * Both sports share PoseCamera, repCounter, and GhostMetricsPanel.
 */

import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

import {
  feedFrame,
  isRealtimeSupported,
  makeRepCounterState,
  toWireVector,
  type GhostRepMetrics,
  type Phase,
  type RepCounterState,
} from "@/lib/realtime/repCounter";
import {
  analyzeBasketballVideo,
  analyzeGymVideo,
  type AnalyzeResponse,
  type BasketballAnalyzeResponse,
  type GhostRepVector,
} from "@/lib/api";
import GhostMetricsPanel from "@/components/GhostMetricsPanel";
import CanonicalReport from "@/components/CanonicalReport";
import BasketballReport from "@/components/BasketballReport";
import TrustPanel from "@/components/TrustPanel";
import FormInsights from "@/components/FormInsights";

const PoseCamera = dynamic(() => import("@/components/PoseCamera"), {
  ssr: false,
  loading: () => (
    <div className="w-full aspect-[3/4] sm:aspect-[4/5] lg:aspect-[16/10] max-h-[80vh]
                    rounded-2xl border border-surface-700 bg-surface-900 flex items-center justify-center">
      <p className="text-sm text-slate-600">Loading camera...</p>
    </div>
  ),
});

// ---------------------------------------------------------------------------
// Exercise registry
// IDs must match app/gym/exercises_v0.py exactly.
// ---------------------------------------------------------------------------

const GYM_EXERCISES: {
  id: string;
  label: string;
  tip: string;
  dumbbell?: boolean;
  category: string;
}[] = [
  {
    id: "dumbbell_bicep_curl",
    label: "Bicep Curl",
    tip: "Side view, camera at chest height, full arm visible. Keep elbow pinned to your side throughout.",
    dumbbell: true,
    category: "Pull",
  },
  {
    id: "overhead_press",
    label: "Overhead Press",
    tip: "Side view, camera at chest height, head to hips in frame. Works with dumbbells.",
    dumbbell: true,
    category: "Push",
  },
  {
    id: "bench_press",
    label: "Bench Press",
    tip: "Side view at bar height, bench and full bar path visible. Works with dumbbells on the floor.",
    dumbbell: true,
    category: "Push",
  },
  {
    id: "romanian_deadlift",
    label: "Romanian Deadlift",
    tip: "Side view, camera at hip height, capture the full hinge angle. Works with dumbbells.",
    dumbbell: true,
    category: "Hinge",
  },
  {
    id: "push_up",
    label: "Push-up",
    tip: "Side view, camera at ground level, shoulder to ankle in frame. No equipment needed.",
    dumbbell: false,
    category: "Push",
  },
  {
    id: "back_squat",
    label: "Back Squat",
    tip: "Side view, camera at hip height, full body in frame including bar.",
    dumbbell: false,
    category: "Squat",
  },
  {
    id: "front_squat",
    label: "Front Squat",
    tip: "Side view, camera at hip height, keep the full bar path visible.",
    dumbbell: false,
    category: "Squat",
  },
  {
    id: "conventional_deadlift",
    label: "Deadlift",
    tip: "Side view, camera at knee to hip height, bar must be visible at lockout.",
    dumbbell: false,
    category: "Hinge",
  },
  {
    id: "barbell_row",
    label: "Barbell Row",
    tip: "Side view, camera at hip height, torso hinge angle and bar path in frame.",
    dumbbell: false,
    category: "Pull",
  },
  {
    id: "pull_up",
    label: "Pull-up",
    tip: "Front view, bar slightly above head, full torso visible from bar to hips.",
    dumbbell: false,
    category: "Pull",
  },
  {
    id: "walking_lunge",
    label: "Walking Lunge",
    tip: "Side view, camera at hip height. Counts every right-leg step.",
    dumbbell: false,
    category: "Lunge",
  },
];

const CATEGORIES = ["Pull", "Push", "Squat", "Hinge", "Lunge"];

type UploadStatus = "idle" | "uploading" | "done" | "error";

// ---------------------------------------------------------------------------
// Pre-flight quality constants (mirrors app/preflight/quality_gate.py)
// ---------------------------------------------------------------------------

const UPPER_BODY_EXERCISE_IDS = new Set([
  "dumbbell_bicep_curl",
  "bench_press",
  "overhead_press",
  "barbell_row",
  "push_up",
  "pull_up",
  "plank",
  "basketball",
  "jump_shot",
]);

const PREFLIGHT_CORE_UPPER = [11, 12, 13, 14, 15, 16] as const;
const PREFLIGHT_CORE_FULL  = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26] as const;
const PREFLIGHT_RING_SIZE  = 90;
const PREFLIGHT_VIS_MIN    = 0.25;
const PREFLIGHT_IFR_MIN    = 0.80;
const PREFLIGHT_MARGIN     = 0.05;

// ---------------------------------------------------------------------------
// Inner page component
// ---------------------------------------------------------------------------

function SportPageInner() {
  const params = useParams<{ sport: string }>();
  const searchParams = useSearchParams();
  const router = useRouter();

  const sport = params.sport as "basketball" | "gym";
  const exerciseId = searchParams.get("exercise") ?? (sport === "basketball" ? "basketball" : null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);
  const [capturedMime, setCapturedMime] = useState("video/webm");
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>("idle");
  const [uploadError, setUploadError] = useState<string | undefined>();
  const [gymResult, setGymResult] = useState<AnalyzeResponse | null>(null);
  const [bballResult, setBballResult] = useState<BasketballAnalyzeResponse | null>(null);
  const haveCanonicalResult = gymResult !== null || bballResult !== null;

  // Eagerly load pose model
  useEffect(() => {
    import("@/lib/pose/landmarkerLoader").then(({ loadPoseLandmarker }) => {
      loadPoseLandmarker().catch(() => {});
    });
  }, []);

  // Rep counter state
  const repStateRef = useRef<RepCounterState>(makeRepCounterState());
  const [repCount, setRepCount] = useState(0);
  const [currentPhase, setCurrentPhase] = useState<Phase>("rest");
  const [currentSignal, setCurrentSignal] = useState<number | null>(null);
  const [lastRep, setLastRep] = useState<GhostRepMetrics | null>(null);

  // Pre-flight quality ring buffer
  const preflightRingRef = useRef<Array<[number, number]>>([]);
  const [preflightOk, setPreflightOk] = useState<boolean | null>(null);
  const [preflightHint, setPreflightHint] = useState<string | null>(null);

  // IMPORTANT: do NOT add repCount to deps — it would recreate this callback
  // on every rep and tear down the RAF loop, causing missed reps.
  const handleLandmarks = useCallback(
    (landmarks: NormalizedLandmark[], ts: number) => {
      if (!exerciseId) return;
      const completed = feedFrame(repStateRef.current, landmarks, exerciseId, ts);
      const s = repStateRef.current;
      setCurrentSignal(s.currentSignal);
      setCurrentPhase(s.currentPhase);
      setRepCount(s.repCount);
      if (completed) setLastRep(completed);

      // Pre-flight ring buffer update
      const coreIdx = exerciseId && UPPER_BODY_EXERCISE_IDS.has(exerciseId)
        ? PREFLIGHT_CORE_UPPER
        : PREFLIGHT_CORE_FULL;
      const core = (coreIdx as readonly number[]).map((i) => landmarks[i]).filter(Boolean);
      if (core.length === 0) return;

      const meanVis = core.reduce((sum, lm) => sum + (lm.visibility ?? 0), 0) / core.length;
      const allInFrame = core.every(
        (lm) =>
          lm.x >= PREFLIGHT_MARGIN && lm.x <= 1 - PREFLIGHT_MARGIN &&
          lm.y >= PREFLIGHT_MARGIN && lm.y <= 1 - PREFLIGHT_MARGIN,
      );
      const ring = preflightRingRef.current;
      ring.push([meanVis, allInFrame ? 1 : 0]);
      if (ring.length > PREFLIGHT_RING_SIZE) ring.shift();

      if (ring.length % 15 === 0 || ring.length === PREFLIGHT_RING_SIZE) {
        const avgVis = ring.reduce((s, [v]) => s + v, 0) / ring.length;
        const ifr = ring.reduce((s, [, f]) => s + f, 0) / ring.length;
        const visOk = avgVis >= PREFLIGHT_VIS_MIN;
        const ifrOk = ifr >= PREFLIGHT_IFR_MIN;
        setPreflightOk(visOk && ifrOk);
        if (!visOk || !ifrOk) {
          const bodyLabel = exerciseId && UPPER_BODY_EXERCISE_IDS.has(exerciseId)
            ? "upper body" : "full body";
          if (!visOk && !ifrOk) {
            setPreflightHint(`Move into better light and make sure your ${bodyLabel} is fully in frame.`);
          } else if (!visOk) {
            setPreflightHint(`Pose confidence low (${(avgVis * 100).toFixed(0)}%). Try moving to a brighter spot.`);
          } else {
            setPreflightHint(`Only ${(ifr * 100).toFixed(0)}% of frames have your ${bodyLabel} fully visible. Adjust camera angle.`);
          }
        } else {
          setPreflightHint(null);
        }
      }
    },
    [exerciseId],
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
      if (sport === "basketball") {
        const result = await analyzeBasketballVideo(capturedBlob, null, capturedMime);
        setBballResult(result);
      } else {
        const ghostReps: GhostRepVector[] = repStateRef.current.completedReps.map(
          (r) => toWireVector(r, exerciseId) as GhostRepVector,
        );
        const result = await analyzeGymVideo(capturedBlob, exerciseId, capturedMime, ghostReps);
        setGymResult(result);
      }
      setUploadStatus("done");
    } catch (err: unknown) {
      setUploadStatus("error");
      setUploadError(err instanceof Error ? err.message : String(err));
    }
  }, [capturedBlob, capturedMime, exerciseId, sport]);

  const resetAll = useCallback(() => {
    setCapturedBlob(null);
    setGymResult(null);
    setBballResult(null);
    setUploadStatus("idle");
    setUploadError(undefined);
    setCameraError(null);
    repStateRef.current = makeRepCounterState();
    setRepCount(0);
    setCurrentPhase("rest");
    setCurrentSignal(null);
    setLastRep(null);
    setCameraActive(false);
    preflightRingRef.current = [];
    setPreflightOk(null);
    setPreflightHint(null);
  }, []);

  const exerciseMeta = useMemo(
    () => sport === "gym" ? GYM_EXERCISES.find((e) => e.id === exerciseId) ?? null : null,
    [sport, exerciseId],
  );

  // Unknown sport guard
  if (sport !== "basketball" && sport !== "gym") {
    return (
      <div className="max-w-2xl mx-auto px-6 py-24 text-center">
        <p className="text-slate-400">Unknown sport: {sport}</p>
        <a href="/" className="mt-4 inline-block text-brand-500 hover:underline">Back to home</a>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Gym exercise picker
  // ---------------------------------------------------------------------------
  if (sport === "gym" && !exerciseId) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-10">
        <div className="mb-8">
          <a href="/" className="inline-flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300 transition-colors mb-6">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
            Home
          </a>
          <h1 className="text-2xl font-bold text-slate-100 mt-2 mb-1">Choose an exercise</h1>
          <p className="text-sm text-slate-500">
            12 compound movements. Select one to begin live tracking and analysis.
          </p>
        </div>

        {CATEGORIES.map((cat) => {
          const exercises = GYM_EXERCISES.filter((e) => e.category === cat);
          return (
            <div key={cat} className="mb-6">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                {cat}
              </p>
              <div className="space-y-1.5">
                {exercises.map((ex) => (
                  <button
                    key={ex.id}
                    onClick={() => router.push(`/gym?exercise=${ex.id}`)}
                    className="w-full rounded-lg border border-surface-700 bg-surface-800 px-4 py-3
                               flex items-center justify-between gap-4
                               hover:border-brand-500/50 hover:bg-surface-700/40
                               transition-all duration-150 group text-left"
                  >
                    <div>
                      <p className="text-sm font-medium text-slate-200 group-hover:text-slate-100 mb-0.5">
                        {ex.label}
                      </p>
                      <p className="text-xs text-slate-500">{ex.tip}</p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {ex.dumbbell && (
                        <span className="text-[10px] font-semibold text-brand-400 border border-brand-500/30
                                         bg-brand-500/10 rounded px-1.5 py-0.5">
                          DB
                        </span>
                      )}
                      <svg className="w-4 h-4 text-slate-600 group-hover:text-brand-500 transition-colors"
                           fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                      </svg>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          );
        })}

        <p className="text-xs text-slate-600 mt-2">
          DB = works with a single pair of dumbbells, no rack required.
        </p>
      </div>
    );
  }

  if (sport === "gym" && exerciseId && !isRealtimeSupported(exerciseId)) {
    return (
      <div className="max-w-2xl mx-auto px-6 py-24 text-center">
        <p className="text-slate-400 mb-3">
          &ldquo;{exerciseId}&rdquo; is not supported for live tracking.
        </p>
        <a href="/gym" className="inline-block text-brand-500 text-sm hover:underline">
          Back to exercise list
        </a>
      </div>
    );
  }

  const sportLabel = sport === "basketball" ? "Basketball" : "Gym";
  const exerciseLabel = sport === "gym"
    ? exerciseMeta?.label ?? exerciseId ?? "Exercise"
    : "Jump Shot";

  // ---------------------------------------------------------------------------
  // Main capture + analysis page
  // ---------------------------------------------------------------------------
  return (
    <div className="max-w-7xl mx-auto px-6 py-8">

      {/* Breadcrumb */}
      <nav className="mb-5 flex items-center gap-2 text-xs text-slate-500">
        <a href="/" className="hover:text-slate-300 transition-colors">Home</a>
        <span className="text-slate-700">/</span>
        {sport === "gym" && exerciseId ? (
          <>
            <a href="/gym" className="hover:text-slate-300 transition-colors">Gym</a>
            <span className="text-slate-700">/</span>
            <span className="text-slate-300">{exerciseLabel}</span>
          </>
        ) : (
          <span className="text-slate-300">{sportLabel}</span>
        )}
      </nav>

      {/* Header */}
      <div className="flex items-start justify-between mb-5">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">{exerciseLabel}</h1>
          <p className="text-xs text-slate-500 mt-0.5 uppercase tracking-wide">
            {sportLabel} analysis
          </p>
        </div>
        <div className="flex items-center gap-2">
          {(capturedBlob || haveCanonicalResult) && (
            <button
              onClick={resetAll}
              className="px-3.5 py-2 rounded-lg text-xs text-slate-400 hover:text-slate-200
                         border border-surface-700 hover:border-surface-600 transition-all"
            >
              Start over
            </button>
          )}
          {!capturedBlob && !haveCanonicalResult && (
            <button
              onClick={() => { setCameraError(null); setCameraActive((v) => !v); }}
              className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-all
                ${cameraActive
                  ? "bg-surface-700 text-slate-300 hover:bg-surface-600 border border-surface-600"
                  : "bg-brand-500 text-white hover:bg-brand-600 shadow-lg shadow-brand-500/20"}`}
            >
              {cameraActive ? "Stop camera" : "Start camera"}
            </button>
          )}
        </div>
      </div>

      {/* Camera setup hint */}
      {!capturedBlob && !haveCanonicalResult && (
        <div className="mb-4 rounded-lg border-l-2 border-brand-500/60 bg-surface-800/60 pl-4 pr-4 py-3 flex items-start gap-3">
          <div className="min-w-0">
            <span className="text-xs font-semibold text-slate-300">Camera setup: </span>
            <span className="text-xs text-slate-400">
              {sport === "gym"
                ? exerciseMeta?.tip ?? "Side view, full body in frame, about 2 m from camera."
                : "Side view preferred. Keep arms and shoulders in frame for the full shooting motion."}
            </span>
            <span className="text-xs text-slate-600 ml-2">
              The Pose % badge shows tracking quality in real time.
            </span>
          </div>
        </div>
      )}

      {/* Camera error */}
      {cameraError && (
        <div className="mb-4 rounded-lg border border-rose-700/50 bg-rose-900/15 px-4 py-3
                        flex items-center gap-2 text-sm text-rose-300">
          <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          </svg>
          {cameraError}
        </div>
      )}

      {/* Pre-flight quality warning */}
      {cameraActive && preflightOk === false && preflightHint && (
        <div className="mb-4 rounded-lg border border-amber-600/40 bg-amber-900/10 px-4 py-3
                        flex items-start gap-3">
          <svg className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          </svg>
          <div>
            <p className="text-sm font-medium text-amber-200">Tracking quality low</p>
            <p className="text-xs text-amber-300/70 mt-0.5">{preflightHint}</p>
          </div>
        </div>
      )}

      {/* Camera viewport or captured clip placeholder */}
      {!capturedBlob && !haveCanonicalResult ? (
        <PoseCamera
          active={cameraActive}
          onLandmarks={handleLandmarks}
          onCaptureComplete={handleCaptureComplete}
          onError={setCameraError}
          maxDurationS={15}
        />
      ) : !haveCanonicalResult ? (
        <div className="w-full aspect-[3/4] sm:aspect-[4/5] lg:aspect-[16/10] max-h-[80vh]
                        rounded-2xl border border-emerald-700/30 bg-surface-800
                        flex flex-col items-center justify-center gap-3 mb-2">
          <div className="rounded-full border border-emerald-700/50 bg-emerald-900/20 p-4">
            <svg className="w-6 h-6 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round"
                    d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z" />
            </svg>
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-emerald-300">
              Clip recorded ({(capturedBlob!.size / 1024).toFixed(0)} KB)
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Click Analyse below to run the biomechanical pipeline.
            </p>
          </div>
        </div>
      ) : null}

      {/* Trust panel (gym only) */}
      {gymResult && <TrustPanel result={gymResult} />}

      {/* Live metrics + canonical report */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        <GhostMetricsPanel
          repCount={repCount}
          currentPhase={currentPhase}
          currentSignal={currentSignal}
          lastRep={lastRep}
          active={cameraActive || haveCanonicalResult}
          unitLabel={sport === "basketball" ? "shot" : "rep"}
        />
        {sport === "gym" ? (
          <CanonicalReport
            result={gymResult}
            uploadState={{ status: uploadStatus, error: uploadError }}
            capturedBlob={capturedBlob}
            exerciseId={exerciseId}
            onUpload={handleUpload}
          />
        ) : (
          <BasketballReport
            result={bballResult}
            uploadState={{ status: uploadStatus, error: uploadError }}
            capturedBlob={capturedBlob}
            onUpload={handleUpload}
          />
        )}
      </div>

      {/* Form insights (gym only) */}
      {gymResult && (
        <div className="mt-6">
          <FormInsights result={gymResult} />
        </div>
      )}
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
