"use client";

/**
 * /basketball  -> jump shot capture + analysis page
 * /gym         -> exercise picker, then capture + analysis
 *
 * TV-compatible: max-w-screen-2xl container, xl/2xl responsive grid.
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
    <div className="w-full aspect-video max-h-[75vh] rounded-2xl border border-surface-700
                    bg-surface-850 flex items-center justify-center">
      <p className="text-sm text-slate-600">Loading camera...</p>
    </div>
  ),
});

// ---------------------------------------------------------------------------
// Exercise registry -- IDs must match app/gym/exercises_v0.py exactly.
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
      <div className="max-w-screen-2xl mx-auto px-6 xl:px-16 py-10">

        {/* Back link */}
        <a href="/"
           className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500
                      hover:text-slate-300 transition-colors mb-8">
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
          Home
        </a>

        <div className="mb-10">
          <h1 className="text-3xl xl:text-4xl font-black text-white mb-2">Choose an exercise</h1>
          <p className="text-slate-500 text-sm">
            12 compound movements. Select one to begin live tracking and analysis.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3">
          {CATEGORIES.map((cat) => {
            const exercises = GYM_EXERCISES.filter((e) => e.category === cat);
            return (
              <div key={cat}>
                <p className="label-section mb-3">{cat}</p>
                <div className="space-y-2">
                  {exercises.map((ex) => (
                    <button
                      key={ex.id}
                      onClick={() => router.push(`/gym?exercise=${ex.id}`)}
                      className="w-full rounded-xl border border-surface-700 bg-surface-800 px-5 py-4
                                 flex items-center justify-between gap-4
                                 hover:border-brand-500/50 hover:bg-surface-750
                                 transition-all duration-150 group text-left"
                    >
                      <div>
                        <p className="text-sm font-semibold text-slate-200 group-hover:text-white mb-0.5">
                          {ex.label}
                        </p>
                        <p className="text-xs text-slate-500 leading-snug">{ex.tip}</p>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        {ex.dumbbell && (
                          <span className="text-[9px] font-bold uppercase tracking-wider text-brand-400
                                           border border-brand-500/40 bg-brand-500/10 rounded px-1.5 py-0.5">
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
        </div>

        <p className="text-xs text-slate-600 mt-6">
          DB = works with a single pair of dumbbells, no rack required.
        </p>
      </div>
    );
  }

  if (sport === "gym" && exerciseId && !isRealtimeSupported(exerciseId)) {
    return (
      <div className="max-w-2xl mx-auto px-6 py-24 text-center">
        <p className="text-slate-400 mb-3">
          &ldquo;{exerciseId}&rdquo; is not yet supported for live tracking.
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
    <div className="max-w-screen-2xl mx-auto px-6 xl:px-16 py-8 xl:py-10">

      {/* Breadcrumb */}
      <nav className="mb-6 flex items-center gap-2 text-xs text-slate-600">
        <a href="/" className="hover:text-slate-300 transition-colors font-medium">Home</a>
        <span className="text-surface-600">/</span>
        {sport === "gym" && exerciseId ? (
          <>
            <a href="/gym" className="hover:text-slate-300 transition-colors">Gym</a>
            <span className="text-surface-600">/</span>
            <span className="text-slate-400 font-medium">{exerciseLabel}</span>
          </>
        ) : (
          <span className="text-slate-400 font-medium">{sportLabel}</span>
        )}
      </nav>

      {/* Page header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl xl:text-3xl font-black text-white leading-tight">{exerciseLabel}</h1>
          <p className="label-section mt-1">{sportLabel} analysis</p>
        </div>
        <div className="flex items-center gap-2.5">
          {(capturedBlob || haveCanonicalResult) && (
            <button
              onClick={resetAll}
              className="px-4 py-2 rounded-lg text-xs font-medium text-slate-400 hover:text-white
                         border border-surface-700 hover:border-surface-600 transition-all"
            >
              Start over
            </button>
          )}
          {!capturedBlob && !haveCanonicalResult && (
            <button
              onClick={() => { setCameraError(null); setCameraActive((v) => !v); }}
              className={`px-5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200
                ${cameraActive
                  ? "bg-surface-700 text-slate-300 hover:bg-surface-600 border border-surface-600"
                  : "bg-brand-500 text-white hover:bg-brand-400 shadow-lg shadow-brand-500/25"}`}
            >
              {cameraActive ? "Stop camera" : "Start camera"}
            </button>
          )}
        </div>
      </div>

      {/* Camera setup hint */}
      {!capturedBlob && !haveCanonicalResult && (
        <div className="mb-5 rounded-xl border-l-2 border-brand-500/60 bg-surface-800/70
                        pl-4 pr-5 py-3.5 flex items-start gap-3">
          <svg className="w-4 h-4 text-brand-500/70 shrink-0 mt-0.5" fill="none"
               viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round"
                  d="M15 10l4.55-2.55A1 1 0 0121 8.39V15.6a1 1 0 01-1.45.89L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
          </svg>
          <div>
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
        <div className="mb-5 rounded-xl border border-rose-700/50 bg-rose-900/15 px-4 py-3
                        flex items-center gap-2.5 text-sm text-rose-300">
          <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round"
                  d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          </svg>
          {cameraError}
        </div>
      )}

      {/* Pre-flight quality warning */}
      {cameraActive && preflightOk === false && preflightHint && (
        <div className="mb-5 rounded-xl border border-amber-600/40 bg-amber-900/10 px-4 py-3
                        flex items-start gap-3">
          <svg className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" fill="none"
               viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round"
                  d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          </svg>
          <div>
            <p className="text-sm font-semibold text-amber-200">Tracking quality low</p>
            <p className="text-xs text-amber-300/70 mt-0.5">{preflightHint}</p>
          </div>
        </div>
      )}

      {/* Camera viewport or clip placeholder */}
      {!capturedBlob && !haveCanonicalResult ? (
        <PoseCamera
          active={cameraActive}
          onLandmarks={handleLandmarks}
          onCaptureComplete={handleCaptureComplete}
          onError={setCameraError}
          maxDurationS={15}
        />
      ) : !haveCanonicalResult ? (
        <div className="w-full aspect-video max-h-[60vh] rounded-2xl border border-emerald-700/30
                        bg-surface-800 flex flex-col items-center justify-center gap-4 mb-2">
          <div className="rounded-full border border-emerald-700/50 bg-emerald-900/20 p-5">
            <svg className="w-7 h-7 text-emerald-400" fill="none" viewBox="0 0 24 24"
                 stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round"
                    d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z" />
            </svg>
          </div>
          <div className="text-center">
            <p className="text-base font-bold text-emerald-300">
              Clip captured -- {(capturedBlob!.size / 1024).toFixed(0)} KB
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Click &ldquo;Run full analysis&rdquo; in the panel below to start the biomechanics pipeline.
            </p>
          </div>
        </div>
      ) : null}

      {/* Trust panel (gym only) */}
      {gymResult && <TrustPanel result={gymResult} />}

      {/* Live metrics + canonical report -- side by side at lg+ */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-[420px_1fr] gap-5 xl:gap-6 mt-6">
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
