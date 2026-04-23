"use client";

/**
 * /basketball   -> capture page for jump shot (legacy /analyze-video)
 * /gym          -> exercise picker, then capture page (v1 /analyze/gym/video)
 *
 * Both sports share:
 *   - PoseCamera   (MediaPipe LIVE_STREAM)
 *   - repCounter   (browser ghost counter w/ quality gates)
 *   - GhostMetricsPanel
 *
 * Where they differ:
 *   - Backend endpoint + response schema
 *   - CanonicalReport vs BasketballReport renderer
 *   - FormInsights (rule-based) for gym; basketball ships its own AI
 *     scout block as part of the canonical response.
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
      <p className="text-sm text-slate-600">Loading camera…</p>
    </div>
  ),
});

// ---------------------------------------------------------------------------
// Exercise registry (frontend mirror of app/gym/exercises_v0.py)
// ---------------------------------------------------------------------------
//
// IDs MUST match the backend exactly (verified 2026-04-19).  Each entry
// also carries a one-line camera tip surfaced as a contextual hint card,
// derived from the backend's `camera_instruction` field but trimmed for
// the live UI context.
//
// hip_thrust is INTENTIONALLY OMITTED — backend exercises_v0 does not
// register it, so picking it would 400 with UnknownExerciseError on
// upload.  Same for plank/farmer_carry which need different rep models.
//
// `dumbbell` flag → exercise works with a single dumbbell or a pair (also
// works barbell/bodyweight where applicable). Exposed in the UI as a chip
// so judges with only DBs can quickly pick a demo movement.

const GYM_EXERCISES: { id: string; label: string; tip: string; dumbbell?: boolean }[] = [
  { id: "dumbbell_bicep_curl", label: "Dumbbell Bicep Curl", tip: "Side view, chest-height, full arm visible; keep elbow pinned to torso", dumbbell: true },
  { id: "overhead_press", label: "Overhead Press", tip: "Side view, chest-height, head-to-hips visible. Works with dumbbells.", dumbbell: true },
  { id: "bench_press", label: "Bench Press", tip: "Side view, bar-height, bench + bar end-to-end. Works with dumbbells (flat bench or floor press).", dumbbell: true },
  { id: "romanian_deadlift", label: "Romanian Deadlift", tip: "Side view, hip-height, capture the hinge angle. Works with dumbbells.", dumbbell: true },
  { id: "push_up", label: "Push-up", tip: "Side view, ground-level, shoulder-to-ankle in frame. No equipment." },
  { id: "back_squat", label: "Back Squat", tip: "Side view, hip-height camera, full body in frame" },
  { id: "front_squat", label: "Front Squat", tip: "Side view, hip-height, keep bar path in frame" },
  { id: "conventional_deadlift", label: "Deadlift", tip: "Side view, knee-to-hip height, bar visible at lockout" },
  { id: "barbell_row", label: "Barbell Row", tip: "Side view, hip-height, torso hinge + bar in frame" },
  { id: "pull_up", label: "Pull-up", tip: "Front view, bar slightly above head, full torso visible" },
  { id: "walking_lunge", label: "Walking Lunge", tip: "Side view, hip-height; counts every right-leg step" },
];

type UploadStatus = "idle" | "uploading" | "done" | "error";

// ---------------------------------------------------------------------------
// A4: Pre-flight quality constants (module-level to avoid re-creation on
// every render). Mirror of evaluation/preflight_thresholds.json and
// app/preflight/quality_gate.py. Both sides must agree exactly.
// ---------------------------------------------------------------------------
const PREFLIGHT_CORE_IDX = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26] as const;
const PREFLIGHT_RING_SIZE = 90;  // ~3 s at 30 fps
const PREFLIGHT_VIS_MIN   = 0.50; // MediaPipe "confident" band lower bound
const PREFLIGHT_IFR_MIN   = 0.80; // 80% of frames must have full body in frame
const PREFLIGHT_MARGIN    = 0.05; // 5% border exclusion zone (matches quality_gate.py)

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

  // Upload + canonical results (one of two — never both at once for a session)
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>("idle");
  const [uploadError, setUploadError] = useState<string | undefined>();
  const [gymResult, setGymResult] = useState<AnalyzeResponse | null>(null);
  const [bballResult, setBballResult] = useState<BasketballAnalyzeResponse | null>(null);
  const haveCanonicalResult = gymResult !== null || bballResult !== null;

  // Eagerly load the pose model so it's ready by the time the user clicks.
  useEffect(() => {
    import("@/lib/pose/landmarkerLoader").then(({ loadPoseLandmarker }) => {
      loadPoseLandmarker().catch(() => {
        // Silently ignore -- PoseCamera surfaces this error itself.
      });
    });
  }, []);

  // Rep counter (ref so RAF mutations don't trigger re-renders)
  const repStateRef = useRef<RepCounterState>(makeRepCounterState());
  const [repCount, setRepCount] = useState(0);
  const [currentPhase, setCurrentPhase] = useState<Phase>("rest");
  const [currentSignal, setCurrentSignal] = useState<number | null>(null);
  const [lastRep, setLastRep] = useState<GhostRepMetrics | null>(null);

  // ---- A4: Client-side pre-flight quality ring buffer --------------------
  // Ring buffer stored in a ref so mutations don't trigger re-renders.
  // Each slot: [mean_core_visibility, is_all_in_frame (0|1)].
  const preflightRingRef = useRef<Array<[number, number]>>([]);
  // Expose quality state to upload gate (true = OK to upload).
  const [preflightOk, setPreflightOk] = useState<boolean | null>(null);
  const [preflightHint, setPreflightHint] = useState<string | null>(null);

  // IMPORTANT: do NOT include `repCount` in the deps. Including it would
  // recreate this callback on every rep, which propagates into PoseCamera's
  // detection-loop useEffect (onLandmarks is in its deps), tearing down and
  // recreating the requestAnimationFrame loop.  That caused brief frame gaps
  // at every rep boundary -- and missed reps.
  const handleLandmarks = useCallback(
    (landmarks: NormalizedLandmark[], ts: number) => {
      if (!exerciseId) return;
      const completed = feedFrame(repStateRef.current, landmarks, exerciseId, ts);
      const s = repStateRef.current;
      setCurrentSignal(s.currentSignal);
      setCurrentPhase(s.currentPhase);
      setRepCount(s.repCount);
      if (completed) setLastRep(completed);

      // ---- A4: update ring buffer and re-evaluate quality ----
      const core = PREFLIGHT_CORE_IDX.map((i) => landmarks[i]).filter(Boolean);
      if (core.length === 0) return; // no landmarks this frame

      const meanVis = core.reduce((sum, lm) => sum + (lm.visibility ?? 0), 0) / core.length;
      const allInFrame = core.every(
        (lm) =>
          lm.x >= PREFLIGHT_MARGIN && lm.x <= 1 - PREFLIGHT_MARGIN &&
          lm.y >= PREFLIGHT_MARGIN && lm.y <= 1 - PREFLIGHT_MARGIN,
      );
      const ring = preflightRingRef.current;
      ring.push([meanVis, allInFrame ? 1 : 0]);
      if (ring.length > PREFLIGHT_RING_SIZE) ring.shift();

      // Only re-evaluate (and potentially setState) every 15 frames to
      // avoid flooding the React scheduler. The boolean flip is the
      // expensive part; the ring mutation above is free.
      if (ring.length % 15 === 0 || ring.length === PREFLIGHT_RING_SIZE) {
        const avgVis = ring.reduce((s, [v]) => s + v, 0) / ring.length;
        const ifr = ring.reduce((s, [, f]) => s + f, 0) / ring.length;
        const visOk = avgVis >= PREFLIGHT_VIS_MIN;
        const ifrOk = ifr >= PREFLIGHT_IFR_MIN;
        const ok = visOk && ifrOk;
        setPreflightOk(ok);
        if (!ok) {
          if (!visOk && !ifrOk) {
            setPreflightHint("Move into better light and step back so your full body is in frame.");
          } else if (!visOk) {
            setPreflightHint(`Pose confidence low (${(avgVis * 100).toFixed(0)}% vs 50% needed). Try better lighting.`);
          } else {
            setPreflightHint(`Only ${(ifr * 100).toFixed(0)}% of frames have your full body in frame (need 80%). Step back.`);
          }
        } else {
          setPreflightHint(null);
        }
      }
    },
    [exerciseId], // PREFLIGHT_* are module-level constants — not in deps
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
        // Build the ghost-rep wire format via toWireVector — single source of
        // truth that handles BOTH the field-name rename
        // (min_visibility → primary_joints_min_visibility) and the
        // signal_amplitude unit conversion (×180 for cyclic_angle exercises
        // so the parity probe compares degrees-with-degrees).  See
        // app/parity/realtime.py probe_reps and the comment block in
        // web/lib/realtime/repCounter.ts toWireVector for why.
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
    // A4: clear stale preflight quality state so the amber hint from a prior
    // session does not bleed into the next one. The ring is a ref (not state)
    // so we mutate it directly; setPreflightOk/Hint trigger one re-render.
    preflightRingRef.current = [];
    setPreflightOk(null);
    setPreflightHint(null);
  }, []);

  // Camera tip for the contextual hint card (gym only — basketball
  // gets its own static instruction below).
  const exerciseMeta = useMemo(
    () => sport === "gym" ? GYM_EXERCISES.find((e) => e.id === exerciseId) ?? null : null,
    [sport, exerciseId],
  );

  // ---- Unknown sport ---------------------------------------------------
  if (sport !== "basketball" && sport !== "gym") {
    return (
      <div className="max-w-2xl mx-auto px-6 py-24 text-center">
        <p className="text-slate-400">Unknown sport: {sport}</p>
        <a href="/" className="mt-4 inline-block text-brand-500 hover:underline">Back to home</a>
      </div>
    );
  }

  // ---- Gym exercise picker --------------------------------------------
  if (sport === "gym" && !exerciseId) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-12">
        <div className="mb-8">
          <a href="/" className="text-sm text-slate-500 hover:text-slate-300 transition-colors">← Home</a>
          <h1 className="text-3xl font-bold text-slate-100 mt-3 mb-1">Gym</h1>
          <p className="text-slate-400">Select an exercise to begin.</p>
        </div>
        <div className="mb-3 flex items-center gap-2 text-xs text-slate-500">
          <span className="rounded-md bg-brand-500/15 px-1.5 py-0.5 text-brand-400 font-medium">DB</span>
          <span>= works with a single pair of dumbbells (no rack required)</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {GYM_EXERCISES.map((ex) => (
            <button
              key={ex.id}
              onClick={() => router.push(`/gym?exercise=${ex.id}`)}
              className="rounded-xl border border-surface-700 bg-surface-800 px-4 py-3
                         text-sm font-medium text-slate-300 text-left
                         hover:border-brand-500/60 hover:bg-surface-700/50 hover:text-slate-100
                         transition-all duration-150 flex items-center justify-between gap-2"
            >
              <span>{ex.label}</span>
              {ex.dumbbell && (
                <span className="rounded-md bg-brand-500/15 px-1.5 py-0.5 text-[10px] text-brand-400 font-semibold">
                  DB
                </span>
              )}
            </button>
          ))}
        </div>
        <p className="text-xs text-slate-600 mt-6 leading-relaxed">
          Twelve movements covering squat / hinge / push / pull / lunge / curl.
          Each exercise streams to MediaPipe BlazePose at ~30 fps for live
          rep counting, then runs the canonical biomech pipeline server-side
          when you stop the recording.
        </p>
      </div>
    );
  }

  const sportLabel = sport === "basketball" ? "Basketball" : "Gym";
  const exerciseLabel =
    sport === "gym" ? exerciseMeta?.label ?? exerciseId ?? "—"
      : "Jump Shot";

  // Unsupported gym exercise (e.g., user hand-typed ?exercise=xxx)
  if (sport === "gym" && exerciseId && !isRealtimeSupported(exerciseId)) {
    return (
      <div className="max-w-2xl mx-auto px-6 py-24 text-center">
        <p className="text-slate-400 mb-2">
          Exercise <code className="font-mono text-slate-300">{exerciseId}</code> is not supported by the realtime ghost counter.
        </p>
        <a href="/gym" className="inline-block text-brand-500 hover:underline">
          Back to gym picker
        </a>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
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
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">{exerciseLabel}</h1>
          <p className="text-sm text-slate-500 mt-0.5">{sportLabel} analysis</p>
        </div>
        <div className="flex items-center gap-2">
          {(capturedBlob || haveCanonicalResult) && (
            <button onClick={resetAll} className="px-4 py-2 rounded-lg text-xs text-slate-500 hover:text-slate-300 transition-colors border border-surface-700">
              Start over
            </button>
          )}
          {!capturedBlob && !haveCanonicalResult && (
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

      {/* Camera framing hint (single source of truth: GYM_EXERCISES.tip
          for gym; static line for basketball) */}
      {!capturedBlob && !haveCanonicalResult && (
        <div className="mb-4 rounded-lg border border-surface-700 bg-surface-800/60 px-4 py-3
                        flex items-start gap-3 text-sm">
          <span aria-hidden className="text-brand-400 mt-0.5">◉</span>
          <div className="text-slate-300">
            <span className="font-medium text-slate-200">Setup:</span>{" "}
            {sport === "gym"
              ? exerciseMeta?.tip ?? "Side view, full body in frame, ~2 m from camera"
              : "Arms + shoulders visible (chest-up / selfie angle is fine). Side view is optional. Rep = one release cycle."}
            <span className="text-xs text-slate-500 ml-2">
              · Counter waits ~0.5 s for you to settle before counting.
            </span>
          </div>
        </div>
      )}

      {cameraError && (
        <div className="mb-4 rounded-lg border border-rose-700/50 bg-rose-900/20 px-4 py-3 text-sm text-rose-300">
          {cameraError}
        </div>
      )}

      {/* A4: Pre-flight quality hint — shown while camera is active and quality is poor.
          Disappears automatically once quality improves (ring buffer self-heals).
          Uses an amber warning (not red error) — the user can still record; we just tell
          them the upload will likely fail so they can reposition first. */}
      {cameraActive && preflightOk === false && preflightHint && (
        <div className="mb-4 rounded-lg border border-amber-600/50 bg-amber-900/15 px-4 py-3 flex items-start gap-3">
          <span className="text-amber-400 text-base mt-0.5" aria-hidden>⚠</span>
          <div>
            <p className="text-sm font-medium text-amber-200">Pose quality low — recording may not analyse reliably</p>
            <p className="text-xs text-amber-300/80 leading-relaxed mt-0.5">{preflightHint}</p>
          </div>
        </div>
      )}

      {/* Camera or captured-clip placeholder */}
      {!capturedBlob && !haveCanonicalResult ? (
        <PoseCamera
          active={cameraActive}
          onLandmarks={handleLandmarks}
          onCaptureComplete={handleCaptureComplete}
          onError={setCameraError}
        />
      ) : !haveCanonicalResult ? (
        <div className="w-full aspect-[3/4] sm:aspect-[4/5] lg:aspect-[16/10] max-h-[80vh]
                        rounded-2xl border border-emerald-700/40 bg-surface-800
                        flex items-center justify-center mb-2">
          <div className="text-center">
            <p className="text-4xl mb-2">🎬</p>
            <p className="text-sm text-emerald-300 font-medium">
              Clip ready — {(capturedBlob!.size / 1024).toFixed(0)} KB
            </p>
            <p className="text-xs text-slate-600 mt-1">
              Click &ldquo;Analyse&rdquo; to run the canonical pipeline
            </p>
          </div>
        </div>
      ) : null}

      {/* Trust panel — gym only (it consumes the v1 envelope). */}
      {gymResult && <TrustPanel result={gymResult} />}

      {/* Metrics + canonical report */}
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

      {/* Form insights — gym only.  Basketball ships its AI scout block
          inside BasketballReport (legacy /analyze-video already returns
          3 athlete_feedback bullets). */}
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
