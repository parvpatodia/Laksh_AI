"use client";

/**
 * /basketball  -> sport capture page for basketball jump shot
 * /gym         -> exercise picker, then capture page
 *
 * This shell is wired up in Day 5 (PoseCamera) and Day 7 (canonical upload).
 * For now it renders the exercise selector for gym or a capture stub for basketball.
 */

import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";

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

function SportPageInner() {
  const params = useParams<{ sport: string }>();
  const searchParams = useSearchParams();
  const router = useRouter();

  const sport = params.sport;
  const exerciseId = searchParams.get("exercise");

  // Unknown sport: 404-ish fallback.
  if (sport !== "basketball" && sport !== "gym") {
    return (
      <div className="max-w-2xl mx-auto px-6 py-24 text-center">
        <p className="text-slate-400">Unknown sport: {sport}</p>
        <a href="/" className="mt-4 inline-block text-brand-500 hover:underline">
          Back to home
        </a>
      </div>
    );
  }

  // Gym with no exercise selected: show exercise picker.
  if (sport === "gym" && !exerciseId) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-12">
        <div className="mb-8">
          <a href="/" className="text-sm text-slate-500 hover:text-slate-300 transition-colors">
            ← Home
          </a>
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

  // Capture view: basketball or gym+exercise.
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

      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">{exerciseLabel}</h1>
          <p className="text-sm text-slate-500 mt-0.5">{sportLabel} analysis</p>
        </div>
      </div>

      {/* Camera + overlay area (wired in Day 5) */}
      <div className="relative w-full aspect-video rounded-2xl border border-surface-700
                      bg-surface-900 flex items-center justify-center mb-6">
        <div className="text-center text-slate-600">
          <p className="text-4xl mb-3">📷</p>
          <p className="text-sm">Camera feed loads here (Day 5)</p>
          <p className="text-xs text-slate-700 mt-1">
            @mediapipe/tasks-vision LIVE_STREAM
          </p>
        </div>
      </div>

      {/* Control bar (wired in Day 5-7) */}
      <div className="flex items-center gap-3 mb-8">
        <button
          disabled
          className="px-5 py-2.5 rounded-lg bg-brand-500/30 text-brand-300 text-sm font-medium
                     border border-brand-500/40 cursor-not-allowed opacity-60"
        >
          Start (coming Day 5)
        </button>
        <span className="text-xs text-slate-600">
          Grant camera permission, then click Start to begin pose tracking
        </span>
      </div>

      {/* Results area (wired in Day 7-8) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl border border-surface-700 bg-surface-800 p-5">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
            Ghost metrics{" "}
            <span className="chip-preview text-xs px-1.5 py-0.5 rounded font-normal ml-1">
              realtime_preview
            </span>
          </h2>
          <p className="text-xs text-slate-600">Appears during live pose tracking.</p>
        </div>
        <div className="rounded-xl border border-surface-700 bg-surface-800 p-5">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
            Canonical result{" "}
            <span className="chip-valid text-xs px-1.5 py-0.5 rounded font-normal ml-1">
              canonical_backend
            </span>
          </h2>
          <p className="text-xs text-slate-600">Appears after clip upload + analysis.</p>
        </div>
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
