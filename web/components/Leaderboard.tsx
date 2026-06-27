"use client";

/**
 * Leaderboard: ranks sessions by form_index per exercise.
 *
 * Honesty contract (extended to the leaderboard): form_index is a transparent,
 * uncalibrated RELATIVE index built only from measured quantities. It is never
 * presented as a validated form grade, and the backend disclaimer is shown
 * verbatim. Each row also exposes the code version (git SHA) that produced it.
 */

import { useCallback, useEffect, useState } from "react";

import {
  fetchLeaderboard,
  type FieldStatus,
  type LeaderboardResponse,
} from "@/lib/api";

const EXERCISE_OPTIONS: { id: string; label: string }[] = [
  { id: "", label: "All exercises" },
  { id: "back_squat", label: "Back Squat" },
  { id: "dumbbell_bicep_curl", label: "Bicep Curl" },
  { id: "overhead_press", label: "Overhead Press" },
  { id: "bench_press", label: "Bench Press" },
  { id: "deadlift", label: "Deadlift" },
];

function prettyExercise(id: string): string {
  const found = EXERCISE_OPTIONS.find((o) => o.id === id);
  if (found && found.id) return found.label;
  return id.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function statusChip(status: FieldStatus): string {
  if (status === "valid") return "chip-valid";
  if (status === "degraded") return "chip-degraded";
  return "chip-unknown";
}

/** Plain-language label for the measurement status (no jargon for general users). */
function statusLabel(status: FieldStatus): string {
  if (status === "valid") return "Clean";
  if (status === "degraded") return "Partial";
  return "Unrated";
}

function rankBadge(rank: number): string {
  return rank === 1 ? "🥇" : rank === 2 ? "🥈" : rank === 3 ? "🥉" : `#${rank}`;
}

function whenShort(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export default function Leaderboard() {
  const [exercise, setExercise] = useState<string>("");
  const [data, setData] = useState<LeaderboardResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (exId: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchLeaderboard(exId || undefined, 25);
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load leaderboard");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(exercise);
  }, [exercise, load]);

  return (
    <section className="max-w-screen-lg mx-auto px-6 xl:px-12 py-10">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4 mb-6">
        <div>
          <p className="text-xs font-semibold tracking-[0.2em] text-brand-400 uppercase mb-2">
            ● Leaderboard
          </p>
          <h1 className="text-3xl font-black text-white">Form Leaderboard</h1>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => void load(exercise)}
            className="px-3 py-2 text-xs font-medium text-slate-400 hover:text-white
                       border border-surface-600 hover:border-surface-500 rounded-lg transition-all"
            aria-label="Refresh leaderboard"
          >
            Refresh
          </button>
          <select
            value={exercise}
            onChange={(e) => setExercise(e.target.value)}
            className="px-3 py-2 text-sm bg-surface-800 border border-surface-600 rounded-lg
                       text-slate-200 focus:outline-none focus:border-brand-500"
            aria-label="Filter by exercise"
          >
            {EXERCISE_OPTIONS.map((o) => (
              <option key={o.id || "all"} value={o.id}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Body */}
      {loading ? (
        <div className="rounded-xl border border-surface-700 bg-surface-800/60 p-10 text-center text-slate-500">
          Loading leaderboard…
        </div>
      ) : error ? (
        <div className="rounded-xl border border-amber-500/30 bg-amber-900/10 p-6 text-sm text-amber-300">
          Could not load the leaderboard: {error}
        </div>
      ) : !data || data.entries.length === 0 ? (
        <div className="rounded-xl border border-surface-700 bg-surface-800/60 p-10 text-center">
          <p className="text-slate-300 font-medium mb-1">No ranked sessions yet</p>
          <p className="text-sm text-slate-500">
            Record a movement on the Gym or Basketball page and you&apos;ll appear here.
          </p>
        </div>
      ) : (
<>
          {/* Desktop table */}
          <div className="hidden md:block rounded-xl border border-surface-700 bg-surface-800/60 overflow-hidden">
            {/* Column header */}
            <div className="grid grid-cols-12 gap-2 px-5 py-3 border-b border-surface-700
                            text-xs uppercase tracking-wider text-slate-500 font-semibold">
              <div className="col-span-1">Rank</div>
              <div className="col-span-4">Athlete</div>
              <div className="col-span-2">Exercise</div>
              <div className="col-span-2 text-right">Form Score</div>
              <div className="col-span-1 text-right">Reps</div>
              <div className="col-span-2 text-right">Verified</div>
            </div>

            {data.entries.map((e) => (
              <div
                key={e.session_id}
                className="grid grid-cols-12 gap-2 px-5 py-3.5 items-center border-b border-surface-800
                           last:border-0 hover:bg-surface-800 transition-colors"
              >
                <div className="col-span-1 text-lg font-black text-slate-300">
                  {rankBadge(e.rank)}
                </div>
                <div className="col-span-4 font-medium text-slate-100 truncate">
                  {e.display_name}
                </div>
                <div className="col-span-2 text-sm text-slate-400 truncate">
                  {prettyExercise(e.exercise_id)}
                </div>
                <div className="col-span-2 text-right">
                  <span className="font-mono text-base font-bold text-white">
                    {e.form_index.toFixed(1)}
                  </span>
                  <span className={`${statusChip(e.form_index_status)} ml-2 text-[11px] px-1.5 py-0.5 rounded`}>
                    {statusLabel(e.form_index_status)}
                  </span>
                </div>
                <div className="col-span-1 text-right font-mono text-sm text-slate-400">
                  {e.n_valid_reps}/{e.n_reps}
                </div>
                <div
                  className="col-span-2 text-right text-xs text-slate-500"
                  title={e.git_commit_sha ? `Analysis code version ${e.git_commit_sha.slice(0, 12)}` : undefined}
                >
                  <span className="text-emerald-500">✓</span> {whenShort(e.created_at)}
                </div>
              </div>
            ))}
          </div>

          {/* Mobile cards — the 12-col table is unreadable on narrow screens */}
          <div className="md:hidden space-y-2">
            {data.entries.map((e) => (
              <div
                key={e.session_id}
                className="rounded-xl border border-surface-700 bg-surface-800/60 p-4
                           flex items-center justify-between gap-3"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <span className="text-lg font-black text-slate-300 shrink-0 w-7 text-center">
                    {rankBadge(e.rank)}
                  </span>
                  <div className="min-w-0">
                    <p className="font-medium text-slate-100 truncate">{e.display_name}</p>
                    <p className="text-xs text-slate-500 truncate">
                      {prettyExercise(e.exercise_id)} · {e.n_valid_reps}/{e.n_reps} reps · {whenShort(e.created_at)}
                    </p>
                  </div>
                </div>
                <div className="text-right shrink-0">
                  <p className="font-mono text-lg font-bold text-white leading-none">
                    {e.form_index.toFixed(1)}
                  </p>
                  <span className={`${statusChip(e.form_index_status)} inline-block mt-1 text-[11px] px-1.5 py-0.5 rounded`}>
                    {statusLabel(e.form_index_status)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Plain-language note — honest, without the jargon. */}
      {data && (
        <p className="text-xs text-slate-500 mt-4 leading-relaxed">
          Form Score reflects how cleanly and consistently you moved — a relative score
          for comparing sessions, not a medical or certified grade.
          {data.backend === "insforge" ? (
            <span className="text-slate-600"> Scores are saved live to InsForge.</span>
          ) : null}
        </p>
      )}
    </section>
  );
}
