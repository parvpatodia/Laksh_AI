"use client";

/**
 * GhostMetricsPanel: live display of realtime_preview ghost metrics.
 *
 * Shows rep count, current phase, smoothed signal bar, and the most
 * recently completed rep's feature vector.  All values are labelled with
 * their status chip (valid / degraded / unknown) so the distinction from
 * canonical_backend results is visually clear.
 */

import type { GhostRepMetrics, GhostField, Phase } from "@/lib/realtime/repCounter";

interface Props {
  repCount: number;
  currentPhase: Phase;
  currentSignal: number | null;
  lastRep: GhostRepMetrics | null;
  active: boolean;
}

function StatusChip({ status }: { status: GhostField["status"] }) {
  const cls =
    status === "valid"
      ? "chip-valid"
      : status === "degraded"
      ? "chip-degraded"
      : "chip-unknown";
  return (
    <span className={`${cls} text-xs px-1.5 py-0.5 rounded font-mono`}>
      {status}
    </span>
  );
}

function FieldRow({ label, f }: { label: string; f: GhostField }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-surface-700/50 last:border-0">
      <span className="text-xs text-slate-500 font-mono">{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-sm text-slate-200 font-mono tabular-nums">
          {f.value !== null ? `${f.value} ${f.unit}` : "--"}
        </span>
        <StatusChip status={f.status} />
      </div>
    </div>
  );
}

export default function GhostMetricsPanel({
  repCount,
  currentPhase,
  currentSignal,
  lastRep,
  active,
}: Props) {
  const phaseColor =
    currentPhase === "eccentric"
      ? "text-amber-400"
      : currentPhase === "concentric"
      ? "text-emerald-400"
      : "text-slate-500";

  return (
    <div className="rounded-xl border border-surface-700 bg-surface-800 p-5">
      <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
        Ghost metrics
        <span className="chip-preview text-xs px-1.5 py-0.5 rounded font-normal">
          realtime_preview
        </span>
      </h2>

      {!active ? (
        <p className="text-xs text-slate-600">Start camera to see live metrics.</p>
      ) : (
        <>
          {/* Live counters */}
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="rounded-lg bg-surface-900/60 px-3 py-2">
              <p className="text-xs text-slate-500 mb-0.5">Reps</p>
              <p className="text-2xl font-bold text-slate-100 font-mono tabular-nums">
                {repCount}
              </p>
            </div>
            <div className="rounded-lg bg-surface-900/60 px-3 py-2">
              <p className="text-xs text-slate-500 mb-0.5">Phase</p>
              <p className={`text-lg font-semibold font-mono ${phaseColor}`}>
                {currentPhase}
              </p>
            </div>
          </div>

          {/* Signal bar */}
          {currentSignal !== null && (
            <div className="mb-4">
              <div className="flex justify-between text-xs text-slate-600 mb-1">
                <span>signal</span>
                <span className="font-mono">{currentSignal.toFixed(2)}</span>
              </div>
              <div className="h-1.5 rounded-full bg-surface-700 overflow-hidden">
                <div
                  className="h-full bg-brand-500 rounded-full transition-all duration-75"
                  style={{ width: `${Math.max(2, currentSignal * 100)}%` }}
                />
              </div>
            </div>
          )}

          {/* Last rep features */}
          {lastRep ? (
            <div>
              <p className="text-xs text-slate-500 mb-2">
                Rep {lastRep.rep_index + 1} features
              </p>
              <FieldRow label="rep_duration_s"          f={lastRep.rep_duration_s} />
              <FieldRow label="eccentric_duration_s"    f={lastRep.eccentric_duration_s} />
              <FieldRow label="concentric_duration_s"   f={lastRep.concentric_duration_s} />
              <FieldRow label="tempo_ratio"             f={lastRep.tempo_ratio_ecc_over_con} />
              <FieldRow label="min_visibility"          f={lastRep.min_visibility} />
            </div>
          ) : (
            <p className="text-xs text-slate-600">
              {repCount === 0
                ? "Perform a rep to see feature breakdown."
                : "Waiting for next rep…"}
            </p>
          )}
        </>
      )}
    </div>
  );
}
