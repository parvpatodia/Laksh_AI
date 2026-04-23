"use client";

/**
 * GhostMetricsPanel — live realtime_preview ghost metrics with
 * user-facing labels.  The wire-format field names are kept in a
 * <details> block so technical reviewers can still see them.
 */

import type { GhostRepMetrics, GhostField, Phase } from "@/lib/realtime/repCounter";
import { LIVE_COUNTER_DISCLAIMER } from "@/lib/realtime/repCounter";

interface Props {
  repCount: number;
  currentPhase: Phase;
  currentSignal: number | null;
  lastRep: GhostRepMetrics | null;
  active: boolean;
  /** "rep" for gym, "shot" for basketball — labels the counter accurately. */
  unitLabel?: "rep" | "shot";
}

function StatusChip({ status }: { status: GhostField["status"] }) {
  const cls =
    status === "valid" ? "chip-valid"
      : status === "degraded" ? "chip-degraded"
        : "chip-unknown";
  return (
    <span className={`${cls} text-xs px-1.5 py-0.5 rounded font-mono`}>
      {status}
    </span>
  );
}

function FieldRow({
  label,
  wire,
  f,
  format,
}: {
  label: string;
  wire: string;
  f: GhostField;
  format?: (v: number) => string;
}) {
  const display =
    f.value !== null
      ? format
        ? `${format(f.value)} ${f.unit !== "ratio" && f.unit !== "norm" ? f.unit : ""}`.trim()
        : `${f.value} ${f.unit}`
      : "—";
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-surface-700/50 last:border-0 gap-2">
      <span className="text-xs text-slate-300" title={wire}>{label}</span>
      <div className="flex items-center gap-2 shrink-0">
        <span className="text-sm text-slate-200 font-mono tabular-nums">{display}</span>
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
  unitLabel = "rep",
}: Props) {
  const phaseColor =
    currentPhase === "eccentric" ? "text-amber-400"
      : currentPhase === "concentric" ? "text-emerald-400"
        : "text-slate-500";

  const counterLabel = unitLabel === "shot" ? "Shots" : "Reps";

  return (
    <div className="rounded-xl border border-surface-700 bg-surface-800 p-5">
      <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
        Live metrics
        <span className="chip-preview text-xs px-1.5 py-0.5 rounded font-normal">
          realtime_preview
        </span>
      </h2>

      {!active ? (
        <p className="text-xs text-slate-600">Start camera to see live metrics.</p>
      ) : (
        <>
          {/* Live counters */}
          <div className="grid grid-cols-2 gap-3 mb-1">
            <div className="rounded-lg bg-surface-900/60 px-3 py-2">
              <p className="text-xs text-slate-500 mb-0.5">{counterLabel}</p>
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
          {/* A3: Honest disclaimer — live counter is a real-time preview, not the final report count */}
          <p className="text-[10px] text-slate-600 mb-3 leading-tight">{LIVE_COUNTER_DISCLAIMER}</p>

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
                {unitLabel === "shot" ? "Shot" : "Rep"} {lastRep.rep_index + 1} breakdown
              </p>
              <FieldRow
                label="Rep duration"
                wire="rep_duration_s"
                f={lastRep.rep_duration_s}
                format={(v) => v.toFixed(2)}
              />
              <FieldRow
                label={unitLabel === "shot" ? "Wind-up time" : "Lowering time"}
                wire="eccentric_duration_s"
                f={lastRep.eccentric_duration_s}
                format={(v) => v.toFixed(2)}
              />
              <FieldRow
                label={unitLabel === "shot" ? "Release time" : "Lifting time"}
                wire="concentric_duration_s"
                f={lastRep.concentric_duration_s}
                format={(v) => v.toFixed(2)}
              />
              <FieldRow
                label="Tempo (lower / lift)"
                wire="tempo_ratio_ecc_over_con"
                f={lastRep.tempo_ratio_ecc_over_con}
                format={(v) => `${v.toFixed(2)}×`}
              />
              <FieldRow
                label="Range of motion"
                wire="signal_amplitude"
                f={lastRep.signal_amplitude}
                format={(v) => v.toFixed(2)}
              />
              <FieldRow
                label="Tracking quality"
                wire="min_visibility"
                f={lastRep.min_visibility}
                format={(v) => `${(v * 100).toFixed(0)}%`}
              />
            </div>
          ) : (
            <p className="text-xs text-slate-600">
              {repCount === 0
                ? `Perform a ${unitLabel} to see the breakdown.`
                : `Waiting for next ${unitLabel}…`}
            </p>
          )}

          <p className="text-[11px] text-slate-600 mt-3 leading-relaxed">
            Counts only after a 0.5 s warm-up and only if each {unitLabel} clears
            duration, range-of-motion, and visibility gates. Under-counting by 1
            is possible; over-counting from camera noise is not.
          </p>
        </>
      )}
    </div>
  );
}
