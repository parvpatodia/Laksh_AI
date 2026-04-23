"use client";

/**
 * GhostMetricsPanel - live realtime_preview ghost metrics.
 *
 * Scoreboard aesthetic: giant rep counter, phase indicator with colour,
 * animated signal bar, and per-rep breakdown.
 */

import type { GhostRepMetrics, GhostField, Phase } from "@/lib/realtime/repCounter";
import { LIVE_COUNTER_DISCLAIMER } from "@/lib/realtime/repCounter";

interface Props {
  repCount: number;
  currentPhase: Phase;
  currentSignal: number | null;
  lastRep: GhostRepMetrics | null;
  active: boolean;
  /** "rep" for gym, "shot" for basketball -- labels the counter accurately. */
  unitLabel?: "rep" | "shot";
}

// ---------------------------------------------------------------------------
// FieldRow: compact key/value row with status dot
// ---------------------------------------------------------------------------

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
  const UNITLESS = new Set(["ratio", "norm", "fraction", ""]);
  const display =
    f.value !== null
      ? format
        ? `${format(f.value)} ${!UNITLESS.has(f.unit) ? f.unit : ""}`.trim()
        : `${f.value} ${f.unit}`
      : "--";

  const statusDot =
    f.status === "valid"    ? "bg-emerald-400"
    : f.status === "degraded" ? "bg-amber-400"
    : "bg-slate-600";

  return (
    <div className="flex items-center justify-between py-2 border-b border-surface-700/30 last:border-0">
      <div className="flex items-center gap-2">
        <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${statusDot}`} />
        <span className="text-xs text-slate-400" title={wire}>{label}</span>
      </div>
      <span className="text-sm text-slate-200 font-mono tabular-nums">{display}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function GhostMetricsPanel({
  repCount,
  currentPhase,
  currentSignal,
  lastRep,
  active,
  unitLabel = "rep",
}: Props) {
  type PhaseConfig = { label: string; color: string; dot: string };
  const phaseConfig: Record<Phase, PhaseConfig> = {
    eccentric:  { label: "Lowering", color: "text-amber-400",   dot: "bg-amber-400" },
    concentric: { label: "Lifting",  color: "text-emerald-400", dot: "bg-emerald-400" },
    rest:       { label: "Ready",    color: "text-slate-500",   dot: "bg-slate-600" },
  };
  const phaseInfo = phaseConfig[currentPhase] ?? phaseConfig.rest;
  const counterLabel = unitLabel === "shot" ? "SHOTS" : "REPS";

  return (
    <div className="rounded-2xl border border-surface-700 bg-surface-800 p-5">

      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <h2 className="text-base font-semibold text-slate-200">Live Tracking</h2>
        {active && (
          <span className="inline-flex items-center gap-1.5 text-[10px] text-brand-400 border border-brand-500/30 bg-brand-500/10 px-2 py-0.5 rounded-full font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse" />
            Live
          </span>
        )}
      </div>

      {!active ? (
        <p className="text-xs text-slate-600 py-4">Start camera to see live metrics.</p>
      ) : (
        <>
          {/* Scoreboard: counter + phase side by side */}
          <div className="rounded-xl bg-surface-900/70 border border-surface-700/50 px-4 py-4 mb-4 flex items-center justify-between">
            <div>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest font-medium mb-1.5">
                {counterLabel}
              </p>
              <p className="text-5xl font-black font-mono tabular-nums text-slate-100 leading-none">
                {repCount}
              </p>
            </div>
            <div className="text-right">
              <p className="text-[10px] text-slate-500 uppercase tracking-widest font-medium mb-1.5">
                Phase
              </p>
              <div className="flex items-center gap-1.5 justify-end">
                <span className={`w-2 h-2 rounded-full ${phaseInfo.dot}`} />
                <p className={`text-lg font-bold ${phaseInfo.color}`}>{phaseInfo.label}</p>
              </div>
            </div>
          </div>

          {/* Signal strength bar */}
          {currentSignal !== null && (
            <div className="mb-4">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] text-slate-600 uppercase tracking-widest">Signal</span>
                <span className="text-xs font-mono text-slate-400">{currentSignal.toFixed(2)}</span>
              </div>
              <div className="h-2 rounded-full bg-surface-700 overflow-hidden">
                <div
                  className="h-full rounded-full bg-brand-500 transition-all duration-75"
                  style={{ width: `${Math.max(2, currentSignal * 100)}%` }}
                />
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <p className="text-[10px] text-slate-600 mb-3 leading-snug">{LIVE_COUNTER_DISCLAIMER}</p>

          {/* Last rep breakdown */}
          {lastRep ? (
            <div>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">
                {unitLabel === "shot" ? "Shot" : "Rep"} {lastRep.rep_index + 1} breakdown
              </p>
              <FieldRow
                label="Duration"
                wire="rep_duration_s"
                f={lastRep.rep_duration_s}
                format={(v) => `${v.toFixed(2)}s`}
              />
              <FieldRow
                label={unitLabel === "shot" ? "Wind-up" : "Lowering"}
                wire="eccentric_duration_s"
                f={lastRep.eccentric_duration_s}
                format={(v) => `${v.toFixed(2)}s`}
              />
              <FieldRow
                label={unitLabel === "shot" ? "Release" : "Lifting"}
                wire="concentric_duration_s"
                f={lastRep.concentric_duration_s}
                format={(v) => `${v.toFixed(2)}s`}
              />
              <FieldRow
                label="Tempo ratio"
                wire="tempo_ratio_ecc_over_con"
                f={lastRep.tempo_ratio_ecc_over_con}
                format={(v) => `${v.toFixed(2)}x`}
              />
              <FieldRow
                label="Range of motion"
                wire="signal_amplitude"
                f={lastRep.signal_amplitude}
                format={(v) => v.toFixed(2)}
              />
              <FieldRow
                label="Pose confidence"
                wire="min_visibility"
                f={lastRep.min_visibility}
                format={(v) => `${(v * 100).toFixed(0)}%`}
              />
            </div>
          ) : (
            <p className="text-xs text-slate-600">
              {repCount === 0
                ? `Perform a ${unitLabel} to see the breakdown.`
                : `Waiting for next ${unitLabel}...`}
            </p>
          )}
        </>
      )}
    </div>
  );
}
