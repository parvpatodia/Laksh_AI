"use client";

/**
 * GhostMetricsPanel - live realtime_preview ghost metrics.
 *
 * Scoreboard aesthetic: stadium-style counter with glow, broadcast phase
 * indicator, animated signal bar, and per-rep breakdown.
 */

import type { GhostRepMetrics, GhostField, Phase } from "@/lib/realtime/repCounter";
import { LIVE_COUNTER_DISCLAIMER } from "@/lib/realtime/repCounter";

interface Props {
  repCount: number;
  currentPhase: Phase;
  currentSignal: number | null;
  lastRep: GhostRepMetrics | null;
  active: boolean;
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
// Animated signal waveform bars
// ---------------------------------------------------------------------------

function SignalBars({ signal }: { signal: number }) {
  const bars = 16;
  const filled = Math.round(signal * bars);
  return (
    <div className="flex items-end gap-0.5 h-6">
      {Array.from({ length: bars }, (_, i) => {
        const active = i < filled;
        const heightPct = 30 + ((i / bars) * 70);
        return (
          <div
            key={i}
            className={`w-1.5 rounded-sm transition-all duration-100 ${
              active
                ? i < filled * 0.5 ? "bg-emerald-500" : i < filled * 0.8 ? "bg-brand-500" : "bg-amber-400"
                : "bg-surface-700"
            }`}
            style={{ height: `${heightPct}%` }}
          />
        );
      })}
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
  type PhaseConfig = { label: string; color: string; dot: string; glow: string };
  const phaseConfig: Record<Phase, PhaseConfig> = {
    eccentric:  { label: "Lowering", color: "text-amber-400",   dot: "bg-amber-400",   glow: "shadow-amber-500/30" },
    concentric: { label: "Lifting",  color: "text-emerald-400", dot: "bg-emerald-400", glow: "shadow-emerald-500/30" },
    rest:       { label: "Ready",    color: "text-slate-500",   dot: "bg-slate-600",   glow: "" },
  };
  const phaseInfo = phaseConfig[currentPhase] ?? phaseConfig.rest;
  const counterLabel = unitLabel === "shot" ? "SHOTS" : "REPS";
  const unitSingular = unitLabel === "shot" ? "Shot" : "Rep";

  return (
    <div className="rounded-2xl border border-surface-700 bg-surface-800 overflow-hidden">

      {/* Header bar */}
      <div className="flex items-center justify-between px-5 pt-4 pb-3 border-b border-surface-700/50">
        <div className="flex items-center gap-2.5">
          {/* Broadcast-style icon */}
          <div className="w-7 h-7 rounded-lg bg-brand-500/10 border border-brand-500/30 flex items-center justify-center">
            <svg className="w-3.5 h-3.5 text-brand-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <circle cx="12" cy="12" r="2" />
              <path strokeLinecap="round" d="M4.93 4.93a10 10 0 0 0 0 14.14M19.07 4.93a10 10 0 0 1 0 14.14M7.76 7.76a6 6 0 0 0 0 8.49M16.24 7.76a6 6 0 0 1 0 8.49" />
            </svg>
          </div>
          <h2 className="text-sm font-bold text-slate-200">Live Tracking</h2>
        </div>
        {active && (
          <span className="inline-flex items-center gap-1.5 text-[10px] text-emerald-400
                           border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 rounded-full font-bold">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            LIVE
          </span>
        )}
      </div>

      {!active ? (
        /* Idle state */
        <div className="flex flex-col items-center justify-center py-12 px-5 text-center">
          <div className="w-14 h-14 rounded-2xl bg-surface-700/50 border border-surface-600/50
                          flex items-center justify-center mb-4">
            <svg className="w-7 h-7 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z" />
            </svg>
          </div>
          <p className="text-sm font-semibold text-slate-400">Waiting for camera</p>
          <p className="text-xs text-slate-600 mt-1">Start camera to see live metrics</p>
        </div>
      ) : (
        <div className="p-5 space-y-4">

          {/* Stadium scoreboard counter */}
          <div className="relative rounded-xl overflow-hidden border border-surface-700/50">
            {/* Background gradient */}
            <div className="absolute inset-0 bg-gradient-to-br from-surface-950 to-surface-900" />
            {/* Subtle grid */}
            <div
              className="absolute inset-0 opacity-[0.04] pointer-events-none"
              style={{
                backgroundImage: "linear-gradient(rgba(255,255,255,.5) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.5) 1px, transparent 1px)",
                backgroundSize: "20px 20px",
              }}
            />
            <div className="relative px-5 py-5 flex items-center justify-between">
              {/* Counter */}
              <div>
                <p className="text-[9px] font-bold uppercase tracking-[0.2em] text-slate-600 mb-2">
                  {counterLabel}
                </p>
                <p
                  className="font-black font-mono tabular-nums leading-none text-white"
                  style={{
                    fontSize: "clamp(3rem, 6vw, 4.5rem)",
                    textShadow: repCount > 0 ? "0 0 40px rgba(14,165,233,0.4)" : "none",
                  }}
                >
                  {String(repCount).padStart(2, "0")}
                </p>
              </div>

              {/* Phase indicator */}
              <div className="text-right">
                <p className="text-[9px] font-bold uppercase tracking-[0.2em] text-slate-600 mb-2">
                  Phase
                </p>
                <div className="flex items-center gap-2 justify-end">
                  <div className={`w-2.5 h-2.5 rounded-full ${phaseInfo.dot} shadow-lg ${phaseInfo.glow}`}
                       style={{ animation: currentPhase !== "rest" ? "pulse 1s ease-in-out infinite" : "none" }} />
                  <p className={`text-xl font-black ${phaseInfo.color}`}>{phaseInfo.label}</p>
                </div>
                {/* Phase arc */}
                <div className="flex gap-1 mt-2 justify-end">
                  {(["eccentric", "rest", "concentric"] as Phase[]).map((p) => (
                    <div
                      key={p}
                      className={`h-0.5 w-5 rounded-full transition-all duration-300 ${
                        currentPhase === p ? phaseInfo.dot : "bg-surface-700"
                      }`}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Signal strength */}
          {currentSignal !== null && (
            <div className="rounded-xl bg-surface-900/60 border border-surface-700/40 px-4 py-3">
              <div className="flex items-center justify-between mb-2.5">
                <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-600">
                  Signal Strength
                </span>
                <span className="text-xs font-mono font-bold text-slate-300">
                  {(currentSignal * 100).toFixed(0)}%
                </span>
              </div>
              <SignalBars signal={currentSignal} />
            </div>
          )}

          {/* Last rep breakdown */}
          {lastRep ? (
            <div className="rounded-xl bg-surface-900/60 border border-surface-700/40 px-4 py-3">
              <div className="flex items-center justify-between mb-3">
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
                  {unitSingular} {lastRep.rep_index + 1} -- Breakdown
                </p>
                <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-brand-500/10 text-brand-400 border border-brand-500/30">
                  LAST
                </span>
              </div>
              <FieldRow label="Duration"  wire="rep_duration_s"        f={lastRep.rep_duration_s}        format={(v) => `${v.toFixed(2)}s`} />
              <FieldRow label={unitLabel === "shot" ? "Wind-up" : "Lowering"} wire="eccentric_duration_s" f={lastRep.eccentric_duration_s} format={(v) => `${v.toFixed(2)}s`} />
              <FieldRow label={unitLabel === "shot" ? "Release" : "Lifting"}  wire="concentric_duration_s" f={lastRep.concentric_duration_s} format={(v) => `${v.toFixed(2)}s`} />
              <FieldRow label="Tempo ratio"      wire="tempo_ratio_ecc_over_con"  f={lastRep.tempo_ratio_ecc_over_con}  format={(v) => `${v.toFixed(2)}x`} />
              <FieldRow label="Range of motion"  wire="signal_amplitude"          f={lastRep.signal_amplitude}          format={(v) => v.toFixed(2)} />
              <FieldRow label="Pose confidence"  wire="min_visibility"            f={lastRep.min_visibility}            format={(v) => `${(v * 100).toFixed(0)}%`} />
            </div>
          ) : (
            <div className="rounded-xl bg-surface-900/40 border border-surface-700/30 px-4 py-4 text-center">
              <p className="text-xs text-slate-600">
                {repCount === 0
                  ? `Perform a ${unitLabel} to see the breakdown`
                  : `Waiting for next ${unitLabel}...`}
              </p>
            </div>
          )}

          {/* Disclaimer */}
          <p className="text-[10px] text-slate-700 leading-snug px-0.5">{LIVE_COUNTER_DISCLAIMER}</p>
        </div>
      )}
    </div>
  );
}
