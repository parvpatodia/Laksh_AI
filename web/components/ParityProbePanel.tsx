"use client";

/**
 * ParityProbePanel: "Live Counter Check" section in CanonicalReport.
 *
 * Explains — in plain English — how closely the live rep counter matched
 * the final video analysis. Rendered when the response envelope contains
 * a non-null parity_probe block (i.e. ghost rep vectors were submitted).
 */

import type { ParityProbe } from "@/lib/api";

interface Props {
  probe: ParityProbe;
}

// Wire-format field names -> friendly labels for the "fields compared" tag list.
const FIELD_FRIENDLY: Record<string, string> = {
  rep_duration_s: "Rep time",
  eccentric_duration_s: "Lowering phase",
  concentric_duration_s: "Lifting phase",
  tempo_ratio_ecc_over_con: "Tempo ratio",
  signal_amplitude: "Range of motion",
  primary_joints_min_visibility: "Pose confidence",
  primary_joints_missing_frac: "Tracking gaps",
};

function friendlyField(wire: string): string {
  return FIELD_FRIENDLY[wire] ?? wire.replace(/_/g, " ");
}

function statusSummary(status: ParityProbe["status"]): {
  headline: string;
  detail: string;
  color: string;
  dotColor: string;
} {
  switch (status) {
    case "within_tolerance":
      return {
        headline: "Live count matched",
        detail: "Your live rep counter agreed closely with the final video analysis.",
        color: "text-emerald-400",
        dotColor: "bg-emerald-400",
      };
    case "outside_tolerance":
      return {
        headline: "Some differences found",
        detail: "The live counter and the video analysis disagreed on some reps. This is normal when joints are briefly out of frame or partially occluded during live tracking.",
        color: "text-amber-400",
        dotColor: "bg-amber-400",
      };
    case "insufficient_data":
      return {
        headline: "Not enough data",
        detail: "Too few completed reps were available to compare the live count against the video analysis.",
        color: "text-slate-400",
        dotColor: "bg-slate-500",
      };
  }
}

/**
 * ParityProbePanel — shows the Live Counter Check block from a canonical response.
 */
export default function ParityProbePanel({ probe }: Props) {
  const { headline, detail, color, dotColor } = statusSummary(probe.status);

  return (
    <div className="rounded-xl border border-surface-700/50 bg-surface-900/40 px-4 py-4">
      {/* Section heading */}
      <div className="flex items-center gap-2 mb-1">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Live Counter Check
        </h3>
        <span className={`inline-flex items-center gap-1 rounded-full border text-[10px] px-2 py-0.5
                          font-medium ${color}
                          ${probe.status === "within_tolerance"
                            ? "bg-emerald-900/30 border-emerald-700/40"
                            : probe.status === "outside_tolerance"
                              ? "bg-amber-900/30 border-amber-700/40"
                              : "bg-slate-800 border-slate-700"}`}>
          <span className={`w-1 h-1 rounded-full ${dotColor}`} />
          {headline}
        </span>
      </div>

      {/* Explanation */}
      <p className="text-xs text-slate-500 leading-relaxed mb-3">{detail}</p>

      {/* Delta stats — only meaningful when there's enough data */}
      {probe.status !== "insufficient_data" && (
        <div className="grid grid-cols-2 gap-2 mb-3">
          <div className="rounded-lg bg-surface-800/70 border border-surface-700/40 px-3 py-2.5">
            <p className="text-[10px] uppercase tracking-wide text-slate-500 mb-0.5">
              Typical deviation
            </p>
            <p className="text-sm font-semibold font-mono text-slate-200 tabular-nums">
              {probe.p90_abs_delta.toFixed(4)}
            </p>
            <p className="text-[10px] text-slate-600 mt-0.5">
              90th-percentile across compared fields
            </p>
          </div>
          <div className="rounded-lg bg-surface-800/70 border border-surface-700/40 px-3 py-2.5">
            <p className="text-[10px] uppercase tracking-wide text-slate-500 mb-0.5">
              Largest deviation
            </p>
            <p className="text-sm font-semibold font-mono text-slate-200 tabular-nums">
              {probe.max_abs_delta.toFixed(4)}
            </p>
            <p className="text-[10px] text-slate-600 mt-0.5">
              Worst single field difference
            </p>
          </div>
        </div>
      )}

      {/* Fields that were compared */}
      {probe.fields_compared.length > 0 && (
        <div>
          <p className="text-[10px] text-slate-600 uppercase tracking-wide mb-1.5">
            Metrics compared
          </p>
          <div className="flex flex-wrap gap-1.5">
            {probe.fields_compared.map((f) => (
              <span
                key={f}
                className="rounded-md bg-surface-800 text-slate-400 border border-surface-700/60
                           text-[10px] px-2 py-0.5"
              >
                {friendlyField(f)}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
