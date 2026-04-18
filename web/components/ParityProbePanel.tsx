"use client";

/**
 * ParityProbePanel: displays the realtime-vs-canonical parity probe result.
 *
 * Rendered in CanonicalReport when the response envelope contains a non-null
 * parity_probe block (i.e. ghost rep vectors were submitted with the upload).
 * Shows the agreement status, aggregate delta statistics, and the list of
 * fields that were compared.
 */

import type { ParityProbe } from "@/lib/api";

interface Props {
  probe: ParityProbe;
}

/**
 * Human-readable one-line description for each probe status.
 */
function statusExplanation(status: ParityProbe["status"]): string {
  switch (status) {
    case "within_tolerance":
      return "Realtime ghost metrics agree with the canonical backend result.";
    case "outside_tolerance":
      return "Realtime ghost metrics diverge significantly from the canonical result. Pose occlusion or signal lag may be the cause.";
    case "insufficient_data":
      return "Too few valid field pairs to compute a reliable parity score.";
  }
}

/**
 * CSS classes for the status badge by probe status value.
 */
function badgeClass(status: ParityProbe["status"]): string {
  switch (status) {
    case "within_tolerance":
      return "chip-valid";
    case "outside_tolerance":
      return "bg-rose-900/40 text-rose-300 border border-rose-700/50";
    case "insufficient_data":
      return "bg-slate-800 text-slate-400 border border-slate-700";
  }
}

/**
 * ParityProbePanel renders the parity_probe block from a canonical response.
 *
 * Only rendered when ``probe`` is non-null (guarded by the parent component).
 */
export default function ParityProbePanel({ probe }: Props) {
  return (
    <div className="rounded-lg border border-surface-700/60 bg-surface-900/40 px-4 py-3 mb-4">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Parity probe
        </span>
        <span className={`${badgeClass(probe.status)} text-xs px-1.5 py-0.5 rounded font-mono`}>
          {probe.status.replace(/_/g, " ")}
        </span>
      </div>

      <p className="text-xs text-slate-500 mb-3">{statusExplanation(probe.status)}</p>

      {probe.status !== "insufficient_data" && (
        <div className="grid grid-cols-2 gap-2 mb-3">
          <div className="rounded bg-surface-800 px-3 py-2">
            <p className="text-xs text-slate-500 mb-0.5">p90 abs delta</p>
            <p className="text-sm font-mono text-slate-200 tabular-nums">
              {probe.p90_abs_delta.toFixed(4)}
            </p>
          </div>
          <div className="rounded bg-surface-800 px-3 py-2">
            <p className="text-xs text-slate-500 mb-0.5">max abs delta</p>
            <p className="text-sm font-mono text-slate-200 tabular-nums">
              {probe.max_abs_delta.toFixed(4)}
            </p>
          </div>
        </div>
      )}

      {probe.fields_compared.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {probe.fields_compared.map((f) => (
            <span
              key={f}
              className="bg-surface-800 text-slate-500 border border-surface-700 text-xs px-1.5 py-0.5 rounded font-mono"
            >
              {f}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
