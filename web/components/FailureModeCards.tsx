"use client";

/**
 * FailureModeCards: renders the three hardcoded failure-mode demo cards.
 *
 * Each card shows what the system outputs when something goes wrong --
 * occluded joints, no reps detected, or multiple people in frame.
 * The goal is to demonstrate that the system surfaces problems honestly
 * via reason_codes rather than silently returning bad numbers.
 */

import type { FieldValue, RepVector } from "@/lib/api";
import { FAILURE_MODES, type FailureMode } from "@/lib/failureModes";

// ---------------------------------------------------------------------------
// Shared sub-components (mirrors CanonicalReport style)
// ---------------------------------------------------------------------------

/** Coloured chip for field-level status (valid / degraded / unknown). */
function StatusChip({ status }: { status: FieldValue["status"] }) {
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

/**
 * Compact row showing a feature name, its value (or "--"), and its status chip.
 * Only renders up to `maxFields` entries so cards stay compact.
 */
function MiniFeatureList({
  features,
  maxFields,
}: {
  features: RepVector["features"];
  maxFields: number;
}) {
  const entries = Object.entries(features).slice(0, maxFields);
  return (
    <div className="space-y-1 mt-2">
      {entries.map(([key, fv]) => (
        <div
          key={key}
          className="flex items-center justify-between gap-2 text-xs"
        >
          <span className="text-slate-500 font-mono truncate">{key}</span>
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="text-slate-300 font-mono tabular-nums">
              {fv.value !== null ? `${fv.value} ${fv.unit}`.trim() : "--"}
            </span>
            <StatusChip status={fv.status} />
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * Mini rep list: one row per rep with rep_status chip and a few key fields.
 * When there are no reps, shows the "no reps" empty state instead.
 */
function MiniRepList({ reps }: { reps: RepVector[] }) {
  if (reps.length === 0) {
    return (
      <p className="text-xs text-slate-500 italic mt-2">
        No reps detected -- feature_vectors is empty.
      </p>
    );
  }
  return (
    <div className="space-y-2 mt-2">
      {reps.map((rep) => {
        const borderColor =
          rep.rep_status === "valid"
            ? "border-emerald-700/40"
            : rep.rep_status === "degraded"
            ? "border-amber-700/40"
            : "border-slate-700/40";
        return (
          <div
            key={rep.rep_index}
            className={`rounded border ${borderColor} bg-surface-900/50 px-3 py-2`}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-slate-400">
                Rep {rep.rep_index + 1}
              </span>
              <StatusChip status={rep.rep_status} />
            </div>
            <MiniFeatureList features={rep.features} maxFields={3} />
          </div>
        );
      })}
    </div>
  );
}

/**
 * Extracts all unique reason_codes from a failure mode's reps and segment,
 * returning them deduplicated for display.
 */
function extractReasonCodes(mode: FailureMode): string[] {
  const codes = new Set<string>();
  const seg = mode.result.segment as
    | { reason_codes?: string[] }
    | null
    | undefined;
  if (seg && Array.isArray(seg.reason_codes)) {
    seg.reason_codes.forEach((c) => codes.add(c));
  }
  for (const rep of mode.result.feature_vectors) {
    for (const fv of Object.values(rep.features)) {
      fv.reason_codes.forEach((c) => codes.add(c));
    }
  }
  return Array.from(codes);
}

/** Human-readable explanation of what the system does for each reason_code. */
const REASON_CODE_EXPLANATIONS: Record<string, string> = {
  visibility_below_threshold:
    "Landmark confidence fell below the minimum threshold. The field is marked degraded rather than fabricating a number.",
  insufficient_signal_variance:
    "The joint angle signal did not vary enough to locate rep boundaries. Zero reps are reported rather than guessing.",
  multi_person_ambiguity:
    "Two or more pose skeletons overlap. The system cannot safely assign landmarks to one athlete, so all reps are flagged degraded.",
};

/**
 * Single failure-mode card.
 * Shows icon + label + description, then the rep list, then how reason_codes
 * surface the failure to the caller.
 */
function FailureModeCard({ mode }: { mode: FailureMode }) {
  const reasonCodes = extractReasonCodes(mode);
  const segStatus = (
    mode.result.segment as { status?: string } | null | undefined
  )?.status;

  return (
    <div className="rounded-xl border border-surface-700 bg-surface-800 p-5 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-start gap-3">
        <span className="text-xl shrink-0 font-mono text-slate-300">
          {mode.icon}
        </span>
        <div>
          <h3 className="text-sm font-semibold text-slate-200">{mode.label}</h3>
          <p className="text-xs text-slate-400 mt-0.5 leading-relaxed">
            {mode.description}
          </p>
        </div>
      </div>

      {/* Segment status */}
      {segStatus && (
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span className="font-mono text-slate-400">segment.status</span>
          <StatusChip status={segStatus as FieldValue["status"]} />
        </div>
      )}

      {/* Rep list */}
      <MiniRepList reps={mode.result.feature_vectors} />

      {/* What the system does */}
      <div className="rounded-lg border border-surface-700/50 bg-surface-900/40 px-3 py-2.5">
        <p className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">
          What the system does
        </p>
        {reasonCodes.length === 0 ? (
          <p className="text-xs text-slate-500">No reason codes emitted.</p>
        ) : (
          <ul className="space-y-1.5">
            {reasonCodes.map((code) => (
              <li key={code}>
                <span className="font-mono text-xs text-amber-400">{code}</span>
                {REASON_CODE_EXPLANATIONS[code] && (
                  <p className="text-xs text-slate-500 mt-0.5">
                    {REASON_CODE_EXPLANATIONS[code]}
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section component
// ---------------------------------------------------------------------------

/**
 * FailureModeCards section.
 *
 * Renders all three failure-mode demo cards in a responsive grid.
 * Intended to appear on the home page below the research contribution callout
 * to show that the system handles bad inputs honestly.
 */
export default function FailureModeCards() {
  return (
    <section aria-labelledby="failure-modes-heading" className="mt-10">
      <div className="mb-6">
        <h2
          id="failure-modes-heading"
          className="text-sm font-semibold text-slate-300 uppercase tracking-wider"
        >
          Failure modes &amp; honesty
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          The system never silently returns bad numbers. When detection fails,{" "}
          <code className="font-mono text-xs bg-surface-700 px-1 rounded">
            reason_codes
          </code>{" "}
          explain the failure and fields are marked{" "}
          <code className="font-mono text-xs bg-surface-700 px-1 rounded">
            degraded
          </code>{" "}
          or{" "}
          <code className="font-mono text-xs bg-surface-700 px-1 rounded">
            unknown
          </code>{" "}
          rather than fabricated.
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {FAILURE_MODES.map((mode) => (
          <FailureModeCard key={mode.id} mode={mode} />
        ))}
      </div>
    </section>
  );
}
