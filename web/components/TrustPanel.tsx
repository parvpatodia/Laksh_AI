"use client";

/**
 * TrustPanel: judge-facing summary of *why* the metrics displayed below
 * should be trusted.  All values come from data the backend already returns
 * (provenance, parity_probe, calibration, schema_version) -- this component
 * just promotes them from a buried <details> block to a first-class panel.
 *
 * Design intent (research-showcase context):
 *   - Sits ABOVE the metrics so judges see it without scrolling.
 *   - Each row is a single sentence: "X is verified because Y."
 *   - No interactive disclosures: trust signals must be visible at a glance.
 *
 * Honesty contract:
 *   - We do not invent confidence numbers.  Every chip maps to a real field
 *     in the response envelope.  When data is missing we say so.
 */

import type { AnalyzeResponse, ParityProbe } from "@/lib/api";

interface Props {
  result: AnalyzeResponse;
}

function shortSha(sha: string | null | undefined, n = 10): string {
  if (!sha) return "—";
  return sha.slice(0, n);
}

function parityChip(probe: ParityProbe | null): {
  label: string;
  cls: string;
  detail: string;
} {
  if (!probe) {
    return {
      label: "N/A",
      cls: "chip-unknown",
      detail: "No live data submitted",
    };
  }
  const cls =
    probe.status === "within_tolerance"
      ? "chip-valid"
      : probe.status === "outside_tolerance"
        ? "chip-degraded"
        : "chip-unknown";
  const friendlyLabel =
    probe.status === "within_tolerance" ? "Matched"
      : probe.status === "outside_tolerance" ? "Diverged"
        : "Not enough data";
  const detail =
    probe.status === "insufficient_data"
      ? "Perform more reps to compare"
      : `Typical deviation ${probe.p90_abs_delta.toFixed(2)}, largest ${probe.max_abs_delta.toFixed(2)}`;
  return { label: friendlyLabel, cls, detail };
}

export default function TrustPanel({ result }: Props) {
  const parity = parityChip(result.parity_probe);
  const calibStatus = result.calibration.evidence_status;
  const calibChipCls =
    calibStatus === "calibrated" || calibStatus === "cited"
      ? "chip-valid"
      : calibStatus === "uncalibrated_v0" || calibStatus === "no_reference_yet"
        ? "chip-unknown"
        : "chip-degraded";
  const calibFriendly =
    calibStatus === "calibrated" ? "Calibrated"
      : calibStatus === "cited" ? "Literature-verified"
        : calibStatus === "uncalibrated_v0" || calibStatus === "no_reference_yet"
          ? "No reference yet"
          : calibStatus.replace(/_/g, " ");

  return (
    <div
      className="rounded-xl border border-brand-500/30 bg-gradient-to-br
                 from-surface-800 via-surface-800 to-brand-900/10
                 p-5 mb-6"
      aria-label="Trust signals"
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
          How to trust this
        </h2>
        <span className="text-xs text-slate-500 font-mono">
          schema v{result.schema_version}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
        {/* Model + frames */}
        <div className="rounded-lg bg-surface-900/60 px-3 py-2.5">
          <p className="text-slate-500 mb-1">Pose model</p>
          <p className="text-slate-200 font-mono">{result.provenance.model}</p>
          <p className="text-slate-600 mt-1">
            {result.n_frames} frames @ {result.fps.toFixed(1)} fps
          </p>
        </div>

        {/* Parity probe */}
        <div className="rounded-lg bg-surface-900/60 px-3 py-2.5">
          <p className="text-slate-500 mb-1">Live vs. recorded accuracy</p>
          <div className="flex items-center gap-2">
            <span className={`${parity.cls} text-xs px-1.5 py-0.5 rounded`}>
              {parity.label}
            </span>
            <span className="text-slate-600">{parity.detail}</span>
          </div>
        </div>

        {/* Calibration policy */}
        <div className="rounded-lg bg-surface-900/60 px-3 py-2.5">
          <p className="text-slate-500 mb-1">Reference ranges</p>
          <div className="flex items-center gap-2">
            <span className={`${calibChipCls} text-xs px-1.5 py-0.5 rounded`}>
              {calibFriendly}
            </span>
          </div>
          <p className="text-slate-600 mt-1">
            {result.calibration.comparable_fields.length === 0
              ? "Metrics shown without grading until reference data is collected"
              : calibStatus === "cited"
                ? `${result.calibration.comparable_fields.length} metric(s) compared to published literature`
                : `${result.calibration.comparable_fields.length} metric(s) with reference ranges`}
          </p>
        </div>

        {/* Manifest fingerprints */}
        <div className="rounded-lg bg-surface-900/60 px-3 py-2.5">
          <p className="text-slate-500 mb-1">Reproducibility</p>
          <div className="text-slate-300 font-mono space-y-0.5">
            <p>
              <span className="text-slate-500">Exercise config </span>
              {shortSha(result.provenance.exercise_manifest_sha)}
            </p>
            <p>
              <span className="text-slate-500">Calibration     </span>
              {shortSha(result.provenance.calibration_manifest_sha)}
            </p>
            <p>
              <span className="text-slate-500">Code version    </span>
              {shortSha(result.provenance.git_commit_sha, 8)}
            </p>
          </div>
        </div>
      </div>

      <p className="text-[11px] text-slate-600 mt-3 leading-relaxed">
        Every metric is reported with its measurement status. Values without a reference range are shown
        but not graded — we never invent confidence we don&apos;t have.
      </p>
    </div>
  );
}
