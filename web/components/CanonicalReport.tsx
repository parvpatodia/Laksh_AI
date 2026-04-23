"use client";

/**
 * CanonicalReport: displays the full v1 form analysis response envelope.
 *
 * Sports stats card aesthetic: bold scorecard at top, per-rep accordion below,
 * provenance collapsed for technical reviewers.
 */

import { useEffect, useState } from "react";
import type {
  AnalyzeResponse,
  CalibrationField,
  FieldValue,
  RepVector,
} from "@/lib/api";
import ParityProbePanel from "@/components/ParityProbePanel";

// ---------------------------------------------------------------------------
// Metric definitions
// ---------------------------------------------------------------------------

interface MetricMeta {
  label: string;
  description: string;
  format?: (v: number, unit: string) => string;
  interpret?: (v: number) => string | null;
}

const METRIC_META: Record<string, MetricMeta> = {
  rep_duration_s: {
    label: "Total Rep Time",
    description: "Complete rep from start to finish.",
    format: (v) => `${v.toFixed(2)}s`,
    interpret: (v) => v < 1.0 ? "Fast -- consider slowing down" : v > 4.0 ? "Controlled pace" : null,
  },
  eccentric_duration_s: {
    label: "Lowering Phase",
    description: "Time lowering the weight. Slower builds more muscle.",
    format: (v) => `${v.toFixed(2)}s`,
    interpret: (v) => v < 0.8 ? "Try to slow the lowering" : null,
  },
  concentric_duration_s: {
    label: "Lifting Phase",
    description: "Time through the working portion of the rep.",
    format: (v) => `${v.toFixed(2)}s`,
  },
  tempo_ratio_ecc_over_con: {
    label: "Tempo Ratio",
    description: "Lowering time / lifting time. Above 1.0 means controlled lowering.",
    format: (v) => `${v.toFixed(2)}x`,
    interpret: (v) => v >= 1.0 ? "Good -- controlled lowering" : "Lowering faster than lifting",
  },
  signal_amplitude: {
    label: "Range of Motion",
    description: "Total joint angle swept. Larger = fuller range.",
    format: (v, u) => `${v.toFixed(1)}${u === "deg" ? "\u00b0" : " " + u}`,
  },
  primary_joints_min_visibility: {
    label: "Pose Confidence",
    description: "How reliably key joints were tracked. Above 70% is solid.",
    format: (v) => `${(v * 100).toFixed(0)}%`,
    interpret: (v) => v < 0.5 ? "Low -- measurements may be imprecise" : v >= 0.8 ? "Good tracking" : null,
  },
  primary_joints_missing_frac: {
    label: "Tracking Gaps",
    description: "Share of frames where a key joint was not detected. Lower is better.",
    format: (v) => `${(v * 100).toFixed(0)}%`,
    interpret: (v) => v > 0.20 ? "High -- consider re-recording with better lighting" : null,
  },
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function RepStatusBadge({ status }: { status: FieldValue["status"] | string }) {
  if (status === "valid") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-900/40 text-emerald-400
                       border border-emerald-700/50 text-[10px] px-2 py-0.5 font-medium">
        <span className="w-1 h-1 rounded-full bg-emerald-400" />
        Good
      </span>
    );
  }
  if (status === "degraded") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-amber-900/30 text-amber-400
                       border border-amber-700/40 text-[10px] px-2 py-0.5 font-medium">
        <span className="w-1 h-1 rounded-full bg-amber-400" />
        Partial
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-slate-800 text-slate-500
                     border border-slate-700 text-[10px] px-2 py-0.5 font-medium">
      Unknown
    </span>
  );
}

// CalibStatusBadge kept for potential use in calibration section
function CalibStatusBadge({ status }: { status: CalibrationField["status"] }) {
  if (status === "within_reference") {
    return (
      <span className="rounded-full bg-emerald-900/30 text-emerald-400 border border-emerald-700/40
                       text-[10px] px-2 py-0.5 font-medium">
        In range
      </span>
    );
  }
  if (status === "outside_reference") {
    return (
      <span className="rounded-full bg-amber-900/30 text-amber-400 border border-amber-700/40
                       text-[10px] px-2 py-0.5 font-medium">
        Outside range
      </span>
    );
  }
  return (
    <span className="rounded-full bg-slate-800 text-slate-500 border border-slate-700
                     text-[10px] px-2 py-0.5 font-medium">
      No reference
    </span>
  );
}

function MetricRow({ wireKey, f }: { wireKey: string; f: FieldValue }) {
  const meta = METRIC_META[wireKey];
  const label = meta?.label ?? wireKey.replace(/_/g, " ");
  const description = meta?.description ?? null;

  const displayValue =
    f.value !== null
      ? meta?.format
        ? meta.format(f.value, f.unit)
        : `${f.value} ${f.unit}`.trim()
      : "--";

  const hint = f.value !== null && meta?.interpret ? meta.interpret(f.value) : null;

  return (
    <div className="py-3 border-b border-surface-700/30 last:border-0">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-slate-200 leading-tight">{label}</p>
          {description && (
            <p className="text-xs text-slate-500 mt-0.5 leading-snug">{description}</p>
          )}
          {hint && (
            <p className="text-xs text-brand-400/80 mt-1">{hint}</p>
          )}
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <span className="text-base font-semibold text-slate-100 font-mono tabular-nums">
            {displayValue}
          </span>
          <RepStatusBadge status={f.status} />
        </div>
      </div>
    </div>
  );
}

function RepCard({ rep, index }: { rep: RepVector; index: number }) {
  const [open, setOpen] = useState(index === 0);

  const borderColor =
    rep.rep_status === "valid"    ? "border-emerald-700/40"
    : rep.rep_status === "degraded" ? "border-amber-700/40"
    : "border-slate-700/50";

  const headerAccent =
    rep.rep_status === "valid"    ? "border-l-emerald-500"
    : rep.rep_status === "degraded" ? "border-l-amber-500"
    : "border-l-slate-600";

  return (
    <div className={`rounded-xl border ${borderColor} overflow-hidden`}>
      <button
        onClick={() => setOpen((v) => !v)}
        className={`w-full flex items-center justify-between px-4 py-3 border-l-2 ${headerAccent}
                    bg-surface-900/40 hover:bg-white/5 transition-colors text-left`}
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-slate-200">Rep {rep.rep_index + 1}</span>
          <RepStatusBadge status={rep.rep_status} />
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[10px] text-slate-600 font-mono">
            frames {rep.start_frame}-{rep.end_frame}
          </span>
          <span className={`text-slate-500 text-xs transition-transform ${open ? "rotate-180" : ""}`}>
            &#9660;
          </span>
        </div>
      </button>

      {open && (
        <div className="px-4 pb-2 pt-1">
          {Object.entries(rep.features).map(([key, fv]) => (
            <MetricRow key={key} wireKey={key} f={fv} />
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Upload progress indicator
// ---------------------------------------------------------------------------

function AnalysisProgress() {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, []);

  const stages = [
    { at: 0,  msg: "Uploading clip..." },
    { at: 3,  msg: "Running MediaPipe Heavy -- 33-landmark pose extraction..." },
    { at: 20, msg: "Measuring per-rep biomechanics..." },
    { at: 40, msg: "Computing rep scores and calibration comparison..." },
  ];
  const current = [...stages].reverse().find((s) => elapsed >= s.at) ?? stages[0];
  const pct = Math.min(97, (elapsed / 60) * 100);

  return (
    <div className="py-8">
      <div className="flex items-center gap-3 mb-5">
        <div className="w-5 h-5 rounded-full border-2 border-brand-500 border-t-transparent animate-spin shrink-0" />
        <div>
          <p className="text-sm font-medium text-slate-200">{current.msg}</p>
          <p className="text-xs text-slate-500 mt-0.5">
            {elapsed}s elapsed -- typically 30-60s for a 15-second clip
          </p>
        </div>
      </div>
      <div className="h-1 rounded-full bg-surface-700 overflow-hidden">
        <div
          className="h-full bg-brand-500 rounded-full transition-all duration-1000"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Props + main component
// ---------------------------------------------------------------------------

interface UploadState {
  status: "idle" | "uploading" | "done" | "error";
  error?: string;
}

interface Props {
  result: AnalyzeResponse | null;
  uploadState: UploadState;
  capturedBlob: Blob | null;
  exerciseId: string | null;
  onUpload: () => void;
}

export default function CanonicalReport({
  result,
  uploadState,
  capturedBlob,
  exerciseId,
  onUpload,
}: Props) {
  // Pre-result state
  if (!result) {
    return (
      <div className="rounded-2xl border border-surface-700 bg-surface-800/80 p-6">
        <div className="flex items-center gap-2 mb-5">
          <h2 className="text-base font-semibold text-slate-200">Form Analysis</h2>
        </div>

        {uploadState.status === "idle" && capturedBlob && exerciseId && (
          <div className="py-2 space-y-3">
            <div className="rounded-xl border border-perf-500/20 bg-perf-500/5 px-4 py-3
                            flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-perf-500/15 border border-perf-500/30
                              flex items-center justify-center shrink-0">
                <svg className="w-4 h-4 text-perf-400" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round"
                        d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-perf-300">Clip captured</p>
                <p className="text-xs text-slate-500">
                  {(capturedBlob.size / 1024).toFixed(0)} KB -- MediaPipe Heavy biomechanics pipeline
                </p>
              </div>
            </div>
            <button
              onClick={onUpload}
              className="w-full px-6 py-4 rounded-xl bg-perf-500 text-white text-sm font-bold
                         hover:bg-perf-400 transition-all duration-200
                         shadow-lg shadow-perf-500/25 hover:shadow-perf-500/40
                         flex items-center justify-center gap-2.5"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
              </svg>
              Analyse my form
            </button>
            <p className="text-[10px] text-slate-600 text-center">
              Per-rep biomechanics -- takes 30-60 s for a 15 s clip
            </p>
          </div>
        )}

        {uploadState.status === "uploading" && <AnalysisProgress />}

        {uploadState.status === "error" && (
          <div className="rounded-xl border border-rose-700/50 bg-rose-900/20 px-4 py-4">
            <p className="text-sm font-semibold text-rose-300 mb-1">Analysis failed</p>
            <p className="text-xs text-rose-400/80 leading-relaxed">
              {uploadState.error ?? "Something went wrong. Re-record with full body clearly visible."}
            </p>
          </div>
        )}

        {uploadState.status === "idle" && !capturedBlob && (
          <div className="text-center py-8 text-slate-600">
            <p className="text-sm">Record a clip above to run form analysis.</p>
          </div>
        )}
      </div>
    );
  }

  // Results view
  const repCount = result.feature_vectors.length;
  const validReps = result.feature_vectors.filter((r) => r.rep_status === "valid").length;
  const degradedReps = result.feature_vectors.filter((r) => r.rep_status === "degraded").length;
  const droppedReps = repCount - validReps - degradedReps;

  return (
    <div className="rounded-2xl border border-surface-700 bg-surface-800/80 overflow-hidden">

      {/* Gradient header */}
      <div className="relative px-6 pt-5 pb-5 border-b border-surface-700/50 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-perf-500/10 via-perf-500/3 to-transparent pointer-events-none" />
        <div className="relative flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <h2 className="text-base font-bold text-slate-200">Form Analysis</h2>
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-900/30
                             text-emerald-400 border border-emerald-700/40 text-[10px] px-2 py-0.5 font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              Verified
            </span>
          </div>
          <span className="text-[10px] text-slate-600 font-mono">{result.fps.toFixed(1)} fps</span>
        </div>
      </div>

      {/* Body: score card + rep list + footnotes */}
      <div className="p-6 space-y-5">

      {/* Score card -- three stat boxes */}
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded-xl bg-surface-900/60 border border-surface-700/40 px-3 py-3 text-center">
          <p className="text-[9px] uppercase tracking-widest text-slate-500 mb-1.5">Detected</p>
          <p className="text-3xl font-black text-slate-100 font-mono leading-none">{repCount}</p>
          <p className="text-[9px] text-slate-600 mt-1">reps</p>
        </div>
        <div className="rounded-xl bg-emerald-900/20 border border-emerald-700/30 px-3 py-3 text-center">
          <p className="text-[9px] uppercase tracking-widest text-emerald-600 mb-1.5">Full reps</p>
          <p className="text-3xl font-black text-emerald-400 font-mono leading-none">{validReps}</p>
          <p className="text-[9px] text-emerald-700 mt-1">valid</p>
        </div>
        <div className="rounded-xl bg-surface-900/60 border border-surface-700/40 px-3 py-3 text-center">
          <p className="text-[9px] uppercase tracking-widest text-slate-500 mb-1.5">Partial</p>
          <p className={`text-3xl font-black font-mono leading-none ${degradedReps > 0 ? "text-amber-400" : "text-slate-600"}`}>
            {degradedReps}
          </p>
          <p className="text-[9px] text-slate-600 mt-1">
            {droppedReps > 0 ? `${droppedReps} dropped` : "reps"}
          </p>
        </div>
      </div>

      {/* No reps detected */}
      {repCount === 0 && (
        <div className="rounded-xl border border-slate-700/50 bg-surface-900/40 px-4 py-5 text-center">
          <p className="text-sm font-medium text-slate-400 mb-1">No reps detected</p>
          <p className="text-xs text-slate-600 leading-relaxed">
            Try a longer clip with the full movement clearly visible from the side.
          </p>
        </div>
      )}

      {/* Rep cards */}
      {repCount > 0 && (
        <div className="space-y-2">
          {result.feature_vectors.map((rep, i) => (
            <RepCard key={rep.rep_index} rep={rep} index={i} />
          ))}
        </div>
      )}

      {/* Live Counter Check (parity probe) */}
      {result.parity_probe !== null && (
        <ParityProbePanel probe={result.parity_probe} />
      )}

      {/* Calibration footnote */}
      <div className="rounded-xl border border-surface-700/40 bg-surface-900/30 px-4 py-3">
        <p className="text-xs text-slate-500 leading-relaxed">
          {result.calibration.evidence_status === "cited"
            ? `Reference ranges for ${result.calibration.exercise_id.replace(/_/g, " ")} sourced from published biomechanics literature (NSCA, ACSM).`
            : `Reference ranges for ${result.calibration.exercise_id.replace(/_/g, " ")} not yet available -- raw values shown without population comparison.`}
        </p>
      </div>

      {/* Provenance (collapsed) */}
      <details className="group">
        <summary className="text-xs text-slate-600 cursor-pointer hover:text-slate-400
                            transition-colors select-none list-none flex items-center gap-1.5">
          <span className="group-open:rotate-90 transition-transform inline-block">&#9658;</span>
          Technical details
        </summary>
        <div className="mt-2 rounded-xl bg-surface-900/50 border border-surface-700/40
                        p-3 font-mono text-xs text-slate-500 space-y-1.5">
          <div className="flex gap-3">
            <span className="text-slate-400 w-32 shrink-0">Pose model</span>
            <span className="text-slate-300 truncate">{result.provenance.model}</span>
          </div>
          <div className="flex gap-3">
            <span className="text-slate-400 w-32 shrink-0">Exercise manifest</span>
            <span>{result.provenance.exercise_manifest_sha.slice(0, 16)}&hellip;</span>
          </div>
          <div className="flex gap-3">
            <span className="text-slate-400 w-32 shrink-0">Calibration</span>
            <span>{result.provenance.calibration_manifest_sha.slice(0, 16)}&hellip;</span>
          </div>
          {result.provenance.git_commit_sha && (
            <div className="flex gap-3">
              <span className="text-slate-400 w-32 shrink-0">Git commit</span>
              <span>{result.provenance.git_commit_sha.slice(0, 12)}</span>
            </div>
          )}
          <div className="flex gap-3">
            <span className="text-slate-400 w-32 shrink-0">Schema</span>
            <span>{result.schema_version}</span>
          </div>
        </div>
      </details>

      </div>{/* /body */}
    </div>
  );
}
