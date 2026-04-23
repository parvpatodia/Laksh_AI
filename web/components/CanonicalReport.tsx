"use client";

/**
 * CanonicalReport: displays the full v1 form analysis response envelope.
 *
 * Shows per-rep metrics with human-readable labels and descriptions,
 * a summary scorecard, calibration footnote, and provenance block.
 * Parity probe (Live Counter Check) rendered when realtime vectors were submitted.
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
// Metric definitions: wire-key -> display label + plain-English description
// + value formatter. Unknown wire keys fall through to raw key + raw value.
// ---------------------------------------------------------------------------

interface MetricMeta {
  label: string;
  description: string;
  format?: (v: number, unit: string) => string;
  /** Optional interpretation hint shown alongside the value. */
  interpret?: (v: number) => string | null;
}

const METRIC_META: Record<string, MetricMeta> = {
  rep_duration_s: {
    label: "Total Rep Time",
    description: "How long the complete rep took from start to finish.",
    format: (v) => `${v.toFixed(2)} s`,
    interpret: (v) => v < 1.0 ? "Fast — consider slowing down" : v > 4.0 ? "Controlled pace" : null,
  },
  eccentric_duration_s: {
    label: "Lowering Phase",
    description: "Time spent lowering the weight. Slower is generally better — it builds more muscle.",
    format: (v) => `${v.toFixed(2)} s`,
    interpret: (v) => v < 0.8 ? "Try to slow the lowering down" : null,
  },
  concentric_duration_s: {
    label: "Lifting Phase",
    description: "Time spent lifting the weight through the working part of the rep.",
    format: (v) => `${v.toFixed(2)} s`,
  },
  tempo_ratio_ecc_over_con: {
    label: "Tempo Ratio",
    description: "Lowering time divided by lifting time. Above 1.0 means you lowered slower than you lifted — the recommended pattern.",
    format: (v) => `${v.toFixed(2)}×`,
    interpret: (v) => v >= 1.0 ? "Good — controlled lowering" : "Lowering faster than lifting",
  },
  signal_amplitude: {
    label: "Range of Motion",
    description: "Total joint angle swept through the rep. Larger values mean a fuller range of motion.",
    format: (v, u) => `${v.toFixed(1)}${u === "deg" ? "°" : " " + u}`,
  },
  primary_joints_min_visibility: {
    label: "Pose Confidence",
    description: "How reliably the key joints were tracked across the clip. Above 70% is solid; below 50% means some measurements may be imprecise.",
    format: (v) => `${(v * 100).toFixed(0)}%`,
    interpret: (v) => v < 0.5 ? "Low — measurements may be less precise" : v >= 0.8 ? "Good tracking" : null,
  },
  primary_joints_missing_frac: {
    label: "Tracking Gaps",
    description: "Share of frames where a key joint wasn't detected. Lower is better; above 20% can reduce accuracy.",
    format: (v) => `${(v * 100).toFixed(0)}%`,
    interpret: (v) => v > 0.20 ? "High — consider re-recording with better lighting" : null,
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
      : "—";

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
  const [open, setOpen] = useState(index === 0); // first rep expanded by default

  const borderColor =
    rep.rep_status === "valid"
      ? "border-emerald-700/40"
      : rep.rep_status === "degraded"
        ? "border-amber-700/40"
        : "border-slate-700/50";

  const headerBg =
    rep.rep_status === "valid"
      ? "bg-emerald-900/10"
      : rep.rep_status === "degraded"
        ? "bg-amber-900/10"
        : "bg-surface-900/30";

  return (
    <div className={`rounded-xl border ${borderColor} overflow-hidden`}>
      <button
        onClick={() => setOpen((v) => !v)}
        className={`w-full flex items-center justify-between px-4 py-3 ${headerBg}
                    hover:bg-white/5 transition-colors text-left`}
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-slate-200">Rep {rep.rep_index + 1}</span>
          <RepStatusBadge status={rep.rep_status} />
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-600 font-mono">
            frames {rep.start_frame}–{rep.end_frame}
          </span>
          <span className="text-slate-600 text-xs">{open ? "▲" : "▼"}</span>
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
    { at: 0,  msg: "Uploading clip…" },
    { at: 3,  msg: "Extracting pose with MediaPipe…" },
    { at: 12, msg: "Measuring per-rep biomechanics…" },
    { at: 22, msg: "Almost done — computing rep scores…" },
  ];
  const current = [...stages].reverse().find((s) => elapsed >= s.at) ?? stages[0];
  return (
    <div className="text-center py-8">
      <div className="inline-block w-9 h-9 rounded-full border-2 border-brand-500
                      border-t-transparent animate-spin mb-4" />
      <p className="text-sm font-medium text-slate-300">{current.msg}</p>
      <p className="text-xs text-slate-600 mt-1">
        {elapsed}s elapsed · typically 30–60 s for a 6-second clip
      </p>
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
  // Pre-result state (no clip yet, uploading, or error).
  if (!result) {
    return (
      <div className="rounded-2xl border border-surface-700 bg-surface-800/80 p-6">
        <div className="flex items-center gap-2 mb-5">
          <h2 className="text-base font-semibold text-slate-200">Form Analysis</h2>
        </div>

        {uploadState.status === "idle" && capturedBlob && exerciseId && (
          <div className="text-center py-6">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-full
                            bg-brand-500/10 border border-brand-500/30 mb-4">
              <span className="text-2xl" aria-hidden>🎬</span>
            </div>
            <p className="text-sm font-medium text-slate-200 mb-1">Clip ready</p>
            <p className="text-xs text-slate-500 mb-5">
              {(capturedBlob.size / 1024).toFixed(0)} KB · runs MediaPipe + biomechanics pipeline
            </p>
            <button
              onClick={onUpload}
              className="px-6 py-2.5 rounded-lg bg-brand-500 text-white text-sm font-semibold
                         hover:bg-brand-600 transition-colors shadow-lg shadow-brand-500/20"
            >
              Analyse my form
            </button>
          </div>
        )}

        {uploadState.status === "uploading" && <AnalysisProgress />}

        {uploadState.status === "error" && (
          <div className="rounded-xl border border-rose-700/50 bg-rose-900/20 px-4 py-4">
            <p className="text-sm font-medium text-rose-300 mb-1">Analysis failed</p>
            <p className="text-xs text-rose-400/80 leading-relaxed">
              {uploadState.error ?? "Something went wrong. Try re-recording with your full body clearly visible."}
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

  // ---- Results view -------------------------------------------------------
  const repCount = result.feature_vectors.length;
  const validReps = result.feature_vectors.filter((r) => r.rep_status === "valid").length;
  const degradedReps = result.feature_vectors.filter((r) => r.rep_status === "degraded").length;

  return (
    <div className="rounded-2xl border border-surface-700 bg-surface-800/80 p-6 space-y-5">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <h2 className="text-base font-semibold text-slate-200">Form Analysis</h2>
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-900/30
                           text-emerald-400 border border-emerald-700/40 text-xs px-2 py-0.5 font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            Verified
          </span>
        </div>
        <span className="text-xs text-slate-600 font-mono">{result.fps.toFixed(1)} fps</span>
      </div>

      {/* Summary scorecard */}
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded-xl bg-surface-900/60 border border-surface-700/40 px-3 py-3 text-center">
          <p className="text-[10px] uppercase tracking-wide text-slate-500 mb-1">Reps detected</p>
          <p className="text-2xl font-bold text-slate-100 font-mono leading-none">{repCount}</p>
        </div>
        <div className="rounded-xl bg-emerald-900/20 border border-emerald-700/30 px-3 py-3 text-center">
          <p className="text-[10px] uppercase tracking-wide text-emerald-600 mb-1">Full reps</p>
          <p className="text-2xl font-bold text-emerald-400 font-mono leading-none">{validReps}</p>
        </div>
        <div className="rounded-xl bg-surface-900/60 border border-surface-700/40 px-3 py-3 text-center">
          <p className="text-[10px] uppercase tracking-wide text-slate-500 mb-1">Partial</p>
          <p className="text-2xl font-bold text-amber-400 font-mono leading-none">{degradedReps}</p>
        </div>
      </div>

      {/* No reps detected */}
      {repCount === 0 && (
        <div className="rounded-xl border border-slate-700/50 bg-surface-900/40 px-4 py-5 text-center">
          <p className="text-sm font-medium text-slate-400 mb-1">No reps detected</p>
          <p className="text-xs text-slate-600 leading-relaxed">
            Try a longer clip with your full movement clearly visible from the side.
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
        <div className="flex items-start gap-2">
          <span className="text-slate-600 text-sm mt-0.5" aria-hidden>📋</span>
          <p className="text-xs text-slate-500 leading-relaxed">
            {result.calibration.evidence_status === "cited"
              ? `Reference ranges for ${result.calibration.exercise_id.replace(/_/g, " ")} are sourced from published biomechanics literature (NSCA, ACSM).`
              : `Reference ranges for ${result.calibration.exercise_id.replace(/_/g, " ")} are not yet available — measurements shown are raw values without population comparison.`}
          </p>
        </div>
      </div>

      {/* Provenance (collapsed by default — for technical users / judges) */}
      <details className="group">
        <summary className="text-xs text-slate-600 cursor-pointer hover:text-slate-400
                            transition-colors select-none list-none flex items-center gap-1.5">
          <span className="group-open:rotate-90 transition-transform inline-block">▶</span>
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
    </div>
  );
}
