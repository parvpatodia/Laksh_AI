"use client";

/**
 * CanonicalReport: displays the full v1 canonical_backend response envelope.
 *
 * Shows per-rep feature vectors with status chips, calibration fields,
 * and provenance hash -- everything a judge needs to audit the result.
 * Parity probe block is rendered in Day 8.
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
// Sub-components
// ---------------------------------------------------------------------------

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

function CalibStatusChip({ status }: { status: CalibrationField["status"] }) {
  const cls =
    status === "within_reference"
      ? "chip-valid"
      : status === "outside_reference"
        ? "chip-degraded"
        : "chip-unknown";
  return (
    <span className={`${cls} text-xs px-1.5 py-0.5 rounded font-mono`}>
      {status.replace(/_/g, " ")}
    </span>
  );
}

// Wire-format -> user-friendly label + value formatter.
// Unknown wire keys fall through to the raw key + raw value (so adding
// a backend field doesn't break the UI).
const FIELD_DISPLAY: Record<
  string,
  { label: string; format?: (v: number, unit: string) => string }
> = {
  rep_duration_s: { label: "Rep duration", format: (v) => `${v.toFixed(2)} s` },
  eccentric_duration_s: { label: "Lowering time", format: (v) => `${v.toFixed(2)} s` },
  concentric_duration_s: { label: "Lifting time", format: (v) => `${v.toFixed(2)} s` },
  tempo_ratio_ecc_over_con: { label: "Tempo (lower / lift)", format: (v) => `${v.toFixed(2)}×` },
  signal_amplitude: { label: "Range of motion", format: (v, u) => `${v.toFixed(1)}${u === "deg" ? "°" : ""}` },
  primary_joints_min_visibility: { label: "Tracking quality", format: (v) => `${(v * 100).toFixed(0)}%` },
  primary_joints_missing_frac: { label: "Frames untracked", format: (v) => `${(v * 100).toFixed(0)}%` },
};

function FieldRow({ label, f }: { label: string; f: FieldValue }) {
  const meta = FIELD_DISPLAY[label];
  const display = meta?.label ?? label;
  const value =
    f.value !== null
      ? meta?.format
        ? meta.format(f.value, f.unit)
        : `${f.value} ${f.unit}`
      : "—";
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-surface-700/40 last:border-0 gap-2">
      <span className="text-xs text-slate-300" title={label}>{display}</span>
      <div className="flex items-center gap-2 shrink-0">
        <span className="text-sm text-slate-200 font-mono tabular-nums">{value}</span>
        <StatusChip status={f.status} />
      </div>
    </div>
  );
}

function RepCard({ rep }: { rep: RepVector }) {
  const statusColor =
    rep.rep_status === "valid"
      ? "border-emerald-700/50"
      : rep.rep_status === "degraded"
        ? "border-amber-700/50"
        : "border-slate-700";

  return (
    <div className={`rounded-lg border ${statusColor} bg-surface-900/50 p-4`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-semibold text-slate-300">
          Rep {rep.rep_index + 1}
        </span>
        <div className="flex items-center gap-2 text-xs text-slate-600 font-mono">
          <span>
            f{rep.start_frame}–f{rep.end_frame}
          </span>
          <StatusChip status={rep.rep_status} />
        </div>
      </div>
      {Object.entries(rep.features).map(([key, fv]) => (
        <FieldRow key={key} label={key} f={fv} />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Upload progress indicator with elapsed time
// ---------------------------------------------------------------------------

/** Gym analysis takes ~10-20s. Elapsed counter turns "is it frozen?" into data. */
function GymUploadingProgress() {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, []);
  const stages = [
    { at: 0,  msg: "Running MediaPipe heavy + gym pipeline…" },
    { at: 8,  msg: "MediaPipe analysing pose landmarks…" },
    { at: 15, msg: "Computing per-rep features…" },
  ];
  const current = [...stages].reverse().find((s) => elapsed >= s.at) ?? stages[0];
  return (
    <div className="text-center py-6">
      <div className="inline-block w-8 h-8 rounded-full border-2 border-brand-500 border-t-transparent animate-spin mb-3" />
      <p className="text-sm text-slate-400">{current.msg}</p>
      <p className="text-xs text-slate-600 mt-1">
        {elapsed}s elapsed · ~10-20 s total for a 5-second clip
      </p>
    </div>
  );
}

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

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function CanonicalReport({
  result,
  uploadState,
  capturedBlob,
  exerciseId,
  onUpload,
}: Props) {
  // Upload CTA: show when we have a blob but no result yet.
  if (!result) {
    return (
      <div className="rounded-xl border border-surface-700 bg-surface-800 p-5">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
          Canonical result
          <span className="chip-valid text-xs px-1.5 py-0.5 rounded font-normal">
            canonical_backend
          </span>
        </h2>

        {uploadState.status === "idle" && capturedBlob && exerciseId && (
          <div className="text-center py-4">
            <p className="text-sm text-slate-400 mb-4">
              Clip ready ({(capturedBlob.size / 1024).toFixed(0)} KB).
              Upload to run full MediaPipe + pipeline analysis.
            </p>
            <button
              onClick={onUpload}
              className="px-6 py-2.5 rounded-lg bg-brand-500 text-white text-sm font-medium
                         hover:bg-brand-600 transition-colors"
            >
              Analyse clip
            </button>
          </div>
        )}

        {uploadState.status === "uploading" && <GymUploadingProgress />}

        {uploadState.status === "error" && (
          <div className="rounded-lg border border-rose-700/50 bg-rose-900/20 px-4 py-3 text-sm text-rose-300">
            {uploadState.error ?? "Upload failed."}
          </div>
        )}

        {uploadState.status === "idle" && !capturedBlob && (
          <p className="text-xs text-slate-600">
            Record a clip first, then upload for canonical analysis.
          </p>
        )}
      </div>
    );
  }

  // Result view.
  const repCount = result.feature_vectors.length;
  const validReps = result.feature_vectors.filter((r) => r.rep_status === "valid").length;

  return (
    <div className="rounded-xl border border-surface-700 bg-surface-800 p-5">
      <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
        Canonical result
        <span className="chip-valid text-xs px-1.5 py-0.5 rounded font-normal">
          canonical_backend
        </span>
      </h2>

      {/* Summary row */}
      <div className="grid grid-cols-3 gap-3 mb-5">
        <div className="rounded-lg bg-surface-900/60 px-3 py-2 text-center">
          <p className="text-xs text-slate-500 mb-0.5">Reps</p>
          <p className="text-2xl font-bold text-slate-100 font-mono">{repCount}</p>
        </div>
        <div className="rounded-lg bg-surface-900/60 px-3 py-2 text-center">
          <p className="text-xs text-slate-500 mb-0.5">Valid</p>
          <p className="text-2xl font-bold text-emerald-400 font-mono">{validReps}</p>
        </div>
        <div className="rounded-lg bg-surface-900/60 px-3 py-2 text-center">
          <p className="text-xs text-slate-500 mb-0.5">FPS</p>
          <p className="text-2xl font-bold text-slate-100 font-mono">
            {result.fps.toFixed(1)}
          </p>
        </div>
      </div>

      {/* Rep cards */}
      {repCount === 0 ? (
        <p className="text-sm text-slate-500 text-center py-4">
          No reps detected. Try a longer or clearer clip.
        </p>
      ) : (
        <div className="space-y-3 mb-5">
          {result.feature_vectors.map((rep) => (
            <RepCard key={rep.rep_index} rep={rep} />
          ))}
        </div>
      )}

      {/* Parity probe (only when realtime vectors were submitted) */}
      {result.parity_probe !== null && (
        <ParityProbePanel probe={result.parity_probe} />
      )}

      {/* Calibration notice */}
      <div className="rounded-lg border border-surface-700/60 bg-surface-900/40 px-3 py-2.5 mb-4">
        <p className="text-xs text-slate-500">
          <span className="font-mono text-slate-400">
            {result.calibration.evidence_status}
          </span>{" "}
          {result.calibration.evidence_status === "cited"
            ? `— ${result.calibration.exercise_id} metrics compared against literature-cited reference ranges (NSCA, ACSM).`
            : `— calibration not yet available for ${result.calibration.exercise_id}. Reference ranges require collected cohort data.`}
        </p>
      </div>

      {/* Provenance */}
      <details className="group">
        <summary className="text-xs text-slate-600 cursor-pointer hover:text-slate-400 transition-colors select-none">
          Provenance ▸
        </summary>
        <div className="mt-2 rounded-lg bg-surface-900/60 p-3 font-mono text-xs text-slate-500 space-y-1">
          <p>
            <span className="text-slate-400">model</span>{" "}
            {result.provenance.model}
          </p>
          <p>
            <span className="text-slate-400">exercise_manifest</span>{" "}
            {result.provenance.exercise_manifest_sha.slice(0, 16)}&hellip;
          </p>
          <p>
            <span className="text-slate-400">calibration_manifest</span>{" "}
            {result.provenance.calibration_manifest_sha.slice(0, 16)}&hellip;
          </p>
          {result.provenance.git_commit_sha && (
            <p>
              <span className="text-slate-400">git</span>{" "}
              {result.provenance.git_commit_sha.slice(0, 12)}
            </p>
          )}
          <p>
            <span className="text-slate-400">schema</span>{" "}
            {result.schema_version}
          </p>
        </div>
      </details>
    </div>
  );
}
