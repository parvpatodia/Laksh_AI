"use client";

/**
 * BasketballReport — renders the legacy /analyze-video response.
 *
 * Why a dedicated component?
 *   The basketball pipeline (Gemini scout report + ChromaDB pro-match
 *   + biomech) predates the v1 envelope and returns a different shape.
 *   Wiring it through the v1 envelope is post-showcase work; for the
 *   demo we present its richer payload (matched NBA pro, 3 athlete
 *   feedback bullets) directly so basketball gets a real canonical
 *   analysis layer instead of just realtime ghosts.
 *
 * Honesty contract
 *   - oracle_caveat (when present) is surfaced inline.
 *   - confidence + analysis_reliability_score are shown as-is.
 *   - athlete_feedback (LLM-generated) is clearly labelled "AI scout"
 *     so judges can distinguish it from the deterministic biomech.
 */

import React from "react";
import type {
  BasketballAnalyzeResponse,
  BasketballAthleteFeedback,
  ShotSegmentation,
} from "@/lib/api";

interface UploadState {
  status: "idle" | "uploading" | "done" | "error";
  error?: string;
}

interface Props {
  result: BasketballAnalyzeResponse | null;
  uploadState: UploadState;
  capturedBlob: Blob | null;
  onUpload: () => void;
}

function fmtNum(n: number | null | undefined, digits = 1, suffix = ""): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${n.toFixed(digits)}${suffix}`;
}

/**
 * Source quality tier -> colour.
 * measured/predicted = green; interpolated = amber; estimated/constant = orange.
 * Judges can hover to read the full label.
 */
function srcDot(src: string | undefined): React.ReactNode {
  if (!src) return null;
  const cls =
    src === "measured"   ? "bg-emerald-400 title-measured"
    : src === "predicted"  ? "bg-sky-400 title-predicted"
    : src === "interpolated" ? "bg-amber-400"
    : "bg-orange-400"; // estimated | constant
  return (
    <span
      className={`inline-block w-1.5 h-1.5 rounded-full ml-1 flex-shrink-0 ${cls}`}
      title={`Source: ${src}`}
    />
  );
}

function MetricCell({
  label,
  value,
  sourceSrc,
  uncertainty,
  uncertaintyUnit = "",
  hint,
}: {
  label: string;
  value: string;
  sourceSrc?: string;
  uncertainty?: number | null;
  uncertaintyUnit?: string;
  hint?: string;
}) {
  return (
    <div className={`rounded-lg bg-surface-900/60 px-3 py-2 ${hint ? "border border-amber-700/30" : ""}`} title={hint}>
      <p className="text-[11px] text-slate-500 uppercase tracking-wider mb-0.5 flex items-center gap-0.5">
        {label}
        {srcDot(sourceSrc)}
      </p>
      <p className="text-base font-mono tabular-nums text-slate-100">{value}</p>
      {uncertainty != null && uncertainty > 0 && (
        <p className="text-[10px] text-slate-600 mt-0.5 font-mono">
          ±{uncertainty.toFixed(uncertainty < 5 ? 1 : 0)}{uncertaintyUnit}
        </p>
      )}
      {hint && (
        <p className="text-[10px] text-amber-400/80 mt-0.5 leading-tight line-clamp-2">{hint}</p>
      )}
    </div>
  );
}

/**
 * A5: Shot-count chip from multi-signal consensus segmentation.
 *
 * Honesty contract: shows raw counts from the segmenter.  Does NOT claim
 * biomech metrics are per-shot medians — the top-line numbers come from
 * the single dominant-shot detection and are labelled accordingly.
 */
function ShotCountChip({ seg }: { seg: ShotSegmentation }) {
  const { n_shots_detected, n_shots_valid, n_shots_degraded } = seg;
  if (n_shots_detected === 0) return null;
  // Guard against inconsistent counts (dropped can't be negative).
  const dropped = Math.max(0, n_shots_detected - n_shots_valid - n_shots_degraded);
  return (
    <div className="flex items-center gap-2 mb-3">
      <span className="text-[11px] text-slate-500 uppercase tracking-wider">Shots detected</span>
      <span className="chip-valid text-[11px] px-1.5 py-0.5 rounded font-mono">
        {n_shots_valid} valid
      </span>
      {n_shots_degraded > 0 && (
        <span className="chip-preview text-[11px] px-1.5 py-0.5 rounded font-mono">
          {n_shots_degraded} degraded
        </span>
      )}
      {dropped > 0 && (
        <span className="bg-rose-900/40 text-rose-300 text-[11px] px-1.5 py-0.5 rounded font-mono">
          {dropped} dropped
        </span>
      )}
    </div>
  );
}

function FeedbackCard({ idx, fb }: { idx: number; fb: BasketballAthleteFeedback }) {
  const title = (fb.title as string | undefined) ?? `Coaching point ${idx + 1}`;
  const body = (fb.feedback as string | undefined) ?? "";
  const drill = (fb.drill as string | undefined) ?? "";
  return (
    <div className="rounded-lg border border-amber-700/40 bg-surface-900/40 p-3">
      <p className="text-sm font-medium text-amber-200 mb-1">{title}</p>
      <p className="text-xs text-slate-300 leading-relaxed mb-2">{body}</p>
      {drill && (
        <div className="border-t border-surface-700/60 pt-2 mt-1">
          <p className="text-[11px] text-slate-500 uppercase tracking-wider mb-0.5">Drill</p>
          <p className="text-xs text-slate-300 leading-relaxed">{drill}</p>
        </div>
      )}
    </div>
  );
}

export default function BasketballReport({
  result,
  uploadState,
  capturedBlob,
  onUpload,
}: Props) {
  // Upload CTA / progress / error state
  if (!result) {
    return (
      <div className="rounded-xl border border-surface-700 bg-surface-800 p-5">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
          Canonical result
          <span className="chip-valid text-xs px-1.5 py-0.5 rounded font-normal">
            canonical_backend
          </span>
        </h2>

        {uploadState.status === "idle" && capturedBlob && (
          <div className="text-center py-4">
            <p className="text-sm text-slate-400 mb-4">
              Clip ready ({(capturedBlob.size / 1024).toFixed(0)} KB).
              Upload to run MediaPipe biomech + AI scout report (~25–40 s).
            </p>
            <button
              onClick={onUpload}
              className="px-6 py-2.5 rounded-lg bg-brand-500 text-white text-sm font-medium hover:bg-brand-600 transition-colors"
            >
              Analyse shot
            </button>
          </div>
        )}

        {uploadState.status === "uploading" && (
          <div className="text-center py-6">
            <div className="inline-block w-8 h-8 rounded-full border-2 border-brand-500 border-t-transparent animate-spin mb-3" />
            <p className="text-sm text-slate-400">
              Running MediaPipe + Gemini scout pipeline…
            </p>
            <p className="text-xs text-slate-600 mt-1">~25–40 s for a 5-second clip</p>
          </div>
        )}

        {uploadState.status === "error" && (
          <div className="rounded-lg border border-rose-700/50 bg-rose-900/20 px-4 py-3 text-sm text-rose-300">
            {uploadState.error ?? "Upload failed."}
          </div>
        )}

        {uploadState.status === "idle" && !capturedBlob && (
          <p className="text-xs text-slate-600">
            Record a jump shot first, then upload for canonical analysis.
          </p>
        )}
      </div>
    );
  }

  const conf = result.confidence ?? result.analysis_reliability_score ?? null;
  const proName =
    (result.matched_pro?.name as string | undefined) ??
    (typeof result.matched_pro === "object" ? null : null);
  const feedback = result.athlete_feedback ?? [];
  const reliabilityCls =
    conf === null ? "text-slate-300"
      : conf >= 70 ? "text-emerald-300"
        : conf >= 40 ? "text-amber-300"
          : "text-rose-300";
  // Helper: extract source label from metric_status for a given metric key.
  const ms = result.metric_status ?? {};
  const src = (key: string) => (ms[key] as { source?: string } | undefined)?.source;
  const hints = result.metric_hints ?? {};

  return (
    <div className="rounded-xl border border-surface-700 bg-surface-800 p-5">
      <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
        Canonical result
        <span className="chip-valid text-xs px-1.5 py-0.5 rounded font-normal">
          canonical_backend
        </span>
        {result.video_quality_label && (
          <span className="ml-auto text-[11px] text-slate-500 font-mono normal-case tracking-normal">
            video quality: {result.video_quality_label}
          </span>
        )}
      </h2>

      {/* A4: Preflight warning — pose detection failed, give actionable hints */}
      {result.preflight_status === "pose_detection_failed" && (result.preflight_hints ?? []).length > 0 && (
        <div className="rounded-lg border border-yellow-600/50 bg-yellow-900/15 px-3 py-2 mb-4">
          <p className="text-xs font-medium text-yellow-300 mb-1">Pose detection was limited — metrics are estimates</p>
          <ul className="list-disc list-inside space-y-0.5">
            {(result.preflight_hints ?? []).map((h, i) => (
              <li key={i} className="text-[11px] text-yellow-200/80">{h}</li>
            ))}
          </ul>
        </div>
      )}

      {/* A5: Multi-shot consensus chip — honest label, no invented median */}
      {result.shot_segmentation && (
        <ShotCountChip seg={result.shot_segmentation as ShotSegmentation} />
      )}

      {/* Top-line numbers — from dominant shot detection.
          Coloured dot = source quality: green=measured, blue=predicted, amber=interpolated, orange=estimated/constant.
          ±uncertainty shown below each value when source < measured. */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
        <MetricCell label="Reliability" value={conf !== null ? `${conf.toFixed(0)}%` : "—"} />
        <MetricCell
          label="Release vel."
          value={fmtNum(result.release_velocity_mps, 2, " m/s")}
          sourceSrc={src("release_velocity_mps")}
        />
        <MetricCell
          label="Shot arc"
          value={fmtNum(result.shot_arc_deg, 1, "°")}
          sourceSrc={src("shot_arc_deg")}
        />
        <MetricCell
          label="Elbow angle"
          value={fmtNum(result.elbow_angle, 1, "°")}
          sourceSrc={src("elbow_angle")}
          uncertainty={result.elbow_angle_uncertainty}
          uncertaintyUnit="°"
          hint={hints["elbow_angle"]}
        />
      </div>
      {(() => {
        const seg = result.shot_segmentation as ShotSegmentation | null | undefined;
        const multiShot = seg && seg.n_shots_detected > 1;
        return (
          <>
            <div className={`grid grid-cols-2 sm:grid-cols-4 gap-2 ${multiShot ? "mb-1" : "mb-5"}`}>
              <MetricCell
                label="Knee angle"
                value={fmtNum(result.knee_angle, 1, "°")}
                sourceSrc={src("knee_angle")}
                uncertainty={result.knee_angle_uncertainty}
                uncertaintyUnit="°"
                hint={hints["knee_angle"]}
              />
              <MetricCell
                label="Hip rotation"
                value={fmtNum(result.hip_rotation_deg, 2, "°")}
                sourceSrc={src("hip_rotation_deg")}
                uncertainty={result.hip_rotation_uncertainty}
                uncertaintyUnit="°"
                hint={hints["hip_rotation_deg"]}
              />
              <MetricCell
                label="Kinetic sync"
                value={fmtNum(result.kinetic_sync_ms, 0, " ms")}
                sourceSrc={src("kinetic_sync_ms")}
              />
              <MetricCell
                label="Balance idx"
                value={fmtNum(result.balance_index, 0, "")}
                sourceSrc={src("balance_index")}
                uncertainty={result.balance_index_uncertainty}
                hint={hints["balance_index"]}
              />
            </div>
            {multiShot && (
              <p className="text-[10px] text-slate-600 mb-4 leading-relaxed">
                Metrics from best shot of {seg!.n_shots_detected} detected. Per-shot breakdown is post-showcase roadmap.
              </p>
            )}
          </>
        );
      })()}

      {/* Reliability annotation */}
      {conf !== null && (
        <p className={`text-xs ${reliabilityCls} mb-4`}>
          Pose reliability {conf.toFixed(0)}% —
          {conf >= 70 ? " biomech numbers are trustworthy."
            : conf >= 40 ? " interpret biomech numbers with care."
              : " biomech numbers are likely unreliable; re-record with clearer framing."}
        </p>
      )}

      {/* Oracle caveat (if backend marked the pro match degraded) */}
      {result.oracle_caveat && (
        <div className="rounded-lg border border-amber-700/40 bg-amber-900/10 px-3 py-2 mb-4">
          <p className="text-xs text-amber-200">{result.oracle_caveat}</p>
        </div>
      )}

      {/* Matched pro */}
      {proName && (
        <div className="mb-5">
          <div className="rounded-lg bg-surface-900/60 px-4 py-3 flex items-center justify-between">
            <div>
              <p className="text-[11px] text-slate-500 uppercase tracking-wider mb-0.5">
                Closest NBA style match
              </p>
              <p className="text-base font-medium text-slate-100">{proName}</p>
              {(result.matched_pro?.team as string | undefined) && (
                <p className="text-xs text-slate-500 mt-0.5">
                  {result.matched_pro!.team as string}
                </p>
              )}
            </div>
            {result.witty_catchphrase && (
              <p className="text-xs text-slate-400 italic max-w-[40%] text-right">
                &ldquo;{result.witty_catchphrase}&rdquo;
              </p>
            )}
          </div>
          <p className="text-[10px] text-slate-600 mt-1 leading-relaxed">
            Style match is estimated from playing profile (position, shooting %, usage) —
            not from motion-capture of the player&apos;s actual shooting form.
          </p>
        </div>
      )}

      {/* Athlete feedback */}
      {feedback.length > 0 && (
        <div className="mb-5">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
            Coaching points
            <span className="chip-preview text-[10px] px-1.5 py-0.5 rounded font-normal normal-case">
              ai_scout
            </span>
          </h3>
          <div className="space-y-2">
            {feedback.slice(0, 3).map((fb, i) => (
              <FeedbackCard key={i} idx={i} fb={fb} />
            ))}
          </div>
          <p className="text-[11px] text-slate-600 mt-2 leading-relaxed">
            Coaching points are generated by Gemini 2.5 Flash from the deterministic
            biomech numbers above and labelled <code className="font-mono text-slate-500">ai_scout</code>
            so you can distinguish them from the measured kinematics.
          </p>
        </div>
      )}

      {/* Validation warnings */}
      {result.validation_warnings && result.validation_warnings.length > 0 && (
        <details className="mt-4 group">
          <summary className="text-xs text-slate-600 cursor-pointer hover:text-slate-400 select-none">
            Validation warnings ({result.validation_warnings.length})
          </summary>
          <ul className="mt-2 list-disc list-inside text-xs text-slate-500 space-y-0.5">
            {result.validation_warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
