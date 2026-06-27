"use client";

/**
 * BasketballReport - renders the legacy /analyze-video response.
 *
 * Broadcast aesthetic: large stat numbers, color-coded performance bands,
 * horizontal fill bars. Inspired by ESPN/NBA box-score layout.
 *
 * Honesty contract (unchanged):
 *   - oracle_caveat surfaces inline.
 *   - confidence + analysis_reliability_score shown as-is.
 *   - athlete_feedback labelled as Gemini-generated coaching commentary.
 *   - No values fabricated when biomech fails: null -> "--".
 */

import React, { useEffect, useState } from "react";
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
  if (n === null || n === undefined || Number.isNaN(n)) return "--";
  return `${n.toFixed(digits)}${suffix}`;
}

// ---------------------------------------------------------------------------
// Performance tier colour system
// ---------------------------------------------------------------------------

type Tier = "elite" | "good" | "developing" | "none";

type TierKey = "release_velocity" | "shot_arc" | "elbow_angle" | "knee_angle" | "reliability";

function getTier(key: TierKey, value: number | null | undefined): Tier {
  if (value === null || value === undefined || Number.isNaN(value)) return "none";
  switch (key) {
    case "release_velocity":
      return value >= 7.0 ? "elite" : value >= 5.0 ? "good" : "developing";
    case "shot_arc":
      return value >= 43 && value <= 47 ? "elite"
        : value >= 38 && value <= 52 ? "good"
          : "developing";
    case "elbow_angle":
      return value >= 80 && value <= 110 ? "elite"
        : value >= 70 && value <= 120 ? "good"
          : "developing";
    case "knee_angle":
      return value >= 120 && value <= 160 ? "elite"
        : value >= 100 && value <= 170 ? "good"
          : "developing";
    case "reliability":
      return value >= 70 ? "elite" : value >= 40 ? "good" : "developing";
  }
}

const TIER_STYLES: Record<Tier, {
  num: string;
  bar: string;
  label: string;
  badge: string;
}> = {
  elite:      { num: "text-emerald-400", bar: "bg-emerald-500", label: "Elite",      badge: "bg-emerald-900/40 text-emerald-400 border-emerald-700/50" },
  good:       { num: "text-sky-400",     bar: "bg-sky-500",     label: "Good",       badge: "bg-sky-900/40 text-sky-400 border-sky-700/50" },
  developing: { num: "text-amber-400",   bar: "bg-amber-500",   label: "Developing", badge: "bg-amber-900/30 text-amber-400 border-amber-700/40" },
  none:       { num: "text-slate-300",   bar: "bg-slate-600",   label: "",           badge: "bg-slate-800 text-slate-500 border-slate-700" },
};

// ---------------------------------------------------------------------------
// StatBox: broadcast-style large-number stat cell with performance bar
// ---------------------------------------------------------------------------

function StatBox({
  label,
  value,
  fill,
  tier,
  srcLabel,
  hint,
}: {
  label: string;
  value: string;
  fill?: number;
  tier: Tier;
  srcLabel?: string;
  hint?: string;
}) {
  const styles = TIER_STYLES[tier];
  return (
    <div className="rounded-xl bg-surface-900/70 border border-surface-700/50 px-4 py-3" title={hint}>
      <div className="flex items-center justify-between mb-1.5">
        <p className="text-[10px] text-slate-500 uppercase tracking-widest font-medium">{label}</p>
        {tier !== "none" && (
          <span className={`text-[9px] px-1.5 py-0.5 rounded font-semibold border ${styles.badge}`}>
            {styles.label}
          </span>
        )}
      </div>
      <p className={`text-2xl font-bold font-mono tabular-nums leading-none ${styles.num}`}>{value}</p>
      {fill !== undefined && (
        <div className="h-0.5 rounded-full bg-surface-700/80 mt-2 overflow-hidden">
          <div
            className={`h-full rounded-full ${styles.bar} transition-all duration-300`}
            style={{ width: `${Math.max(3, Math.min(97, fill * 100))}%` }}
          />
        </div>
      )}
      {srcLabel && <p className="text-[9px] text-slate-600 mt-1.5">{srcLabel}</p>}
      {hint && <p className="text-[10px] text-amber-400/70 mt-1 leading-tight line-clamp-2">{hint}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shot-count chip (A5 multi-shot consensus)
// ---------------------------------------------------------------------------

function ShotCountChip({ seg }: { seg: ShotSegmentation }) {
  const { n_shots_detected, n_shots_valid, n_shots_degraded } = seg;
  if (n_shots_detected === 0) return null;
  const dropped = Math.max(0, n_shots_detected - n_shots_valid - n_shots_degraded);
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-[10px] text-slate-500 uppercase tracking-widest">Shot detection</span>
      <span className="text-[10px] px-2 py-0.5 rounded-full font-mono bg-emerald-900/40 text-emerald-400 border border-emerald-700/40">
        {n_shots_valid} valid
      </span>
      {n_shots_degraded > 0 && (
        <span className="text-[10px] px-2 py-0.5 rounded-full font-mono bg-amber-900/30 text-amber-400 border border-amber-700/40">
          {n_shots_degraded} degraded
        </span>
      )}
      {dropped > 0 && (
        <span className="text-[10px] px-2 py-0.5 rounded-full font-mono bg-rose-900/30 text-rose-400 border border-rose-700/40">
          {dropped} dropped
        </span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Progress indicator with a fill bar and stage messages
// ---------------------------------------------------------------------------

function UploadingProgress() {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, []);

  const stages = [
    { at: 0,  msg: "Uploading clip to analysis server..." },
    { at: 5,  msg: "Running MediaPipe Heavy -- 33-landmark pose at 30 fps..." },
    { at: 45, msg: "Extracting release kinematics and joint angles..." },
    { at: 60, msg: "Gemini generating personalised coaching commentary..." },
  ];
  const current = [...stages].reverse().find((s) => elapsed >= s.at) ?? stages[0];
  const pct = Math.min(97, (elapsed / 75) * 100);

  return (
    <div className="py-6">
      <div className="flex items-center gap-3 mb-5">
        <div className="w-5 h-5 rounded-full border-2 border-brand-500 border-t-transparent animate-spin shrink-0" />
        <div>
          <p className="text-sm font-medium text-slate-200">{current.msg}</p>
          <p className="text-xs text-slate-500 mt-0.5">
            {elapsed}s elapsed -- 60-80s total for a 15-second clip
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
// Coaching point card
// ---------------------------------------------------------------------------

function FeedbackCard({ idx, fb }: { idx: number; fb: BasketballAthleteFeedback }) {
  const title = (fb.title as string | undefined) ?? `Coaching point ${idx + 1}`;
  const body = (fb.feedback as string | undefined) ?? "";
  const drill = (fb.drill as string | undefined) ?? "";
  const accentColors = ["border-l-brand-500", "border-l-emerald-500", "border-l-amber-500"];
  const accent = accentColors[idx % accentColors.length];

  return (
    <div className={`rounded-r-xl border-l-2 ${accent} bg-surface-900/50 pl-3 pr-4 py-3`}>
      <p className="text-sm font-semibold text-slate-200 mb-1">{title}</p>
      <p className="text-xs text-slate-400 leading-relaxed">{body}</p>
      {drill && (
        <>
          <p className="text-[10px] text-slate-600 uppercase tracking-widest mt-2.5 mb-0.5">Drill</p>
          <p className="text-xs text-slate-400 leading-relaxed">{drill}</p>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function BasketballReport({
  result,
  uploadState,
  capturedBlob,
  onUpload,
}: Props) {
  // Pre-result state
  if (!result) {
    return (
      <div className="rounded-2xl border border-surface-700 bg-surface-800 p-5">
        <div className="flex items-center gap-2 mb-4">
          <h2 className="text-base font-semibold text-slate-200">Shot Analysis</h2>
          <span className="text-[10px] px-2 py-0.5 rounded-full border border-brand-500/40 bg-brand-500/10 text-brand-400 font-medium">
            Full Analysis
          </span>
        </div>

        {uploadState.status === "idle" && capturedBlob && (
          <div className="py-2 space-y-3">
            {/* Clip info card */}
            <div className="rounded-xl border border-brand-500/20 bg-brand-500/5 px-4 py-3
                            flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-brand-500/15 border border-brand-500/30
                              flex items-center justify-center shrink-0">
                <svg className="w-4 h-4 text-brand-400" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round"
                        d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-brand-300">Clip captured</p>
                <p className="text-xs text-slate-500">
                  {(capturedBlob.size / 1024).toFixed(0)} KB -- MediaPipe Heavy + Gemini coaching
                </p>
              </div>
            </div>
            {/* CTA */}
            <button
              onClick={onUpload}
              className="w-full px-6 py-4 rounded-xl bg-brand-500 text-white text-sm font-bold
                         hover:bg-brand-400 transition-all duration-200
                         shadow-lg shadow-brand-500/30 hover:shadow-brand-500/50
                         flex items-center justify-center gap-2.5"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
              </svg>
              Run full biomechanics analysis
            </button>
            <p className="text-[10px] text-slate-600 text-center">
              33-landmark pose detection -- takes 60-80 s for a 15 s clip
            </p>
          </div>
        )}

        {uploadState.status === "uploading" && <UploadingProgress />}

        {uploadState.status === "error" && (
          <div className="rounded-xl border border-rose-700/50 bg-rose-900/15 px-4 py-4">
            <p className="text-sm font-semibold text-rose-300 mb-1">Analysis failed</p>
            <p className="text-xs text-rose-400/80 leading-relaxed">
              {uploadState.error ?? "Upload failed. Try re-recording with full body visible."}
            </p>
          </div>
        )}

        {uploadState.status === "idle" && !capturedBlob && (
          <p className="text-xs text-slate-600 py-2">
            Record a jump shot above, then run full analysis.
          </p>
        )}
      </div>
    );
  }

  // Result view
  const conf = result.confidence ?? result.analysis_reliability_score ?? null;
  const proName = (result.matched_pro?.name as string | undefined) ?? null;
  const feedback = result.athlete_feedback ?? [];
  const biomechFailed =
    result.release_velocity_mps == null &&
    result.shot_arc_deg == null &&
    result.knee_angle == null &&
    result.elbow_angle == null;
  const ms = result.metric_status ?? {};
  const hints = result.metric_hints ?? {};

  const srcLabel = (key: string): string | undefined => {
    const s = (ms[key] as { source?: string } | undefined)?.source;
    return s && s !== "measured" ? `Source: ${s}` : undefined;
  };

  return (
    <div className="rounded-2xl border border-surface-700 bg-surface-800 overflow-hidden">

      {/* Gradient header */}
      <div className="relative px-5 pt-5 pb-4 border-b border-surface-700/50 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-brand-500/10 via-brand-500/3 to-transparent pointer-events-none" />
        <div className="relative flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <h2 className="text-base font-bold text-slate-200">Shot Analysis</h2>
            <span className="text-[10px] px-2 py-0.5 rounded-full border border-brand-500/40 bg-brand-500/10 text-brand-400 font-bold">
              Full Analysis
            </span>
          </div>
          {result.video_quality_label && (
            <span className="text-[10px] text-slate-600 font-mono">{result.video_quality_label}</span>
          )}
        </div>
      </div>

      <div className="p-5 space-y-5">

      {/* Preflight warning */}
      {result.preflight_status === "pose_detection_failed" && (result.preflight_hints ?? []).length > 0 && (
        <div className="rounded-xl border border-yellow-600/40 bg-yellow-900/10 px-3 py-3">
          <p className="text-xs font-semibold text-yellow-300 mb-1.5">Pose detection was limited</p>
          <ul className="space-y-1">
            {(result.preflight_hints ?? []).map((h, i) => (
              <li key={i} className="text-[11px] text-yellow-200/80 flex items-start gap-1.5">
                <span className="shrink-0 mt-0.5">-</span>{h}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Shot count */}
      {result.shot_segmentation && (
        <ShotCountChip seg={result.shot_segmentation as ShotSegmentation} />
      )}

      {/* Primary stat boxes */}
      {!biomechFailed && (
        <>
          <div className="grid grid-cols-2 gap-2.5">
            <StatBox
              label="Release Speed"
              value={fmtNum(result.release_velocity_mps, 1, " m/s")}
              tier={getTier("release_velocity", result.release_velocity_mps)}
              fill={
                result.release_velocity_mps != null
                  ? Math.min(1, result.release_velocity_mps / 10)
                  : undefined
              }
              srcLabel={srcLabel("release_velocity_mps")}
            />
            <StatBox
              label="Shot Arc"
              value={fmtNum(result.shot_arc_deg, 1, "\u00b0")}
              tier={getTier("shot_arc", result.shot_arc_deg)}
              fill={
                result.shot_arc_deg != null
                  ? (result.shot_arc_deg - 30) / 30
                  : undefined
              }
              srcLabel={srcLabel("shot_arc_deg")}
            />
          </div>

          <div className="grid grid-cols-2 gap-2.5">
            <StatBox
              label="Elbow Angle"
              value={fmtNum(result.elbow_angle, 1, "\u00b0")}
              tier={getTier("elbow_angle", result.elbow_angle)}
              fill={result.elbow_angle != null ? result.elbow_angle / 180 : undefined}
              srcLabel={srcLabel("elbow_angle")}
              hint={hints["elbow_angle"]}
            />
            <StatBox
              label="Knee Drive"
              value={fmtNum(result.knee_angle, 1, "\u00b0")}
              tier={getTier("knee_angle", result.knee_angle)}
              fill={result.knee_angle != null ? result.knee_angle / 180 : undefined}
              srcLabel={srcLabel("knee_angle")}
              hint={hints["knee_angle"]}
            />
          </div>

          {/* Secondary metrics row */}
          <div className="grid grid-cols-3 gap-2">
            {(
              [
                { label: "Hip Rotation", value: fmtNum(result.hip_rotation_deg, 1, "\u00b0") },
                { label: "Body Sync", value: fmtNum(result.kinetic_sync_ms, 0, " ms") },
                { label: "Balance", value: fmtNum(result.balance_index, 0) },
              ] as const
            ).map(({ label, value }) => (
              <div
                key={label}
                className="rounded-xl bg-surface-900/50 border border-surface-700/40 px-3 py-3 text-center"
              >
                <p className="text-[9px] text-slate-600 uppercase tracking-widest mb-1.5">{label}</p>
                <p className="text-lg font-bold font-mono tabular-nums text-slate-300">{value}</p>
              </div>
            ))}
          </div>

          {/* Reliability bar */}
          {conf !== null && (
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] text-slate-500 uppercase tracking-widest">Pose reliability</span>
                <span className={`text-sm font-bold font-mono ${TIER_STYLES[getTier("reliability", conf)].num}`}>
                  {conf.toFixed(0)}%
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-surface-700 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${TIER_STYLES[getTier("reliability", conf)].bar}`}
                  style={{ width: `${conf}%` }}
                />
              </div>
              {conf < 40 && (
                <p className="text-[10px] text-amber-400/80 mt-1.5">
                  Low confidence -- re-record with full body visible in good light.
                </p>
              )}
            </div>
          )}

          {/* Multi-shot footnote */}
          {(() => {
            const seg = result.shot_segmentation as ShotSegmentation | null | undefined;
            return seg && seg.n_shots_detected > 1 ? (
              <p className="text-[10px] text-slate-600 leading-relaxed">
                Metrics from best shot of {seg.n_shots_detected} detected. Per-shot breakdown is post-showcase roadmap.
              </p>
            ) : null;
          })()}
        </>
      )}

      {/* Biomech failed */}
      {biomechFailed && (
        <div className="rounded-xl border border-amber-600/40 bg-amber-900/10 px-4 py-4">
          <p className="text-sm font-semibold text-amber-200 mb-1">Biomechanics not measured</p>
          <p className="text-xs text-amber-300/80 leading-relaxed">
            Joints were occluded or the clip was too short to extract reliable landmarks.
            Re-record: full body visible, good lighting, 5+ seconds of motion.
          </p>
        </div>
      )}

      {/* Fallback (population averages) banner */}
      {!biomechFailed && result.analysis_mode === "fallback" &&
        (result.fallback_reason_codes ?? []).includes("analysis_exception") && (
        <div className="rounded-xl border border-orange-600/40 bg-orange-900/10 px-4 py-3">
          <p className="text-sm font-semibold text-orange-200 mb-1">Population averages shown</p>
          <p className="text-xs text-orange-300/80 leading-relaxed">
            Video analysis encountered an error. Values are typical amateur ranges, not measurements from your clip.
          </p>
        </div>
      )}

      {/* Oracle caveat */}
      {result.oracle_caveat && (
        <div className="rounded-xl border border-surface-700/50 bg-surface-900/40 px-3 py-2.5">
          <p className="text-xs text-slate-400 leading-relaxed">{result.oracle_caveat}</p>
        </div>
      )}

      {/* Matched pro player card */}
      {proName && (
        <div className="rounded-xl border border-surface-700/60 bg-surface-900/50 overflow-hidden">
          <div className="px-4 py-3 border-b border-surface-700/40">
            <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Closest NBA style match</p>
            <div className="flex items-baseline gap-3 justify-between">
              <p className="text-xl font-bold text-slate-100">{proName}</p>
              {(result.matched_pro?.team as string | undefined) && (
                <p className="text-xs text-slate-500 shrink-0">{result.matched_pro!.team as string}</p>
              )}
            </div>
          </div>
          {result.witty_catchphrase && (
            <p className="px-4 py-2.5 text-xs text-slate-400 italic">
              &ldquo;{result.witty_catchphrase}&rdquo;
            </p>
          )}
          <p className="px-4 pb-3 text-[9px] text-slate-600 leading-relaxed">
            Style match estimated from playing profile (position, shooting %, usage) --
            not motion-capture of that player&apos;s shooting form.
          </p>
        </div>
      )}

      {/* Coaching points */}
      {feedback.length > 0 && (
        <div>
          <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-3">Coaching points</p>
          <div className="space-y-2.5">
            {feedback.slice(0, 3).map((fb, i) => (
              <FeedbackCard key={i} idx={i} fb={fb} />
            ))}
          </div>
          <p className="text-[10px] text-slate-600 mt-2">
            Generated by Gemini from measured kinematics above.
          </p>
        </div>
      )}

      {/* Validation warnings */}
      {result.validation_warnings && result.validation_warnings.length > 0 && (
        <details className="group">
          <summary className="text-[11px] text-slate-600 cursor-pointer hover:text-slate-400 list-none flex items-center gap-1.5">
            <span className="group-open:rotate-90 transition-transform inline-block text-xs">&#9658;</span>
            Validation notes ({result.validation_warnings.length})
          </summary>
          <ul className="mt-2 space-y-0.5">
            {result.validation_warnings.map((w, i) => (
              <li key={i} className="text-xs text-slate-500 flex items-start gap-1.5">
                <span className="shrink-0 mt-0.5">-</span>{w}
              </li>
            ))}
          </ul>
        </details>
      )}

      </div>{/* /body */}
    </div>
  );
}
