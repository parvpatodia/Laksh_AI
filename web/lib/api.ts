/**
 * Typed client for the Laksh.ai v1 backend API.
 *
 * All functions read NEXT_PUBLIC_API_BASE from the environment so the same
 * build works against localhost:8000 (dev) and the Fly.io production URL.
 */

// ---------------------------------------------------------------------------
// Types (mirrors app/api/v1/schema.py)
// ---------------------------------------------------------------------------

export type FieldStatus = "valid" | "degraded" | "unknown";
export type CalibrationFieldStatus =
  | "no_reference_yet"
  | "unavailable"
  | "within_reference"
  | "outside_reference";
export type AnalysisMode = "canonical_backend" | "realtime_preview";
export type SportId = "basketball" | "gym";

export interface FieldValue {
  value: number | null;
  unit: string;
  status: FieldStatus;
  reason_codes: string[];
}

export interface RepVector {
  rep_index: number;
  start_frame: number;
  end_frame: number;
  peak_frame: number;
  rep_status: FieldStatus;
  features: Record<string, FieldValue>;
}

export interface CalibrationField {
  status: CalibrationFieldStatus;
  range: [number, number] | null;
  value: number | null;
  evidence_status: string;
  evidence_source: string | null;
}

export interface CalibrationPerRep {
  rep_index: number;
  fields: Record<string, CalibrationField>;
}

export interface CalibrationBlock {
  exercise_id: string;
  evidence_status: string;
  evidence_source: string | null;
  comparable_fields: string[];
  per_rep: CalibrationPerRep[];
}

export interface Provenance {
  git_commit_sha: string | null;
  pose_baseline_version: string;
  exercise_manifest_sha: string;
  calibration_manifest_sha: string;
  calibration_manifest_version: string;
  model: string;
}

export interface ParityProbe {
  fields_compared: string[];
  max_abs_delta: number;
  p90_abs_delta: number;
  status: "within_tolerance" | "outside_tolerance" | "insufficient_data";
}

export interface AnalyzeResponse {
  schema_version: string;
  sport_id: SportId;
  exercise_id: string;
  source: string;
  fps: number;
  n_frames: number;
  analysis_mode: AnalysisMode;
  provenance: Provenance;
  segment: unknown;
  feature_vectors: RepVector[];
  calibration: CalibrationBlock;
  parity_probe: ParityProbe | null;
}

export interface HealthResponse {
  status: string;
  v1_schema_version: string;
  provenance: Provenance;
}

export interface SportInfo {
  id: SportId;
  name: string;
  available: boolean;
  exercises: string[];
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function apiBase(): string {
  const base = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
  return base.replace(/\/$/, "");
}

/** Browser → Fly round-trip can exceed 60s (MediaPipe + Gemini). */
const ANALYZE_VIDEO_TIMEOUT_MS = 180_000;

async function fetchVideoAnalyze(
  url: string,
  form: FormData,
  label: string,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ANALYZE_VIDEO_TIMEOUT_MS);
  const base = apiBase();
  try {
    return await fetch(url, {
      method: "POST",
      body: form,
      mode: "cors",
      credentials: "omit",
      signal: controller.signal,
    });
  } catch (e: unknown) {
    if (e instanceof Error && e.name === "AbortError") {
      throw new Error(
        `${label}: timed out after ${ANALYZE_VIDEO_TIMEOUT_MS / 1000}s. Is ${base} up?`,
      );
    }
    if (e instanceof TypeError) {
      throw new Error(
        `${label}: ${e.message} — cannot reach API at ${base}. Check NEXT_PUBLIC_API_BASE (Vercel) and CORS on the Fly app.`,
      );
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${apiBase()}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status} ${path}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** GET /v1/health */
export async function fetchHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/v1/health");
}

/** GET /v1/sports */
export async function fetchSports(): Promise<SportInfo[]> {
  return apiFetch<SportInfo[]>("/v1/sports");
}

/**
 * One ghost rep vector as accumulated by the browser-side repCounter.
 * Sent alongside the video blob to populate the parity_probe block.
 */
export interface GhostRepVector {
  rep_index: number;
  features: Record<string, { value: number | null; unit: string; status: string; reason_codes: string[] }>;
}

export interface GymAnalyzeRequest {
  exercise_id: string;
  fps: number;
  frames: Record<string, { x: number; y: number; z: number; visibility: number } | null>[];
}

/**
 * POST /v1/analyze/gym with pre-extracted frames (frames_json source).
 * Used when MediaRecorder is not available or for testing with fixture data.
 */
export async function analyzeGym(req: GymAnalyzeRequest): Promise<AnalyzeResponse> {
  return apiFetch<AnalyzeResponse>("/v1/analyze/gym", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// ---------------------------------------------------------------------------
// Basketball (legacy /analyze-video endpoint)
// ---------------------------------------------------------------------------
//
// Basketball was built before the v1 envelope landed and still uses the
// pre-existing /analyze-video route, which returns a richer payload
// (Gemini-powered scout report + ChromaDB-matched NBA pro + biomech
// numbers).  Wiring it through the v1 envelope is a post-showcase task;
// for the demo we surface the legacy response in a dedicated component
// so basketball users get a real canonical analysis with form coaching.
//
// Schema below is *partial* — only fields the BasketballReport renders.
// Server returns more (oracle_caveat, witty_catchphrase, etc.) which we
// allow via index signatures rather than enumerating exhaustively.

export interface BasketballMetricStatus {
  source?: string;
  confidence?: number;
}

export interface MatchedPro {
  name?: string;
  team?: string;
  headshot?: string | null;
  [k: string]: unknown;
}

export interface BasketballAthleteFeedback {
  title?: string;
  feedback?: string;
  drill?: string;
  [k: string]: unknown;
}

export interface BasketballAnalyzeResponse {
  athlete_name?: string;
  sport?: string;
  confidence?: number;                 // 0-100
  analysis_reliability_score?: number; // 0-100
  release_velocity_mps?: number | null;
  shot_arc_deg?: number | null;
  knee_angle?: number | null;
  elbow_angle?: number | null;
  kinetic_sync_ms?: number | null;
  fluidity_score?: number | null;
  hip_rotation_deg?: number | null;
  balance_index?: number | null;
  scout_report?: string;
  athlete_feedback?: BasketballAthleteFeedback[];
  witty_catchphrase?: string;
  matched_pro?: MatchedPro | null;
  oracle_caveat?: string | null;
  oracle_match_degraded?: boolean;
  kinematic_deltas?: Record<string, number | string>;
  metric_status?: Record<string, BasketballMetricStatus>;
  validation_warnings?: string[];
  video_quality_label?: string;
  video_quality_score?: number;
  api_schema_version?: string;
  [k: string]: unknown;
}

/**
 * POST /analyze-video  (legacy basketball pipeline)
 *
 * Slower than the gym pipeline (~20-40 s) because it runs Gemini 2.5
 * Flash on the clip in addition to MediaPipe + biomech.  Returns a
 * scout report, 3 athlete-feedback bullets, and an NBA-pro match.
 */
export async function analyzeBasketballVideo(
  blob: Blob,
  athleteName: string | null = null,
  mimeType = "video/webm",
): Promise<BasketballAnalyzeResponse> {
  const form = new FormData();
  form.append("video", new File([blob], "clip.webm", { type: mimeType }));
  form.append("sport", "basketball");
  if (athleteName && athleteName.trim()) {
    form.append("athlete_name", athleteName.trim());
  }
  const url = `${apiBase()}/analyze-video`;
  const res = await fetchVideoAnalyze(url, form, "Basketball analyze");
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status} /analyze-video: ${text}`);
  }
  return res.json() as Promise<BasketballAnalyzeResponse>;
}

/**
 * POST /v1/analyze/gym/video  (Day 7+8)
 *
 * Uploads the raw WebM blob from MediaRecorder and triggers the full
 * canonical backend pipeline (MediaPipe heavy model + gym measurement spine).
 * When ``ghostReps`` is provided, the backend runs the parity probe and
 * returns a populated ``parity_probe`` block in the response envelope.
 *
 * @param blob        - Raw WebM/MP4 Blob from MediaRecorder.
 * @param exerciseId  - Exercise identifier, e.g. "back_squat".
 * @param mimeType    - MIME type of the blob (default "video/webm").
 * @param ghostReps   - Optional ghost rep vectors from the browser repCounter.
 */
export async function analyzeGymVideo(
  blob: Blob,
  exerciseId: string,
  mimeType = "video/webm",
  ghostReps?: GhostRepVector[],
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("exercise_id", exerciseId);
  form.append("video", new File([blob], "clip.webm", { type: mimeType }));
  if (ghostReps && ghostReps.length > 0) {
    form.append("realtime_vectors_json", JSON.stringify(ghostReps));
  }

  const url = `${apiBase()}/v1/analyze/gym/video`;
  const res = await fetchVideoAnalyze(url, form, "Gym analyze");
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status} /v1/analyze/gym/video: ${text}`);
  }
  return res.json() as Promise<AnalyzeResponse>;
}
