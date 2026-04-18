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

/**
 * POST /v1/analyze/gym/video  (Day 7)
 *
 * Uploads the raw WebM blob from MediaRecorder and triggers the full
 * canonical backend pipeline (MediaPipe heavy model + gym measurement spine).
 *
 * @param blob        - Raw WebM/MP4 Blob from MediaRecorder.
 * @param exerciseId  - Exercise identifier, e.g. "back_squat".
 * @param mimeType    - MIME type of the blob (default "video/webm").
 */
export async function analyzeGymVideo(
  blob: Blob,
  exerciseId: string,
  mimeType = "video/webm",
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("exercise_id", exerciseId);
  form.append("video", new File([blob], "clip.webm", { type: mimeType }));

  const url = `${apiBase()}/v1/analyze/gym/video`;
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status} /v1/analyze/gym/video: ${text}`);
  }
  return res.json() as Promise<AnalyzeResponse>;
}
