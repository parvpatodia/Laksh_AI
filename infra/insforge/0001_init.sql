-- Laksh.ai InsForge schema (provisioned by the agent via `insforge db import`).
-- users/auth are provided by InsForge's built-in auth; we add the analysis
-- domain: one row per analysis session + one row per rep, plus a leaderboard index.

CREATE TABLE IF NOT EXISTS sessions (
    session_id              TEXT PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    sport_id                TEXT NOT NULL,
    exercise_id             TEXT NOT NULL,
    display_name            TEXT NOT NULL DEFAULT 'anon',
    git_commit_sha          TEXT,
    pose_baseline_version   TEXT,
    model                   TEXT,
    source                  TEXT,
    fps                     DOUBLE PRECISION,
    n_frames                INTEGER,
    n_reps                  INTEGER NOT NULL DEFAULT 0,
    n_valid_reps            INTEGER NOT NULL DEFAULT 0,
    form_index              DOUBLE PRECISION,
    form_index_status       TEXT,
    form_index_reason_codes JSONB,
    form_index_components   JSONB
);

CREATE TABLE IF NOT EXISTS rep_results (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    rep_index   INTEGER NOT NULL,
    rep_status  TEXT NOT NULL,
    measured    JSONB
);

-- Leaderboard read path: best form_index per exercise.
CREATE INDEX IF NOT EXISTS idx_sessions_leaderboard ON sessions (exercise_id, form_index DESC);
CREATE INDEX IF NOT EXISTS idx_rep_results_session ON rep_results (session_id);
