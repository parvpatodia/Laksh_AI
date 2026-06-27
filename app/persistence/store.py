"""Session store abstraction, SQLite fallback, Null store, and backend factory.

Callers depend on :class:`SessionStore`, never on a concrete backend. The
backend is chosen by :func:`get_session_store` from the
``LAKSH_PERSISTENCE_BACKEND`` env flag:

* ``sqlite`` (default) -- local SQLite file; the always-available fallback so
  the live demo never breaks.
* ``insforge`` -- InsForge-backed store, selected once the MCP/key are wired.
  Until the adapter exists, this safely falls back to SQLite.
* ``none`` -- a no-op store (persistence disabled).
"""
from __future__ import annotations

import abc
import logging
import os
import sqlite3
from functools import cache
from pathlib import Path
from typing import Optional

from app.persistence.models import LeaderboardEntry, SessionRecord

log = logging.getLogger(__name__)

#: Default SQLite location. Overridable via env so the Fly container can point
#: it at the mounted volume for cross-restart persistence if desired.
_DEFAULT_DB_PATH = Path(os.environ.get("LAKSH_SQLITE_PATH", "/tmp/laksh_sessions.db"))


class SessionStore(abc.ABC):
    """Persistence interface for analysis sessions + leaderboard reads."""

    backend_name: str = "abstract"

    @abc.abstractmethod
    def persist(self, record: SessionRecord) -> str:
        """Persist a session; return its ``session_id``."""

    @abc.abstractmethod
    def leaderboard(
        self, exercise_id: Optional[str] = None, limit: int = 10
    ) -> list[LeaderboardEntry]:
        """Return best-scored sessions (highest ``form_index`` first)."""

    @abc.abstractmethod
    def health(self) -> bool:
        """Return True if the backend is reachable."""


class NullSessionStore(SessionStore):
    """No-op store: accepts writes, serves an empty leaderboard."""

    backend_name = "null"

    def persist(self, record: SessionRecord) -> str:
        return record.session_id

    def leaderboard(self, exercise_id: Optional[str] = None, limit: int = 10) -> list[LeaderboardEntry]:
        return []

    def health(self) -> bool:
        return True


class SqliteSessionStore(SessionStore):
    """Local SQLite-backed store. The demo's always-available fallback."""

    backend_name = "sqlite"

    def __init__(self, db_path: Path | str = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id        TEXT PRIMARY KEY,
                    created_at        TEXT NOT NULL,
                    sport_id          TEXT NOT NULL,
                    exercise_id       TEXT NOT NULL,
                    display_name      TEXT NOT NULL,
                    git_commit_sha    TEXT,
                    form_index        REAL,
                    form_index_status TEXT,
                    n_reps            INTEGER,
                    n_valid_reps      INTEGER,
                    record_json       TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_leaderboard "
                "ON sessions(exercise_id, form_index DESC)"
            )

    def persist(self, record: SessionRecord) -> str:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    session_id, created_at, sport_id, exercise_id, display_name,
                    git_commit_sha, form_index, form_index_status, n_reps,
                    n_valid_reps, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.created_at,
                    record.sport_id,
                    record.exercise_id,
                    record.display_name,
                    record.git_commit_sha,
                    record.form_index,
                    record.form_index_status,
                    record.n_reps,
                    record.n_valid_reps,
                    record.model_dump_json(),
                ),
            )
        return record.session_id

    def leaderboard(
        self, exercise_id: Optional[str] = None, limit: int = 10
    ) -> list[LeaderboardEntry]:
        # Only sessions with a measurable index rank; unscored sessions are
        # persisted but never pollute the board.
        query = "SELECT * FROM sessions WHERE form_index IS NOT NULL"
        params: list[object] = []
        if exercise_id:
            query += " AND exercise_id = ?"
            params.append(exercise_id)
        query += " ORDER BY form_index DESC, created_at ASC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            LeaderboardEntry(
                rank=i + 1,
                session_id=row["session_id"],
                display_name=row["display_name"],
                exercise_id=row["exercise_id"],
                form_index=row["form_index"],
                form_index_status=row["form_index_status"],
                n_valid_reps=row["n_valid_reps"],
                n_reps=row["n_reps"],
                created_at=row["created_at"],
                git_commit_sha=row["git_commit_sha"],
            )
            for i, row in enumerate(rows)
        ]

    def health(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False


def get_session_store(
    backend: Optional[str] = None, db_path: Path | str | None = None
) -> SessionStore:
    """Resolve the configured store. Defaults to SQLite; never raises."""
    resolved = (backend or os.environ.get("LAKSH_PERSISTENCE_BACKEND", "sqlite")).lower()

    if resolved == "none":
        return NullSessionStore()
    if resolved == "insforge":
        # WHY: the InsForge adapter is wired only once the MCP/key are
        # provisioned. Falling back keeps the live demo + leaderboard working.
        log.warning(
            "LAKSH_PERSISTENCE_BACKEND=insforge but the adapter is not wired yet; "
            "using the SQLite fallback."
        )
        resolved = "sqlite"
    if resolved != "sqlite":
        log.warning("unknown persistence backend %r; using sqlite", resolved)
    return SqliteSessionStore(db_path=db_path or _DEFAULT_DB_PATH)


@cache
def get_default_store() -> SessionStore:
    """Process-wide store resolved from env. Cached so all routers share one."""
    return get_session_store()
