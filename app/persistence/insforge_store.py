"""InsForge-backed SessionStore.

Persists sessions + per-rep results to, and reads the leaderboard from, the
InsForge cloud database via its REST API:

    POST/GET {base}/api/database/records/{table}
    Authorization: Bearer <ik_ project key>
    insert body is a JSON array; reads use PostgREST filters
    (order=col.desc, limit=N, col=eq.val, col=not.is.null)

Verified live against the project before wiring. Reads degrade to an empty
leaderboard on any failure so the endpoint never 500s; writes surface failures
to the caller (analysis persistence is best-effort upstream).
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from app.persistence.models import LeaderboardEntry, SessionRecord
from app.persistence.store import SessionStore

log = logging.getLogger(__name__)

#: (method, path, body) -> (http_status, parsed_json_or_None)
Transport = Callable[[str, str, Optional[Any]], tuple[int, Any]]

_SESSIONS = "/api/database/records/sessions"
_REP_RESULTS = "/api/database/records/rep_results"


class InsForgeSessionStore(SessionStore):
    """SessionStore backed by the InsForge cloud REST API."""

    backend_name = "insforge"

    def __init__(self, api_url: str, api_key: str, transport: Optional[Transport] = None) -> None:
        self._base = api_url.rstrip("/")
        self._key = api_key
        self._transport = transport or self._http

    def _http(self, method: str, path: str, body: Optional[Any] = None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self._base + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 -- fixed https host
                raw = resp.read().decode()
                return resp.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as e:
            return e.code, None

    def persist(self, record: SessionRecord) -> str:
        session_row = {
            "session_id": record.session_id,
            "created_at": record.created_at,
            "sport_id": record.sport_id,
            "exercise_id": record.exercise_id,
            "display_name": record.display_name,
            "git_commit_sha": record.git_commit_sha,
            "pose_baseline_version": record.pose_baseline_version,
            "model": record.model,
            "source": record.source,
            "fps": record.fps,
            "n_frames": record.n_frames,
            "n_reps": record.n_reps,
            "n_valid_reps": record.n_valid_reps,
            "form_index": record.form_index,
            "form_index_status": record.form_index_status,
            "form_index_reason_codes": record.form_index_reason_codes,
            "form_index_components": record.form_index_components,
        }
        status, _ = self._transport("POST", _SESSIONS, [session_row])
        if status >= 300:
            raise RuntimeError(f"InsForge session insert failed: HTTP {status}")

        rep_rows = [
            {
                "session_id": record.session_id,
                "rep_index": rep.rep_index,
                "rep_status": rep.rep_status,
                "measured": rep.measured,
            }
            for rep in record.fingerprint
        ]
        if rep_rows:
            rs, _ = self._transport("POST", _REP_RESULTS, rep_rows)
            if rs >= 300:
                # Non-fatal: the session row (which the leaderboard needs) is saved.
                log.warning("InsForge rep_results insert failed: HTTP %s (session kept)", rs)
        return record.session_id

    def leaderboard(
        self, exercise_id: Optional[str] = None, limit: int = 10
    ) -> list[LeaderboardEntry]:
        params = ["form_index=not.is.null", "order=form_index.desc", f"limit={int(limit)}"]
        if exercise_id:
            params.append(f"exercise_id=eq.{urllib.parse.quote(exercise_id)}")
        path = f"{_SESSIONS}?{'&'.join(params)}"
        try:
            status, rows = self._transport("GET", path, None)
        except Exception:  # noqa: BLE001 -- a read failure must not 500 the endpoint
            log.warning("InsForge leaderboard read error", exc_info=True)
            return []
        if status >= 300 or not isinstance(rows, list):
            log.warning("InsForge leaderboard read failed: HTTP %s", status)
            return []

        entries: list[LeaderboardEntry] = []
        for i, row in enumerate(rows):
            fi = row.get("form_index")
            if fi is None:
                continue
            entries.append(
                LeaderboardEntry(
                    rank=i + 1,
                    session_id=row.get("session_id", ""),
                    display_name=row.get("display_name", "anon"),
                    exercise_id=row.get("exercise_id", ""),
                    form_index=float(fi),
                    form_index_status=row.get("form_index_status", "unknown"),
                    n_valid_reps=int(row.get("n_valid_reps") or 0),
                    n_reps=int(row.get("n_reps") or 0),
                    created_at=str(row.get("created_at", "")),
                    git_commit_sha=row.get("git_commit_sha"),
                )
            )
        return entries

    def health(self) -> bool:
        try:
            status, _ = self._transport("GET", f"{_SESSIONS}?limit=1", None)
            return status < 300
        except Exception:  # noqa: BLE001
            return False
