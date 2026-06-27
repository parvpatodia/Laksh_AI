"""FastAPI dependencies for the v1 surface.

Keeping the store behind a dependency lets routes stay backend-agnostic and
lets tests override the store with a temp-dir SQLite instance via
``app.dependency_overrides``.
"""
from __future__ import annotations

from app.persistence.store import SessionStore, get_default_store


def get_store() -> SessionStore:
    """Return the process-wide session store."""
    return get_default_store()
