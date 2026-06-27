"""FastAPI dependencies for the v1 surface.

Keeping the store behind a dependency lets routes stay backend-agnostic and
lets tests override the store with a temp-dir SQLite instance via
``app.dependency_overrides``.
"""
from __future__ import annotations

import os

from app.coaching.you_search import YouComClient
from app.persistence.store import SessionStore, get_default_store


def get_store() -> SessionStore:
    """Return the process-wide session store."""
    return get_default_store()


def get_you_client() -> YouComClient:
    """Return a You.com client, or a disabled one when grounding is off.

    Grounding is enabled iff ``YOU_API_KEY`` is set and
    ``LAKSH_COACHING_GROUNDING`` is not ``off``. A disabled client makes the
    coaching endpoint degrade to ungrounded (no-citation) cues.
    """
    if os.environ.get("LAKSH_COACHING_GROUNDING", "").lower() == "off":
        return YouComClient(api_key=None)
    return YouComClient(livecrawl=os.environ.get("YOU_LIVECRAWL"))
