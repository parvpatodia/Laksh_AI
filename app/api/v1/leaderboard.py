"""``GET /v1/leaderboard`` -- best form-index sessions per exercise.

The ranking metric (``form_index``) is a transparent, uncalibrated *relative*
index built only from measured quantities. It is deliberately NOT a validated
form grade, and the response says so in ``disclaimer`` so the honesty contract
extends all the way to the leaderboard.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.deps import get_store
from app.persistence.models import LeaderboardEntry
from app.persistence.store import SessionStore

router = APIRouter(tags=["leaderboard"])

_DISCLAIMER = (
    "form_index is a transparent, uncalibrated relative ranking derived only "
    "from measured quantities (valid-rep ratio, tracking quality, tempo "
    "consistency). It is NOT a validated form grade and claims no reference range."
)


class LeaderboardResponseModel(BaseModel):
    """Payload for ``GET /v1/leaderboard``."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    exercise_id: Optional[str] = None
    count: int
    disclaimer: str = _DISCLAIMER
    entries: list[LeaderboardEntry] = Field(default_factory=list)


@router.get(
    "/leaderboard",
    response_model=LeaderboardResponseModel,
    summary="Best form-index sessions, optionally filtered by exercise",
)
def get_leaderboard(
    exercise_id: Optional[str] = Query(None, description="Filter to one exercise id"),
    limit: int = Query(10, ge=1, le=100),
    store: SessionStore = Depends(get_store),
) -> LeaderboardResponseModel:
    entries = store.leaderboard(exercise_id=exercise_id, limit=limit)
    return LeaderboardResponseModel(
        backend=store.backend_name,
        exercise_id=exercise_id,
        count=len(entries),
        entries=entries,
    )
