"""``POST /v1/coaching/ground`` -- ground detected coaching faults in You.com sources.

The frontend detects faults deterministically (FormInsights) and POSTs them
here; the server retrieves real sources per fault via You.com and returns each
cue with resolvable citations. When grounding is disabled (no key /
``LAKSH_COACHING_GROUNDING=off``) the cues come back ungrounded but intact, so
the report never breaks.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.deps import get_you_client
from app.coaching.grounding import GroundedCue, ground_faults
from app.coaching.you_search import YouComClient

router = APIRouter(tags=["coaching"])

_DISCLAIMER = (
    "Coaching faults are detected by deterministic rules on your measured reps. "
    "Citations are live sources retrieved via You.com to ground the remediation; "
    "the detection itself is never LLM-invented."
)


class FaultInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = ""
    cue: str = ""
    query: Optional[str] = None


class GroundCoachingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exercise_id: str
    faults: list[FaultInput]
    freshness: Optional[str] = "year"


class GroundCoachingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exercise_id: str
    grounding_enabled: bool
    backend: str
    disclaimer: str = _DISCLAIMER
    cues: list[GroundedCue] = Field(default_factory=list)


@router.post(
    "/coaching/ground",
    response_model=GroundCoachingResponse,
    summary="Ground deterministic coaching faults in cited You.com sources",
)
def ground_coaching(
    req: GroundCoachingRequest,
    client: YouComClient = Depends(get_you_client),
) -> GroundCoachingResponse:
    cues = ground_faults(
        req.exercise_id,
        [f.model_dump() for f in req.faults],
        client=client,
        freshness=req.freshness,
    )
    return GroundCoachingResponse(
        exercise_id=req.exercise_id,
        grounding_enabled=client.enabled,
        backend="you.com" if client.enabled else "fallback",
        cues=cues,
    )
