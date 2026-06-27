"""Ground deterministic coaching faults in real You.com sources.

A fault is detected by an explicit rule on measured reps (see
``web/components/FormInsights.tsx``). This module takes those faults and, per
fault, retrieves real sources for the exercise + fault and attaches resolvable
citations. The deterministic cue text is preserved verbatim -- grounding ADDS
sources, it never rewrites or invents the measured finding.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.coaching.you_search import YouComClient

# REF: fault ids mirror web/components/FormInsights.tsx insight ids. Each maps to
# the search terms that retrieve remediation sources for that specific fault.
_FAULT_QUERY_TERMS: dict[str, str] = {
    "tempo-fast-eccentric": "eccentric tempo control slow descent technique",
    "tempo-pause": "paused rep tempo training technique",
    "consistency-low": "rep cadence consistency fatigue form breakdown",
    "rom-degrading": "range of motion drop fatigue technique cue",
    "vis-low": "filming setup camera angle full body in frame",
}

_EXERCISE_TERMS: dict[str, str] = {
    "back_squat": "back squat",
    "dumbbell_bicep_curl": "dumbbell bicep curl",
    "overhead_press": "overhead press",
    "bench_press": "bench press",
    "deadlift": "deadlift",
}


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    snippet: str = ""


class GroundedCue(BaseModel):
    """A deterministic cue plus the real sources that ground its remediation."""

    model_config = ConfigDict(extra="forbid")

    fault_id: str
    title: str
    cue: str
    query: str
    grounded: bool
    citations: list[Citation] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


def prettify_exercise(exercise_id: str) -> str:
    return _EXERCISE_TERMS.get(exercise_id, exercise_id.replace("_", " "))


def build_query(exercise_id: str, fault: dict[str, Any]) -> str:
    """Compose a search query from the exercise and the specific fault."""
    exercise = prettify_exercise(exercise_id)
    terms = _FAULT_QUERY_TERMS.get(
        fault.get("id", ""), fault.get("title", "") or "form technique"
    )
    return f"{exercise} {terms} coaching"


def ground_faults(
    exercise_id: str,
    faults: list[dict[str, Any]],
    client: YouComClient,
    *,
    freshness: Optional[str] = "year",
    max_citations: int = 3,
) -> list[GroundedCue]:
    """Attach You.com citations to each fault's deterministic cue.

    Never raises: a disabled client, a failed search, or zero results all
    degrade to an ungrounded cue (cue text preserved) with an explanatory
    reason code.
    """
    cues: list[GroundedCue] = []
    for fault in faults:
        fault_id = str(fault.get("id", ""))
        query = str(fault.get("query") or build_query(exercise_id, fault))
        citations: list[Citation] = []
        reason_codes: list[str] = []

        if not client.enabled:
            reason_codes.append("grounding_unavailable_no_key")
        else:
            try:
                sources = client.search(query, count=max_citations, freshness=freshness)
            except Exception:  # noqa: BLE001 -- grounding must never break the report
                sources = []
                reason_codes.append("you_com_search_failed")
            citations = [
                Citation(title=s.title, url=s.url, snippet=s.snippet)
                for s in sources[:max_citations]
            ]
            if citations:
                reason_codes.append("you_com_grounded")
                if freshness:
                    reason_codes.append(f"freshness:{freshness}")
            elif "you_com_search_failed" not in reason_codes:
                reason_codes.append("no_sources_found")

        cues.append(
            GroundedCue(
                fault_id=fault_id,
                title=str(fault.get("title", "")),
                cue=str(fault.get("cue", "")),
                query=query,
                grounded=bool(citations),
                citations=citations,
                reason_codes=reason_codes,
            )
        )
    return cues
