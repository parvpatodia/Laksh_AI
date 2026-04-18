"""``GET /v1/health`` - liveness probe that also pins manifest SHAs."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.provenance import build_provenance
from app.api.v1.schema import (
    V1_RESPONSE_SCHEMA_VERSION,
    HealthResponseModel,
)

router = APIRouter(tags=["meta"])


@router.get(
    "/health",
    response_model=HealthResponseModel,
    summary="Liveness probe with manifest SHAs",
)
def health() -> HealthResponseModel:
    """Return ``ok`` plus a provenance block.

    A client that pinned a specific calibration SHA at build time can
    verify the running server still matches it, which is what catches
    "someone deployed a new calibration but the frontend is still
    claiming cited evidence against the old ranges" drift.
    """
    return HealthResponseModel(
        status="ok",
        v1_schema_version=V1_RESPONSE_SCHEMA_VERSION,
        provenance=build_provenance(model="none_frames_json"),
    )
