"""``GET /v1/sports`` - capabilities advertisement for the frontend."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.schema import SportInfoModel
from app.gym.exercises_v0 import EXERCISES_V0
from app.sport_configs import get_available_sports

router = APIRouter(tags=["meta"])


@router.get(
    "/sports",
    response_model=list[SportInfoModel],
    summary="List sports the server can analyse",
)
def list_sports() -> list[SportInfoModel]:
    """Return one row per registered sport.

    For ``gym`` the ``exercises`` field is populated from the frozen v0
    exercise taxonomy so the frontend can render the exercise picker
    without a second round-trip.
    """
    rows: list[SportInfoModel] = []
    for row in get_available_sports():
        exercises: list[str] = []
        if row["id"] == "gym":
            exercises = sorted(EXERCISES_V0.keys())
        rows.append(
            SportInfoModel(
                id=row["id"],
                name=row["name"],
                available=bool(row["available"]),
                exercises=exercises,
            )
        )
    return rows
