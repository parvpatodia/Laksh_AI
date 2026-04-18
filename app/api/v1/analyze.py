"""``/v1/analyze`` routes.

Day-1 exposes the gym path only. Basketball lands on a later day once
:class:`~app.physics_engine.KinematicAnalyzer` is adapted to emit the
unified ``feature_vectors`` schema; until then, basketball analysis
continues to flow through the legacy ``/analyze-video`` endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from app.api.v1.provenance import build_provenance
from app.api.v1.schema import (
    AnalyzeGymRequest,
    AnalyzeResponseModel,
)
from app.gym.pipeline import UnknownExerciseError, analyze_gym_clip
from app.gym.pose_adapter import frames_json_to_canonical_frames

router = APIRouter(tags=["analyze"])


@router.post(
    "/analyze/gym",
    response_model=AnalyzeResponseModel,
    status_code=status.HTTP_200_OK,
    summary="Analyse a gym clip from pre-extracted canonical pose frames",
)
def analyze_gym(req: AnalyzeGymRequest) -> AnalyzeResponseModel:
    """Run the gym measurement spine on ``req.frames``.

    The endpoint accepts pre-extracted canonical-joint frames (the same
    shape as the ``--frames-json`` CLI fixture) so it does not require
    MediaPipe at request time. This is the path the browser will POST
    to after computing landmarks client-side via @mediapipe/tasks-vision.
    """
    try:
        canonical_frames = frames_json_to_canonical_frames(req.frames)
    except Exception as e:  # pose_adapter is permissive; explicit guard is cheap
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"invalid frames payload: {e}",
        ) from e

    try:
        result = analyze_gym_clip(
            exercise_id=req.exercise_id,
            fps=req.fps,
            canonical_frames=canonical_frames,
            source="frames_json",
        )
    except UnknownExerciseError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e

    # Wrap the pipeline result into the v1 response envelope.
    envelope = {
        "sport_id": "gym",
        "exercise_id": result["exercise_id"],
        "source": result["source"],
        "fps": result["fps"],
        "n_frames": result["n_frames"],
        "analysis_mode": "canonical_backend",
        "provenance": build_provenance(model="none_frames_json").model_dump(),
        "segment": result["segment"],
        "feature_vectors": [
            {
                "rep_index": fv["rep_index"],
                "start_frame": fv["start_frame"],
                "end_frame": fv["end_frame"],
                "peak_frame": fv["peak_frame"],
                "rep_status": fv["rep_status"],
                "features": fv["features"],
            }
            for fv in result["feature_vectors"]
        ],
        "calibration": result["calibration"],
        "parity_probe": None,
    }
    try:
        return AnalyzeResponseModel.model_validate(envelope)
    except ValidationError as e:
        # A schema drift between pipeline and response model is a server
        # bug, not a client error.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"response envelope failed validation: {e.errors()}",
        ) from e
