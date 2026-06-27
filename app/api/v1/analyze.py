"""``/v1/analyze`` routes.

Day-1: gym frames_json path.
Day-7: gym video path (multipart WebM -> MediaPipe heavy -> pipeline).
Day-8: realtime parity probe wired into the video endpoint.

Basketball analysis still flows through the legacy ``/analyze-video``
endpoint until the KinematicAnalyzer result-builder is adapted to the
unified feature_vectors schema (planned post-showcase).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.api.v1.deps import get_store
from app.api.v1.provenance import build_provenance
from app.api.v1.schema import (
    AnalyzeGymRequest,
    AnalyzeResponseModel,
    LeaderboardStandingModel,
    ParityProbeModel,
)
from app.gym.pipeline import UnknownExerciseError, analyze_gym_clip
from app.gym.pose_adapter import frames_json_to_canonical_frames
from app.parity.realtime import probe_reps
from app.persistence.models import build_session_record
from app.persistence.store import SessionStore

log = logging.getLogger(__name__)

router = APIRouter(tags=["analyze"])


def _persist_and_rank(
    store: SessionStore, envelope: AnalyzeResponseModel, display_name: str
) -> LeaderboardStandingModel | None:
    """Persist the session and compute its leaderboard standing -- both best-effort.

    Persistence is a side effect of analysis: if the store is down the user still
    gets their report, just without a standing. Never raises.
    """
    try:
        record = build_session_record(envelope.model_dump(), display_name=display_name)
        store.persist(record)
    except Exception:  # noqa: BLE001 -- persistence must never break analysis
        log.warning("session persistence failed (non-fatal)", exc_info=True)
        return None

    if record.form_index is None:
        return None  # unscored session never ranks

    rank: int | None = None
    total = 0
    try:
        board = store.leaderboard(exercise_id=record.exercise_id, limit=1000)
        total = len(board)
        rank = next((e.rank for e in board if e.session_id == record.session_id), None)
    except Exception:  # noqa: BLE001 -- a ranking miss must not break the report
        log.warning("leaderboard standing lookup failed (non-fatal)", exc_info=True)

    return LeaderboardStandingModel(
        form_index=record.form_index,
        form_index_status=record.form_index_status,
        rank=rank,
        total=total,
    )

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

_MAX_CLIP_BYTES = 50 * 1024 * 1024  # 50 MB


def _build_envelope(
    result: dict,
    model: str,
    parity_probe: ParityProbeModel | None = None,
) -> AnalyzeResponseModel:
    """Wrap a pipeline result dict into the v1 response envelope.

    Parameters
    ----------
    result:
        Raw dict returned by :func:`app.gym.pipeline.analyze_gym_clip`.
    model:
        Pose model identifier string for the provenance block.
    parity_probe:
        Optional validated parity probe block. ``None`` when realtime
        vectors were not submitted or could not be compared.
    """
    envelope = {
        "sport_id": "gym",
        "exercise_id": result["exercise_id"],
        "source": result["source"],
        "fps": result["fps"],
        "n_frames": result["n_frames"],
        "analysis_mode": "canonical_backend",
        "provenance": build_provenance(model=model).model_dump(),
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
        "parity_probe": parity_probe.model_dump() if parity_probe is not None else None,
    }
    try:
        return AnalyzeResponseModel.model_validate(envelope)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"response envelope failed validation: {e.errors()}",
        ) from e


# ---------------------------------------------------------------------------
# POST /v1/analyze/gym  (frames_json source)
# ---------------------------------------------------------------------------


@router.post(
    "/analyze/gym",
    response_model=AnalyzeResponseModel,
    status_code=status.HTTP_200_OK,
    summary="Analyse a gym clip from pre-extracted canonical pose frames",
)
def analyze_gym(
    req: AnalyzeGymRequest,
    store: SessionStore = Depends(get_store),
) -> AnalyzeResponseModel:
    """Run the gym measurement spine on ``req.frames``.

    Accepts pre-extracted canonical-joint frames (same shape as the
    ``--frames-json`` CLI fixture). The browser POSTs here after computing
    landmarks client-side via @mediapipe/tasks-vision.
    """
    try:
        canonical_frames = frames_json_to_canonical_frames(req.frames)
    except Exception as e:
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    envelope = _build_envelope(result, model="none_frames_json")
    envelope.leaderboard_standing = _persist_and_rank(store, envelope, req.display_name or "anon")
    return envelope


# ---------------------------------------------------------------------------
# POST /v1/analyze/gym/video  (raw video source — Day 7)
# ---------------------------------------------------------------------------


@router.post(
    "/analyze/gym/video",
    response_model=AnalyzeResponseModel,
    status_code=status.HTTP_200_OK,
    summary="Analyse a gym clip from a raw video file (MediaPipe heavy backend)",
)
async def analyze_gym_video(
    exercise_id: str = Form(..., description="Exercise identifier, e.g. 'back_squat'"),
    video: UploadFile = File(..., description="Raw WebM/MP4 clip from MediaRecorder"),
    realtime_vectors_json: str | None = Form(
        None,
        description=(
            "Optional JSON-encoded list of ghost rep vectors from the browser "
            "repCounter, used to compute the parity_probe block."
        ),
    ),
    display_name: str | None = Form(None, description="Leaderboard display name"),
    store: SessionStore = Depends(get_store),
) -> AnalyzeResponseModel:
    """Run the full canonical pipeline on an uploaded video file.

    1. Write the upload to a secure temp file.
    2. Call ``extract_canonical_frames`` (MediaPipe heavy, VIDEO mode).
    3. Run the gym measurement spine via ``analyze_gym_clip``.
    4. If ``realtime_vectors_json`` is provided, run :func:`probe_reps`
       against the canonical feature vectors and embed the result in the
       ``parity_probe`` block of the response envelope.
    5. Return the v1 response envelope with ``source='video'`` and
       ``model='mediapipe_pose_landmarker_heavy'``.

    The endpoint is synchronous from the client's perspective (it blocks
    until MediaPipe finishes). On a 1 GB Fly machine a 5-second 720p clip
    takes ~8-15 seconds. This is acceptable for the showcase demo; see
    ADR 0003 for async queue plans.
    """
    raw = await video.read()
    if len(raw) > _MAX_CLIP_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"clip too large: {len(raw)} bytes > {_MAX_CLIP_BYTES} limit",
        )
    if len(raw) < 512:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="clip appears empty or truncated",
        )

    # Write to a temp file that MediaPipe can open via OpenCV.
    suffix = ".webm" if (video.content_type or "").startswith("video/webm") else ".mp4"
    tmp_path = os.path.join(tempfile.gettempdir(), f"laksh_{uuid.uuid4().hex}{suffix}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(raw)

        try:
            from app.gym.pose_adapter import extract_canonical_frames
        except ImportError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"MediaPipe not available: {e}",
            ) from e

        try:
            fps, canonical_frames = extract_canonical_frames(tmp_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"pose extraction failed: {e}",
            ) from e

        try:
            result = analyze_gym_clip(
                exercise_id=exercise_id,
                fps=fps,
                canonical_frames=canonical_frames,
                source="video",
            )
        except UnknownExerciseError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
        except Exception as e:
            # Catches internal pipeline errors (scipy, numpy, feature extraction) so
            # an unexpected edge-case never returns a raw 500 Internal Server Error.
            log.exception("analyze_gym_clip raised unexpected error: %s", e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Clip analysis failed: {e}. Try re-recording with your full body visible.",
            ) from e

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Parity probe: compare realtime ghost vectors against canonical output.
    parity_probe: ParityProbeModel | None = None
    if realtime_vectors_json:
        try:
            realtime_vecs: list[dict] = json.loads(realtime_vectors_json)
            if realtime_vecs:
                canonical_vecs = result["feature_vectors"]
                probe_raw = probe_reps(realtime_vecs, canonical_vecs)
                parity_probe = ParityProbeModel.model_validate(probe_raw)
        except Exception:
            log.warning(
                "parity probe failed -- skipping (realtime_vectors_json may be malformed)",
                exc_info=True,
            )

    envelope = _build_envelope(result, model="mediapipe_pose_landmarker_heavy", parity_probe=parity_probe)
    envelope.leaderboard_standing = _persist_and_rank(store, envelope, display_name or "anon")
    return envelope
