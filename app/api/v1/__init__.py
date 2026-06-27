"""Laksh.ai HTTP API v1.

The v1 surface unifies basketball and gym under a single response schema
(:mod:`app.api.v1.schema`) with explicit provenance and analysis-mode
tagging. See ``docs/adr/0004-realtime-dual-path.md`` for the dual-path
(realtime_preview + canonical_backend) design.

Routes
------
POST /v1/analyze/gym
    Run the gym measurement spine on pre-extracted pose frames
    (:mod:`app.gym.pipeline`). No MediaPipe required.

GET  /v1/sports
    List sports with per-sport capability flags
    (:mod:`app.sport_configs`).

GET  /v1/health
    Liveness probe with manifest SHAs so a calling client can verify
    the server is running the taxonomy / calibration it expects.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.analyze import router as analyze_router
from app.api.v1.coaching import router as coaching_router
from app.api.v1.health import router as health_router
from app.api.v1.leaderboard import router as leaderboard_router
from app.api.v1.sports import router as sports_router

#: Prefix applied when the v1 router is mounted on the parent app.
V1_PREFIX = "/v1"

router = APIRouter(prefix=V1_PREFIX)
router.include_router(health_router)
router.include_router(sports_router)
router.include_router(analyze_router)
router.include_router(leaderboard_router)
router.include_router(coaching_router)

__all__ = ["V1_PREFIX", "router"]
