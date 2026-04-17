# ADR 0002: P3 — Canonical joints in `KinematicAnalyzer` (planned)

## Status

**Partially implemented.** Parity telemetry (canonical vs legacy 2D angles at key frames) ships behind `LAKSH_USE_CANONICAL_JOINTS`; default metrics still use the legacy index path. Full “metrics from canonical only” remains gated on frozen basketball manifest review. Tracks [POSE_UPGRADE_EXECUTION_PLAN.md](../POSE_UPGRADE_EXECUTION_PLAN.md) §0 P3.

## Context

`KinematicAnalyzer` ([app/physics_engine.py](../../app/physics_engine.py)) still reads MediaPipe landmark indices directly. P0–P2 built **canonical COCO-17** types, mappings, and gym/basketball baselines; product parity requires the same contract in the live analysis path.

## Decision (when implemented)

- Add a **feature flag** (environment or config) defaulting **off** until a frozen **basketball manifest diff** shows acceptable parity.
- Map per-frame pose → `canonical` joint dict before metrics; preserve **provenance** in debug output.
- Document **single_pass** behavior explicitly (no silent multipass inflation for “prod” numbers).

## Consequences

- Touches hot path (`physics_engine`); requires golden/benchmark regression on `evaluation/manifest.csv` and gym rows.
- Unblocks honest multi-backend comparisons for user-facing metrics.
