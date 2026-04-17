# ADR 0001: Phase A pose baseline — MediaPipe in the eval harness

- **Status:** Accepted  
- **Date:** 2026-03-27  
- **Context:** Phase A ([RESEARCH_PLAN_POSE_AND_LTX.md](../RESEARCH_PLAN_POSE_AND_LTX.md)) requires a **fixed, reproducible** 2D pose path to score gym clips before rep segmentation and coaching product logic. Alternatives (e.g. RTMPose, ViTPose-class) differ in accuracy, license, ops, and GPU assumptions.

## Decision

1. **Standardize Phase A benchmarking** on **MediaPipe Pose Landmarker (heavy)** behind `app.pose.backends` / `run_mediapipe_pose_baseline`, with **versioned provenance** and **shared FFmpeg normalize** with `KinematicAnalyzer`.
2. **Do not block** adding a second backend implementing the same `PoseBaselineResult` + JSONL contract; any backbone promoted to “default” for reporting must pass **manifest-backed A/B** documented with the eval output.
3. **Heuristic usable gate** (`pose_usable_heuristic`) is **calibrated via** [evaluation/gym_pose_calibration.json](../../evaluation/gym_pose_calibration.json), not ad hoc per-clip code changes.

## Consequences

- **Positive:** Fast iteration, single-stack CI, clear provenance fields, aligned basketball and gym decode paths.
- **Negative:** MediaPipe is not SOTA on all cluttered gym frames; product claims must track **measured** usable-pose rates on the **curated manifest**, not generic SOTA language.
- **Rollback:** Removing MediaPipe from the product is independent of this ADR; eval history remains valid for the schema version recorded in JSONL.

## Notes

Superseding this ADR requires a new ADR referencing **comparative metrics** on the internal manifest and any schema or contract bumps.
