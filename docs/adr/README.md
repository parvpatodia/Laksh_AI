# Architecture Decision Records (Laksh.ai)

Short, durable decisions for components that are costly to reverse (pose stack, eval contracts, major dependencies).

| ADR | Title | Status |
|-----|--------|--------|
| [0001](./0001-phase-a-mediapipe-baseline.md) | Phase A pose baseline: MediaPipe in eval harness | Accepted |
| [0002](./0002-p3-canonical-in-kinematic-analyzer.md) | P3: canonical joints in `KinematicAnalyzer` | Accepted (partial) |

**Execution contract** (mapping, gates, labeling claims, P0–P4): [POSE_UPGRADE_EXECUTION_PLAN.md](../POSE_UPGRADE_EXECUTION_PLAN.md). **P0:** canonical + MediaPipe map + provenance `1.2.0`. **P1a:** optional `rtmpose` backend (`rtmlib`) + `mapping_rtmpose_coco17` + `requirements-pose-optional.txt`.

When an ADR is superseded, add **Superseded by ADR-XXXX** to the header and keep the file for history.
