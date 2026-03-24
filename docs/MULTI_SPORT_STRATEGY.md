# Multi-Sport Strategy: Unified vs. Sport-Specific — Research-Grounded Analysis

> **Purpose**: Decide whether to build sport-specific systems or one unified platform that works across multiple sports. Based on research from Frontiers, IEEE, MDPI, SportsPose, IPTC, and pose estimation surveys.

---

## Executive Summary

**Recommendation: Build one unified system with a sport-config layer.**

The same pose skeleton, video pipeline, and quality/uncertainty logic work across sports. What changes is: event phases, key joints of interest, metric definitions, and pro databases. A config-driven architecture (which `sport_configs.py` already sketches) lets you add tennis, golf, soccer, etc. without rewriting the core. Research strongly supports this: human movement shares motor primitives; pose models are sport-agnostic; event detection uses a common structure (prep → action → follow-through) with sport-specific phases.

---

## 1. What Research Says: Shared vs. Sport-Specific

### A. Human Movement: Common Primitives

| Source | Finding | Implication |
|-------|---------|-------------|
| **PMC (2012)** — [Kinematic Motion Primitives](https://pmc.ncbi.nlm.nih.gov/articles/PMC3469981/) | Human movement is built from **motor primitives** — invariant modules reused across activities | The same underlying motion structure (joint angles, velocities, timing) applies across sports |
| **Frontiers (2022)** — [Unifying framework for motor primitives](https://www.frontiersin.org/articles/10.3389/fncom.2022.926345) | Anechoic mixture model subsumes many primitive models; enables **comparison across studies and domains** | One analytical framework can describe movements from basketball, tennis, golf |

**Implication**: The physics of joint angles, velocities, and kinetic chains is shared. Sport-specificity lives in *which* joints and *when* (event phases), not in the math.

---

### B. Pose Estimation: Sport-Agnostic by Design

| Source | Finding | Implication |
|-------|---------|-------------|
| **SportsPose (CVPR 2023)** | 176,000+ poses from **24 subjects, 5 sports**, same 3D skeleton; validated vs. marker-based mocap (34.5 mm error) | One pose model serves multiple sports; no need for sport-specific pose networks |
| **DeepSportLab (2021)** | Unified framework: ball detection + player segmentation + pose in **team sports** (basketball) | Multi-task, but pose backbone is shared |
| **AdaptPose (2021)** | Cross-dataset adaptation for 3D pose; 14% improvement in generalization | Transfer across sports/domains is feasible with adaptation |
| **Pose2Sim** | Markerless kinematics from **any cameras**, any pose model; supports OpenSim; **sport-agnostic** | Same pipeline: 2D pose → 3D → kinematics |
| **MMPose** | Modular: human, animal, hand; plug-and-play backbones; **sport-agnostic** | Extensibility through config, not new models per sport |

**Implication**: MediaPipe, RTMPose, MMPose are sport-agnostic. You do *not* need a different pose model per sport.

---

### C. Event Detection: Common Structure, Sport-Specific Phases

| Sport | Phase Model | Source |
|-------|-------------|--------|
| **Tennis serve** | 8 stages in 3 phases: Preparation (start, release, loading, cocking) → Acceleration (acceleration, contact) → Follow-through (deceleration, finish) | [PMC (2012)](https://ncbi.nlm.nih.gov/pmc/articles/PMC3445225/) |
| **Basketball shot** | 5 phases: Setup → Dip → Drive → Release → Follow-through | Our `sport_configs.py` |
| **Golf swing** | 5+ phases: Address → Backswing → Downswing → Impact → Follow-through | Standard golf biomechanics |
| **Generic throwing** | 6-stage model for overhead athletes | Same PMC source |

**Pattern**: Prep → Peak/Critical Moment → Follow-through. The *names* and *number of sub-phases* differ, but the structure is shared.

**Implication**: One event-detection abstraction with sport-specific phase definitions. Config defines phases; core logic finds "key moments" (min/max joints, velocity peaks) within each phase.

---

### D. Metrics: Sport-Specific Formulas, Shared Infrastructure

| Sport | Key Metrics | Shared? |
|-------|-------------|---------|
| **Basketball** | Release velocity, shot arc, knee/elbow angles, kinetic sync, fluidity, balance | Formula sport-specific; joint-angle math shared |
| **Tennis** | Racket head speed, toss height, shoulder external rotation, knee flexion at trophy | Different joints; same 3D angle / velocity logic |
| **Golf** | Club path, face angle, hip rotation, X-factor (shoulder-hip separation) | Different; same principle: angles, timing, sequencing |
| **Soccer** | Kick velocity, plant foot angle, hip flexion at contact | Again: angles, velocities, timing |

**Implication**: The *compute layer* (angle from 3 points, velocity from displacement, timing from frames) is shared. The *definitions* (which joints, which frames) live in config.

---

## 2. Architecture: Unified Core + Sport Config

### Core (Shared Across All Sports)

```
Video Input
    ↓
Pose Extraction (MediaPipe / RTMPose) — sport-agnostic
    ↓
Video Quality Score (resolution, FPS, visibility, people count) — sport-agnostic
    ↓
Uncertainty Quantification (per-frame variance, MAD) — sport-agnostic
    ↓
Sport Config Lookup (id: "basketball" | "tennis" | "golf" | ...)
```

### Sport Config (Per-Sport Definition)

Each sport config defines:

| Field | Purpose | Example (Basketball) | Example (Tennis) |
|-------|---------|----------------------|------------------|
| `event_phases` | Names of phases for UI and logic | Setup, Dip, Drive, Release, Follow-through | Preparation, Acceleration, Follow-through |
| `key_joints` | Which landmarks matter for this sport | wrist, elbow, shoulder, knee, hip, ankle | shoulder, elbow, wrist, hip, knee |
| `event_detection` | How to find key frames | wrist Y max = dip; wrist Y min = release | shoulder ER max = trophy; contact = racket-ball |
| `metrics` | What to compute and display | release_velocity_mps, shot_arc_deg, ... | racket_head_speed, toss_height, ... |
| `ideal_ranges` | Target values per metric | knee 140–165°, arc 45–55° | knee flexion at trophy 20–40° |
| `pro_db_collection` | ChromaDB collection for pro matching | apex_oracle_v7 (NBA) | tennis_oracle_v1 (ATP) |

### Data Flow

1. User selects sport (or default: basketball).
2. Backend loads `sport_configs[sport_id]`.
3. Pose extraction runs (same for all sports).
4. Event detection uses config's phase rules (e.g. "release = argmin(wrist_y)" for basketball; "contact = peak(shoulder_rotation)" for tennis).
5. Metrics computed via config-defined formulas.
6. Pro match from sport-specific ChromaDB (or skip if `pro_db_collection` is null).

---

## 3. What This Means for Your Codebase

### Already in Place

- `sport_configs.py`: `BASKETBALL_CONFIG`, `TENNIS_CONFIG`, `GOLF_CONFIG` (stubs).
- Structure: `event_phases`, `metrics`, `min_clip_sec`, `pro_db_collection`.

### Not Yet Wired

- `app/main.py` and `app/physics_engine.py` are basketball-hardcoded (wrist dip/release, 8D vector).
- Event detection, metric formulas, and pro matching assume basketball.

### Migration Path (Long-Term)

1. **Extract event detection** into a pluggable module: `get_key_frames(frames, config)` → `{dip_frame, release_frame, ...}`.
2. **Extract metric computation** into config-driven functions: `compute_metrics(joints, key_frames, config)`.
3. **Per-sport ChromaDB** (or shared with sport_id in metadata) for pro matching.
4. **UI**: Sport selector already exists; feed it from `sport_configs`.

---

## 4. Sport-Specific vs. Unified: Honest Comparison

| Approach | Pros | Cons |
|----------|------|------|
| **Separate system per sport** | Tailored UX; no cross-sport coupling | Duplicate code; N× maintenance; harder to share improvements |
| **Unified + config** | One codebase; shared quality/uncertainty; add sports by config | Some abstraction complexity; config must be rich enough |
| **Hybrid** | Core unified; sport-specific "plugins" for edge cases | More moving parts; need clear plugin contract |

**Research supports unified + config**: Pose2Sim, MMPose, vailá, IPTC Sport Schema all use a generic core with extensions. Sport-specific systems (e.g. CourtMotion for basketball) focus on one domain; platforms (Pose2Sim, Sports2D) are sport-agnostic.

---

## 5. Recommended Roadmap for Multi-Sport

### Phase 2 (Current): Basketball + Unified Foundations

- Implement video quality score, uncertainty, method cards for **basketball**.
- Design APIs and data structures so they can later take a `sport_id` parameter.
- In `physics_engine`, isolate "basketball-specific" logic with clear comments (e.g. `# Basketball: dip = argmax(wrist_y)`).

### Phase 3: Basketball Accuracy + Sport Config Wiring

- Add multi-shot, ball tracking, RTMPose for basketball.
- Refactor: move event detection and metric formulas behind `sport_config`.
- Basketball becomes the first "full" sport config.

### Phase 4: Second Sport (Tennis or Golf)

- Add `TENNIS_CONFIG` or `GOLF_CONFIG` with real phases and metrics.
- Implement tennis serve event detection (trophy, contact) or golf impact detection.
- Add tennis/golf metrics (e.g. racket speed, shoulder rotation).
- Optional: separate ChromaDB collection for tennis/golf pros.

### Phase 5+: Scale

- More sports by adding configs.
- Shared: pose, quality, uncertainty, UI.
- Sport-specific: phases, metrics, pro DB.

---

## 6. What to Build Now (Phase 2) With Multi-Sport in Mind

| Component | Multi-Sport Safe? | Action |
|-----------|-------------------|--------|
| Video quality score | Yes | Implement; no sport dependence |
| Uncertainty (±) | Yes | Implement; uses same joints |
| Method cards | Partially | Start with basketball; template so tennis/golf can follow |
| Confidence factors | Yes | Implement; generic (quality, people, visibility) |
| Event detection | No | Keep basketball-specific for now; add `# TODO: sport_config` |
| Metric formulas | No | Keep basketball-specific; refactor in Phase 3 |

**Principle**: Add nothing that *blocks* multi-sport. Quality, uncertainty, and confidence are universal. Event detection and metrics stay篮球-specific until we refactor to config-driven.

---

## 7. References (Summary)

| Topic | Source |
|-------|--------|
| Motor primitives | PMC (2012), Frontiers (2022) |
| SportsPose dataset | DTU/CVPR 2023 — 5 sports, same skeleton |
| Event phases | PMC (2012) — tennis 8-stage; IEEE — generic framework |
| Pose2Sim, MMPose | Open-source, sport-agnostic |
| IPTC Sport Schema | Competitions, events, actions — extensible by sport |
| DeepSportLab, AdaptPose | Multi-sport pose; cross-dataset transfer |

---

## 8. Conclusion

**One unified system with a sport-config layer is the right approach.** The research shows that:

1. Pose estimation is sport-agnostic.
2. Movement shares common kinematic primitives.
3. Event structure (prep → action → follow-through) is shared; phases are sport-specific.
4. Metrics differ by sport, but the underlying math (angles, velocities, timing) is shared.
5. Existing platforms (Pose2Sim, MMPose, IPTC) use this pattern.

Build Phase 2 with basketball as the default, but keep the architecture config-ready. In Phase 3, refactor event detection and metrics to be config-driven, then add tennis or golf as the second sport.

---

*Document version: 1.0 | Research completed for multi-sport strategy*
