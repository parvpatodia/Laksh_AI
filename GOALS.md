# Project Goals

## Vision

**A gym-focused movement coach that users actually open repeatedly:** upload a set of **known exercises**, get **rep-level feedback** grounded in **measured kinematics and calibrated models**, and (once mature) **personalized coaching video**—including **before / after or corrected-motion visualizations** driven by **state-of-the-art generative video** (e.g. **Lightricks LTX-Video / LTX-2** and successors), optionally conditioned on **body shape from a 360° onboarding scan**.

Long-term, **sports-specific modes** (basketball, etc.) are **in scope only after** the gym loop proves **reliable media + evaluation**; the current basketball stack in-repo is a **stepping stone**, not the final product definition.

## Current Phase

**Strategic pivot + technical migration:** From **basketball-only demo** toward **gym exercise MVP** while preserving what works (FastAPI ingest, pose pipeline patterns, LLM narrative, evaluation discipline). Generative video (LTX-class) and 360° body priors are **Phase B/C**, not pretend-shipped features.

## Milestones

### Milestone 1: Gym vertical — exercise scope + measurement core

- **Target date:** TBD
- **Definition of done:**
  - **Exercise v0 list** agreed (e.g. 8–15 movements with clear camera instructions)—documented for users and tests.
  - **Rep segmentation** working above a defined **F1 / IoU bar** on an internal labeled mini-set (even 30–50 clips is enough to start if methodology is sound).
  - **Per-rep feature vector** defined (joint angles, velocities, depth proxies, symmetry)—each field has **valid / degraded / unknown** semantics like today’s `metric_status` pattern.
  - **Calibration policy:** No new **silent** hardcoded “ideal angles” in code; reference ranges come from **documented sources** (literature bins + optional dataset fit) and live in **versioned config** committed with the eval run that justified them.
  - **`pytest`** green; new tests cover classification/segmentation boundaries and failure modes.
- **Status:** **In progress** — Phase A **measurement spine** (manifest, calibration, MediaPipe baseline JSONL, shared FFmpeg path, reason-code taxonomy, **canonical COCO-17 joint contract** + MediaPipe mapping + provenance **1.2.0**) is in-repo. **Execution contract:** [docs/POSE_UPGRADE_EXECUTION_PLAN.md](docs/POSE_UPGRADE_EXECUTION_PLAN.md) — **P0 done**; **P1a–P1b done** (RTMPose path, A/B compare + orchestration, strict readiness CI); **next pose-engineering step** in that doc is **P2** (person ROI / tracker on hard subset) unless you prioritize **P3** (canonical into `KinematicAnalyzer`).
- **Blockers:** Curated labeled gym video (L1 tier for strong accuracy claims); ONNX/GPU env for P1 if not CPU-only.

### Milestone 2: Coaching media — before / after & LTX-class integration

- **Target date:** TBD
- **Definition of done:**
  - **Product spec** for “before / after”: same user, same exercise—**synthetic corrected motion** vs **overlay-only** fallback documented.
  - **LTX-class** (or measured-better alternative) **spike**: conditioning signals chosen (pose sequence, depth/edge, reference frame), latency and **$ / minute** characterized for startup-scale budget.
  - **Safety:** On-screen labeling that generated segments are **AI visualization**, not medical advice; opt-in consent for likeness use in generation.
  - **Quality gate:** Human rating or structured checklist on a **fixed eval reel** before beta (honest small-n is fine).
- **Status:** Not started
- **Blockers:** GPU/API access; legal/copy review for generative likeness.

### Milestone 3: Personalization — 360° body scan + longitudinal progress

- **Target date:** TBD
- **Definition of done:**
  - **Capture protocol** for 360° scan (duration, clothing, background) that yields **stable scale** for height / segment lengths within stated error bands.
  - **Body model** path (e.g. SMPL/SMPL-X fit or commercial API) with **uncertainty outputs**; no claim of medical body composition.
  - **Integration:** Personalized cues and generative conditioning use **posterior body params**; A/B or offline eval shows **measurable** improvement vs generic priors on a held-out set.
  - **Privacy:** Data retention, deletion, and on-device vs server processing documented and implemented.
- **Status:** Not started
- **Blockers:** Ethics/privacy review; quality of fit on diverse bodies.

### Milestone 4 (later): Sports expansion

- **Target date:** TBD
- **Definition of done:** At least **one** additional sport mode ships only if Milestones 1–2 are **complete**, with sport-specific eval manifests and **no regression** on gym core quality.
- **Status:** Deferred
- **Blockers:** Depends on Milestones 1–2.

## Non-Goals (near term)

- **Positioning basketball jump shot as the final product** — it stays supported in code until migrated, but **roadmap and agent priorities** follow **gym-first** ([PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)).
- **“Any movement in any sport”** from a single upload — deferred until gym + media loop is proven.
- **Clinical / medical claims** — injury diagnosis, load prescription for pathology, eating-disorder-linked body metrics — out of scope.
- **Pretending generative video is ground truth** — synthetic demos must be **labeled** and **grounded** in measured issues.
- **Chasing every new model weekly without evaluation** — new components require **benchmark + ADR**.
- **100% perfect pose on all clutter/occlusion** — out of scope; ship **confidence and refusal**.

## Quality Bar

- **Evidence-linked coaching:** User-facing text maps to **computed features** and their uncertainty.
- **Reproducibility:** Thresholds and priors are **versioned** and tied to eval artifacts.
- **Tests:** Core parsing, API contracts, and critical CV paths covered by automated tests.
- **Research credibility:** Non-trivial modeling choices reference **papers, standard benchmarks, or internal ablations** in `docs/` or PR descriptions—not vibes.

## Constraints

- **R&D process:** For pose, generative video, or evaluation overhauls, follow **research → plan → plan review → execute**; keep [docs/RESEARCH_PLAN_POSE_AND_LTX.md](docs/RESEARCH_PLAN_POSE_AND_LTX.md) and the **execution contract** [docs/POSE_UPGRADE_EXECUTION_PLAN.md](docs/POSE_UPGRADE_EXECUTION_PLAN.md) current. No large model swap without manifest-backed metrics (§4–§5 there).
- **North star vertical:** **Gym / strength training** first; sports are **sequential**, not parallel, until media milestone justifies split focus.
- **Stack evolution:** FastAPI + Python remain default; **pose / gen-video** components may swap after comparative eval.
- **Team size:** Prefer **small, measurable milestones** and **evaluation manifests** over big-bang rewrites.
- **Regulatory sensitivity:** Fitness coaching only; avoid regulated medical positioning without counsel.

---

*This file overrides casual README bullets when they conflict with **Vision**, **Milestones**, or **Non-Goals** until deliberately revised.*

**Note on generative video:** Primary reference is **LTX-2.3** ([Lightricks/LTX-2](https://github.com/Lightricks/LTX-2), [Hugging Face weights](https://huggingface.co/Lightricks/LTX-2.3), [paper arXiv:2601.03233](https://arxiv.org/abs/2601.03233)) with **`ltx-trainer`** for custom **LoRA / IC-LoRA**; **retrain LoRAs** when upgrading across latent-space generations (vendor migration FAQ). Replace only when a **better-evaluated** option wins on quality, cost, and license.
