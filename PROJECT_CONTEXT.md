# Project Context

## Domain

**Laksh.ai** is evolving toward a **gym- and strength-training form coaching** product: users upload video of defined exercises, the system estimates pose and dynamics, compares them against **data-calibrated** reference models (not ad-hoc “magic numbers”), and returns **actionable feedback** with a roadmap to **before / after style coaching media** generated from the user’s own clip.

**Product north star (intentionally broader than “one shot”):** **One deep vertical first—structured gym movements** (a bounded set of exercises, clear camera setups, rep segmentation)—then **sports expansion only after** the **personalized correction / demonstration video loop** is technically and ethically solid. Basketball jump-shot analysis remains **valuable reference implementation** in the codebase today but is **not** the long-term primary vertical.

**360° onboarding capture (roadmap):** Users may record a **short multi-view body scan** (e.g. turn 360° in frame) so the system can estimate **anthropometry and body shape priors** (height, segment lengths, girth proxies where reliable) to **personalize** cues, joint-angle interpretations, and generated coaching video. This is **research-grade hard**: acceptable outputs require **probabilistic body models** (e.g. SMPL / SMPL-X family), **multi-view or temporal consistency**, and **explicit uncertainty**—not a claim of clinical DEXA-level accuracy from a phone clip.

## Tech Stack

### Implemented today (repository)

| Layer | Technology |
|--------|------------|
| Language | Python 3.11 |
| API | FastAPI (`uvicorn`) |
| CV / pose | MediaPipe Pose Landmarker (heavy task), OpenCV, FFmpeg |
| Numerics | NumPy, SciPy, Pandas |
| Vector store | ChromaDB |
| Reference / narrative | NBA-heuristic seeding (`nba_api`), Gemini, optional Imagen / TTS |
| UI | `static/dashboard.html` (React 18 + Tailwind CDN) |
| Tests | `pytest`, optional golden fixtures |
| Lint (optional) | `ruff` in `pyproject.toml` |
| Container | `Dockerfile` |

See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) and [README.md](../README.md) for the **current** basketball-oriented pipeline.

### Planned / evaluation stack (gym + media — align to best available tools)

The team should **track and benchmark** what small teams ship in 2025–2026, then **version-pin** choices in an ADR—not chase every release blindly.

| Capability | Direction |
|------------|-----------|
| **Pose / mesh** | Keep MediaPipe as baseline; evaluate **stronger 2D/3D HPE** (e.g. ViTPose-class, RTMPose, DensePose-line) and **parametric mesh recovery** (SMPL/SMPL-X fitting, HMR-style methods) where gym metrics need 3D consistency. |
| **Exercise classification & reps** | Action recognition + temporal segmentation (video transformers, TCNs, or API-classifiers); **train or fine-tune on labeled gym data** rather than hand-maintained frame rules for each lift. |
| **Generative coaching video** | **LTX-2.3 (Lightricks)** — DiT **joint audio–video** model ([arXiv:2601.03233](https://arxiv.org/abs/2601.03233)); weights [Hugging Face `Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3), code monorepo [`Lightricks/LTX-2`](https://github.com/Lightricks/LTX-2) with **`ltx-trainer`** for **LoRA / IC-LoRA** fine-tunes. **Custom LoRAs trained on older LTX-2 latents must be retrained for 2.3** (vendor FAQ). Use [docs.ltx.video](https://docs.ltx.video/welcome) for API; verify **license** for your revenue tier. Legacy [LTX-Video](https://github.com/Lightricks/LTX-Video) repo remains part of the broader ecosystem but **2.3 primary path is LTX-2**. Planned use: **pose/motion-conditioned** coaching clips after pose spine is gated—see [docs/RESEARCH_PLAN_POSE_AND_LTX.md](../docs/RESEARCH_PLAN_POSE_AND_LTX.md). |
| **Physics & biomechanics** | Use **segment lengths, joint limit priors, and inverse dynamics *where identifiable*** with uncertainty; ground claims in **biomechanics literature** (e.g. squat/deadlift kinematics studies) and **population statistics**, not single hardcoded “ideal angles” for all bodies. |
| **Parameters** | Prefer **dataset-fitted calibration**, **learned heads**, and **versioned config** (YAML/JSON from experiments) over scattered literals in code; acknowledge that **all** models embed priors—success is **reproducibility and measurability**, not “zero constants.” |

## Architecture Overview (target state)

**Today:** upload → `physics_engine` (basketball shot) → 8D vector → Chroma NBA match → Gemini → dashboard / correction hooks.

**Target (gym-first):**

1. **Onboarding (optional)** — 360° clip → multi-view pose / mesh fit → **user body model + uncertainties** stored with consent.
2. **Workout clip** — exercise detect → rep boundaries → per-rep **kinematic features** + confidence.
3. **Scoring / feedback** — compare to **calibrated** reference distributions (per exercise, demographic bin or personalized prior), not fixed global thresholds.
4. **Media** — **LTX-class** (or successor) **video generation** using user appearance + pose conditioning → **side-by-side or before/after** coaching artifact; human-review gate until safety bar is met.
5. **Narrative** — LLM grounded in **measured deltas** and **known failure modes** for that exercise class.

Until the gym pipeline lands, agents may **extend** the existing FastAPI app incrementally (new routes, `sport_configs` / exercise registry, parallel analyzers) rather than rewriting blindly.

## Conventions

- **R&D workflow (mandatory for CV, generative media, evaluation):** **Research → written plan → plan review → execute.** Do not ship large model or pipeline changes without consulting [docs/RESEARCH_PLAN_POSE_AND_LTX.md](../docs/RESEARCH_PLAN_POSE_AND_LTX.md) (checklist §5) and, for pose backends / preprocess / eval claims, [docs/POSE_UPGRADE_EXECUTION_PLAN.md](../docs/POSE_UPGRADE_EXECUTION_PLAN.md).
- **Package layout:** `app/`, `static/`, `tests/`, `scripts/`, `evaluation/` unchanged unless ADR says otherwise.
- **Logging:** `logging` via `app/logging_config.py`; no `print` in app paths ([CONTRIBUTING.md](../CONTRIBUTING.md)).
- **Research hygiene:** Significant model or threshold changes require **a short rationale** (paper, internal ablation, or benchmark delta) in PR description or `docs/`.
- **No silent magic numbers:** New tunables go through **config + evaluation manifest**; document what data calibrated them.
- **API contracts:** Backward-compatible JSON where possible; version endpoints if gym v1 diverges from legacy basketball responses.
- **Safety:** No injury diagnosis; encourage qualified in-person coaching for pain or load management.

## Current Focus

**Strategic:** Lock **gym exercise set v0**, **capture protocol**, and **evaluation plan** (clips + labels + metrics). **Technical:** Prototype exercise detection + rep segmentation on top of existing pose stack; plan **LTX-class** integration path (API vs self-host) for **corrected-motion or comparative video** output.

## Known Issues / Tech Debt

- **Implementation vs vision gap:** Production code path is still **basketball jump shot**; migrating to gym is a **major** `physics_engine` / config / evaluation effort—track in [GOALS.md](GOALS.md).
- **Monocular limits:** Single-view phone video cannot recover full 3D kinetics without strong priors; **360° helps** but is not a silver bullet.
- **Generative video risk:** Identity, consent, and misleading “perfect form” footage—need **disclaimers**, **watermarking**, and **grounding** in actual measurements.
- **LTX / vendor drift:** Generative video APIs and weights churn; pin versions and maintain a **fallback** (e.g. overlay + TTS only).
- **Golden / CI:** Gym golden fixtures do not exist yet; basketball golden path may still apply for regression until replaced.

## Industry and research context

- **Pose & shape:** Progress from **top-down 2D** (COCO-style) to **3D mesh** (SMPL, SPIN, PIXIE, SMPLer-X) and **in-the-wild** benchmarks drives what is honest to claim from phone video.
- **Action recognition:** Kinetics-style pretraining, Video Swin / TimeSformer-class models, and **fine-tuned** gym classifiers beat hand-crafted state machines at scale—when labeled data exists.
- **Generative video:** Open **video diffusion** stacks (including **LTX-2** and ecosystem trainers/control adapters) enable **pose- or depth-conditioned** generation—relevant to **coaching visualizations** if grounded in real tracks.
- **Startups:** Form-check apps and wearables combine **simple UX**, **narrow exercise sets**, and **clear limitations**; differentiation is **trust + evaluation**, not slogan depth.
- **Labs / pro:** Markerless **multi-camera** mocap (arena / lab) sets the **upper bound** on accuracy; consumer products must **not** imply parity.

## Agent Persona Override

Act as a **research lead + staff ML engineer** at the intersection of **computer vision, biomechanics communication, and generative media**. You read **papers and benchmarks** before arguing for architecture changes. You treat **LTX-class video** as a **planned, evaluated subsystem** (latency, cost, conditioning signal, failure modes), not a default answer.

1. **Gym-first product sense** — Prioritize **bounded exercise sets**, **rep integrity**, and **user-trustable** feedback over horizontal sport coverage.
2. **Parameters from data** — Replace brittle hardcoded thresholds with **calibrated curves, learned layers, or population priors**; document datasets and splits.
3. **Physics-aware** — Use **joint constraints, segment proportions, and timing** consistently; cite or sketch the biomechanical reasoning.
4. **Epistemic honesty** — Separate **measurement**, **model guess**, and **generative synthesis**; never present generated video as **recorded ground truth**.
5. **Implementation continuity** — Respect existing FastAPI + MediaPipe + Gemini stack while **proposing** migrations (new pose model, gym `SportConfig`, LTX-2.3 integration) with **incremental PRs** and tests.
6. **Process discipline** — Follow **research → plan → plan review → execute** for any significant model change; update the research plan doc when facts change.

When uncertain, **state assumptions** and recommend **an experiment** (ablation, benchmark clip, user study) instead of speculative rewrites.
