# Research-grade plan: pose estimation + LTX-2.3 media stack

This document records **primary-source research** (as of March 2025), a **staged technical plan**, and a **plan review checklist** before any major implementation. It aligns with [files/PROJECT_CONTEXT.md](../files/PROJECT_CONTEXT.md) and [files/GOALS.md](../files/GOALS.md).

---

## 0. Operating procedure (non-negotiable)

Every significant change to CV, generative media, or evaluation should follow:

1. **Research** — papers, model cards, official docs, license terms, compute needs, failure modes.
2. **Plan** — written phases, success metrics, dependencies, risks (this doc is the living artifact).
3. **Plan review** — Section 6 checklist; revise plan if any item fails.
4. **Execute** — smallest vertical slice that tests the hypothesis; benchmark; then integrate.

Skipping steps creates “phase loops” without cumulative advantage.

---

## 1. LTX-2.3: product and stack (from vendor materials)

### 1.1 What LTX-2.3 is

- **LTX-2.3** is the latest release in the **LTX-2** family: a **diffusion transformer (DiT) joint audio–video** foundation model (paper: [arXiv:2601.03233](https://arxiv.org/abs/2601.03233), linked from [Hugging Face model card](https://huggingface.co/Lightricks/LTX-2.3) and [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)).
- Capabilities emphasized on [the LTX-2.3 product page](https://website.ltx.io/model/ltx-2-3): sharper detail (updated VAE), stronger **prompt adherence** (larger text connector), improved **image-to-video**, cleaner **audio**, **native portrait (1080×1920)**, **up to ~20 s**, **Fast vs Pro** flows, **4K / 50 fps** positioning for production workflows.
- **Open weights** on Hugging Face; **code** in the **LTX-2** monorepo (not only legacy LTX-Video): packages **`ltx-core`**, **`ltx-pipelines`**, **`ltx-trainer`**.

### 1.2 Checkpoints and pipelines (engineering-relevant)

From [Hugging Face `Lightricks/LTX-2.3` README](https://huggingface.co/Lightricks/LTX-2.3) and [LTX-2 root README](https://github.com/Lightricks/LTX-2):

| Artifact | Role |
|----------|------|
| `ltx-2.3-22b-dev` | Full trainable model (bf16) |
| `ltx-2.3-22b-distilled` | Fast path (e.g. 8-step style usage) |
| `ltx-2.3-22b-distilled-lora-384` | Distilled LoRA used in multi-stage pipelines |
| Spatial / temporal **upscalers** | Multi-stage high-res / high-FPS pipelines |
| **Gemma 3** text encoder | Required dependency for the stack |
| Published **LoRA / IC-LoRA** weights | e.g. union control, **motion-track** control; additional camera/pose LoRAs listed in README (note version lineage—some assets may be **2.19b**-named; verify compatibility with 2.3 before production) |

Pipelines include text/image-to-video (one- and two-stage), **distilled**, **IC-LoRA** (video-to-video / image-to-video transforms), **audio-to-video**, **retake**, **keyframe interpolation**, etc.

### 1.3 Fine-tuning and custom LoRA

- Official training path: **`packages/ltx-trainer`** in [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) ([LTX-Trainer README](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/README.md)).
- Model card states reproducing published LoRAs/IC-LoRAs is supported and that **style / motion / likeness** fine-tunes can be **fast** in many settings—still subject to **your data rights, consent, and safety review**.
- **Critical migration note** (LTX-2.3 FAQ on [website.ltx.io/model/ltx-2-3](https://website.ltx.io/model/ltx-2-3)): if you rely on **custom LoRAs** trained on an **older** LTX-2 latent space, you **must retrain for 2.3** before migrating. Plan assumes **no reuse** of unknown third-party LoRAs without verification.

### 1.4 Distribution and license (decision inputs)

- **API**: [docs.ltx.video](https://docs.ltx.video/welcome) — text / image / audio to video, **retake**, **extend**; per-second pricing (see [pricing](https://docs.ltx.video/pricing)).
- **Open source**: Website states deployment **local / on-prem** and mentions **free use for companies under $10M annual revenue** under the **LTX Model License**; **larger commercial** embedding may require the **licensing program**. **Always read** the current [LICENSE](https://github.com/Lightricks/LTX-2/blob/main/LICENSE) and sales terms before shipping a product.

### 1.5 Implication for Laksh.ai

- **LTX-2.3** is a strong candidate for **grounded coaching media** (image-to-video / video-to-video, pose or motion conditioning via IC-LoRA ecosystem) **after** pose tracks and product spec are stable.
- **Dependency order**: **Pose + metrics first**; generative video is **not** a substitute for accurate kinematic estimation.

---

## 2. Pose estimation on casual uploads: problem framing

### 2.1 Why uploads are hard

- **Uncontrolled camera**: unknown intrinsics, motion, cropping, rolling shutter.
- **Appearance**: baggy clothes, occlusion (bar, plates, other people), motion blur.
- **Single view**: full 3D joint torques are **not identifiable** without priors; 2D/3D mesh methods still **hallucinate** depth under ambiguity.

### 2.2 Research directions (representative, not exhaustive)

| Track | Examples / anchors | Strengths | Risks / cost |
|-------|-------------------|------------|----------------|
| **Top-down 2D HPE** | COCO-style ViT backbones (e.g. **ViTPose** lineage), **RTMPose** (MMPose) | Strong keypoints on clear bodies; faster iteration | Depth ambiguous; limb swaps under occlusion |
| **Whole-body mesh / SMPL** | **SMPLer-X** (NeurIPS 2023, [arXiv:2309.17448](https://arxiv.org/abs/2309.17448)), **SMPLest-X** ([arXiv:2501.09782](https://arxiv.org/abs/2501.09782)) | Expressive pose + shape; better for personalization | Compute; domain gap on gym clutter |
| **Foundation “everything” models** | Evolving EHPS / ViT-Huge scaling laws | Single model across tasks | Ops complexity; license and latency |
| **Commercial APIs** | Various pose/body APIs | Speed to demo | Cost, lock-in, privacy |

**MediaPipe** (current): excellent for **latency and integration**, weaker for **challenging gym** frames than SOTA 2D/mesh when quality is the bottleneck.

### 2.3 Long-term architecture (avoid perpetual “phases”)

Pick a **stable interface** in code:

- **Input**: normalized video clip + optional calibration hints.
- **Output**: per-frame **keypoints or mesh** + **per-joint confidence** + **global shot quality** flag.
- **Downstream**: exercise classifier and rep segmenter consume this **tensor API**, not a specific vendor’s raw types.

Then **swap backbones** (MediaPipe → RTMPose → SMPLer-X-class) without rewriting business logic—each swap is benchmarked against the same manifest.

---

## 3. Competitive and ecosystem scan (for positioning, not copying)

| Actor | Relevance |
|-------|-----------|
| **[HomeCourt](https://www.homecourt.ai/)** | Consumer **mobile CV** for basketball: shot metrics, combine-style drills, anthropometry from device camera—proof that **narrow UX + on-device** can win engagement; different stack than server-side gym form. |
| **[Ludimos](https://www.ludimos.com/)** | **Cricket**-centric AI coaching + academy workflows: tagging, match analytics, biomechanics narrative—shows **sport-specific vertical + coach tools** pattern. |
| **[Hudl](https://www.hudl.com/)** | **Scale** play: hybrid **ML + human verification** (e.g. Hudl IQ tracking), calibrated fields, physical metrics—sets the bar for **trust at team/school** level; not a phone-upload gym app, but shows how **serious** buyers think about accuracy. |

**Investor-relevant takeaway**: Defensibility comes from **evaluation harness + vertical workflow + honest uncertainty**, not from claiming SOTA on every YouTube clip.

---

## 4. Technical plan (phased, with gates)

### Phase A — Measurement spine (highest priority)

**Goal:** Reliable pose **or** explicit “cannot analyze” for gym v0 exercises.

**Implemented in repo (baseline slice):** [evaluation/gym_manifest.template.csv](../evaluation/gym_manifest.template.csv), [evaluation/gym_pose_calibration.json](../evaluation/gym_pose_calibration.json), [app/pose/mediapipe_baseline.py](../app/pose/mediapipe_baseline.py), [app/pose/calibration.py](../app/pose/calibration.py), [app/pose/gym_manifest.py](../app/pose/gym_manifest.py), [app/pose/provenance.py](../app/pose/provenance.py), [scripts/eval_pose_baseline.py](../scripts/eval_pose_baseline.py) (including `--validate-only`), specs [docs/gym_pose_evaluation.md](./gym_pose_evaluation.md) and [docs/POSE_EVALUATION_PROTOCOL.md](./POSE_EVALUATION_PROTOCOL.md). FFmpeg normalization is shared via [app/pose/preprocess.py](../app/pose/preprocess.py) with `KinematicAnalyzer`. RTMPose (or other) should plug in via `app.pose.backends` and the same JSONL shape plus backend-specific `provenance`.

1. **Build / curate** an internal **benchmark manifest** (30–200 clips): lighting, occlusion, phone angles; **sparse** 2D/3D pseudo-gold (even expert keyframe labels on a subset).
2. **Baseline**: current MediaPipe pipeline → metrics: detection rate, joint stability, failure taxonomy.
3. **Candidate upgrade**: integrate **one** strong **2D** model (e.g. RTMPose or ViTPose-class via a maintained implementation) behind the pose interface; **A/B** on manifest.
4. **Gate**: Documented **precision/recall** on rep boundaries and **% clips** with usable pose; product copy matches those numbers.

### Phase B — 3D / shape (when 2D plateaus)

1. Pilot **SMPLer-X / SMPLest-X** (or equivalent EHPS) on the same manifest; measure **mesh stability** and **angle agreement** with 2D projections.
2. Use shape params only with **uncertainty** and **consent** (see GOALS Milestone 3).

### Phase C — LTX-2.3 coaching media

1. **Spike** API vs local **LTX-2** inference for **one** flow (e.g. image-to-video + conditioning).
2. Train or adapt **IC-LoRA / custom LoRA** only with **legal dataset**; verify **2.3** compatibility.
3. **Gate**: side-by-side human review on fixed reel; **labeled as synthetic**; fallback to overlay-only.

### Phase D — Scale to additional sports

Only after **Phase A–C gates** pass for gym (per [files/GOALS.md](../files/GOALS.md)).

---

## 5. Plan review checklist (before execution)

Use this after writing or updating the plan:

- [ ] **Problem statement** is one paragraph, falsifiable.
- [ ] **Success metrics** are numeric or categorical (not “better UX”).
- [ ] **Primary sources** cited (paper arXiv, official README, license).
- [ ] **Compute + $** rough budget for each phase.
- [ ] **Privacy / consent** path for body and generative likeness.
- [ ] **Rollback**: baseline (MediaPipe or overlay-only) still works if new model fails.
- [ ] **Scope**: no parallel “rewrite everything” without Phase A results.

If any box is unchecked, **revise the plan** before large code or spend.

---

## 6. References (quick links)

- LTX-2.3 product: [website.ltx.io/model/ltx-2-3](https://website.ltx.io/model/ltx-2-3)  
- LTX open-source hub: [ltx.io/model/open-source](https://ltx.io/model/open-source)  
- Docs / API: [docs.ltx.video](https://docs.ltx.video/welcome)  
- Code: [github.com/Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)  
- Weights: [huggingface.co/Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3)  
- Paper: [arXiv:2601.03233](https://arxiv.org/abs/2601.03233)  
- SMPLer-X: [arXiv:2309.17448](https://arxiv.org/abs/2309.17448)  
- SMPLest-X: [arXiv:2501.09782](https://arxiv.org/abs/2501.09782)  

*Re-run literature and vendor review quarterly; generative video and pose SOTA move quickly.*
