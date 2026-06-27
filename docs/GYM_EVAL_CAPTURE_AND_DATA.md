# Gym evaluation — capture protocol and internal data practice

This document supports **Phase A** in [RESEARCH_PLAN_POSE_AND_LTX.md](./RESEARCH_PLAN_POSE_AND_LTX.md): comparable clips, honest metrics, and defensible methodology. It applies to the **internal benchmark manifest** ([evaluation/gym_manifest.template.csv](../evaluation/gym_manifest.template.csv)), not to end-user product terms (those require separate legal/product review).

Formal metric definitions: [POSE_EVALUATION_PROTOCOL.md](./POSE_EVALUATION_PROTOCOL.md).

---

## 1. Capture protocol (technical)

Goal: reduce **uncontrolled variance** so `detection_rate` and `pose_usable_heuristic` deltas reflect pipeline or backbone changes—not ad hoc recording differences.

### 1.1 Subject and framing

| Guideline | Rationale |
|-----------|-----------|
| **Primary subject** occupies a **stable share** of frame height (roughly **⅓–½** of frame height for full-body lifts). | Extreme wide shots inflate `multiple_people_detected` noise and lower effective resolution after 720p normalize. |
| **Single-lifter rows** should show **only one person** executing the movement; bystanders out of frame or far background. | The landmarker uses the **first** detected pose; extra people add ambiguity ([reason codes](./POSE_EVALUATION_PROTOCOL.md#reason-code-registry)). |
| **Multi-person test rows** (`tags` include `multi_person`) are **intentional** stress tests—document in `notes` who is the intended subject if relevant. | Separates “failure to isolate” from “pipeline bug.” |

### 1.2 Camera angle by exercise class (v0)

Angles are **reporting metadata** (`tags` / `notes`); they are not a guarantee of 3D joint accuracy.

| Exercise family | Preferred primary angle | Secondary (optional row) |
|------------------|-------------------------|---------------------------|
| Squat, hinge (deadlift, RDL) | **Side (sagittal)** — full body feet to head | Frontal (coronal) for knee tracking |
| Bench / horizontal press | **Diagonal ~30–45°** or side if shoulders visible | — |
| Pull-up / vertical pull | **Front or 45°**; bar and full hang visible | — |
| Row variants | **Side** to spine and arm plane | — |

### 1.3 Timebase, duration, resolution

| Parameter | Recommendation |
|-----------|----------------|
| **Duration** | **≥5 s** for “clean” rows so `n_frames` is well above the usable-gate minimum after decimation; **≤90 s** per clip for batch sanity unless stress-testing. Rows labeled `short_clip` may be shorter to test edge behavior (expect unstable metrics). |
| **Frame rate** | **30 fps** preferred; variable frame rate (common on phones) is **acceptable** if FFmpeg normalize is used in eval (see `ffmpeg_preprocess_applied` in JSONL). |
| **Resolution** | **720p or higher** on the long side at capture is sufficient; the pipeline caps long side at **720** after normalize. |
| **Orientation** | Portrait vs landscape is allowed; rotation should be **metadata-correct** on the file (FFmpeg bakes rotation when normalize runs). |

### 1.4 Lighting and environment

- **Avoid heavy backlighting** against windows; it drives down visibility scores without necessarily testing pose logic.
- **Occlusion rows** (bar on shins, crowded gym) should be **tagged explicitly** so regressions are interpreted as expected difficulty, not surprise.

### 1.5 File format

- **Container:** MP4 (phone default is fine).
- **Codec:** H.264 is ideal for cross-machine decode; **HEVC (H.265)** is acceptable for **iPhone stress rows**—always record `ffmpeg_preprocess_applied` when comparing runs.

---

## 2. Manifest discipline

- **`clip_id`:** Stable opaque id; do not rename after a published eval run (breaks longitudinal comparison).
- **`path`:** Repo-root-relative as in the template; binary files stay **gitignored**; only paths and metadata are versioned.
- **`exercise_id`:** Controlled vocabulary per release (e.g. `squat`, `deadlift`); free text only if explicitly marked experimental in `notes`.
- **`tags`:** Pipe-separated tokens (`squat|side_view|phone_clean`). Use a **consistent lexicon** across rows (document new tokens in the PR that adds clips).
- **`expect_pose_usable` / `expect_min_detection_rate`:** Set only after human review of that clip; enables `--strict-manifest` in CI.

---

## 3. Internal data handling (R&D)

This is **operational hygiene**, not legal advice.

| Topic | Practice |
|-------|----------|
| **Consent** | Contributors who appear on camera should **consent** to use for internal R&D and benchmarking; document consent method your organization requires. |
| **PII** | Do not put names, emails, or member IDs in `notes`, filenames, or commit messages. Use `clip_id` only. |
| **Storage** | Clips are **local or org-controlled storage**; the repo tracks **manifests and metrics**, not necessarily the binaries. |
| **Retention** | Define a **retention period** for raw clips (e.g. delete after N months or after model generation upgrade) consistent with your policy. |
| **Labeling** | Pseudo-gold keyframe labels, when added, should be **versioned** (file or dataset version) and referenced from the manifest or a sidecar spec. |

---

## 4. Related documents

- [docs/gym_pose_evaluation.md](./gym_pose_evaluation.md) — commands and field meanings.
- [docs/adr/0001-phase-a-mediapipe-baseline.md](./adr/0001-phase-a-mediapipe-baseline.md) — Phase A backbone decision.
- [docs/GOLDEN_VIDEO_GUIDE.md](./GOLDEN_VIDEO_GUIDE.md) — basketball regression golden clip (separate from gym manifest).
