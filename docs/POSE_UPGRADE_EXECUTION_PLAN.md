# Pose upgrade & evaluation — execution contract

This document **tightens** the strategic roadmap (see [RESEARCH_PLAN_POSE_AND_LTX.md](./RESEARCH_PLAN_POSE_AND_LTX.md)) with **gates, schemas, and claim discipline** so implementation stays interpretable. It incorporates an internal expert review (canonical representation, labeling, denoise, multipass honesty, ops modes, basketball parity).

**Audience:** engineers implementing backends, preprocess, or metrics. **Not** legal advice for consent or licensing—verify terms for each weight checkpoint.

---

## 0. Plan status & how to execute

| Phase | Status | What to do next |
|-------|--------|-----------------|
| **P0** | **Done** | Canonical types + MediaPipe→canonical map + provenance (`pose_baseline` **1.2.0**). |
| **P1a** | **Done** | **RTMPose** path via optional **`rtmlib`** (`Body`: YOLOX + RTMPose ONNX): `app/pose/rtmpose_baseline.py`, `mapping_rtmpose_coco17`, `get_pose_backend("rtmpose")`, shared `gym_baseline_metrics`. Install: `pip install -r requirements-pose-optional.txt`. First run may **download** model zips. |
| **P1b** | **Done** (optional polish remains) | **Tooling:** `compare_pose_baseline_jsonl.py`, `pose_baseline_compare.py` (mean/median/min/max Δ, **FFmpeg mismatch confound** notes), `make compare-pose-ab`, **`run_pose_ab_eval_compare.py` / `make eval-pose-ab-orchestrate`**. **Ops:** readiness report **schema 1.2.0** (subprocess native probes, `probe_error` / `interpreter`, **MediaPipe `.task` SHA-256** vs `app/pose/expected_artifacts.py`), `make check-pose-readiness` / **`check-pose-readiness-strict`**, **CI strict gate** after model download. Interpret A/B as **L0** (§5). **Archival:** `eval_scorecard_header.py` **1.1.0** — `--jsonl` (repeatable) hashes pose JSONL for PR/scorecard bundles. *Optional later:* pin downloaded ONNX file hashes in JSONL provenance. |
| **P2** | **In progress** | **Shipped:** `--person-isolation haar_mil_v1` (`app/pose/person_isolation.py`); **compare:** `p2_l0` block + per-clip Haar attempts in `pose_baseline_compare.py`; **orchestration:** `scripts/run_pose_isolation_ab_compare.py`, `make eval-pose-isolation-ab`; **template:** `evaluation/gym_manifest_hard.template.csv`. **Next:** curated real clips on hard manifest + ONNX detector if Haar insufficient. |
| **P3** | **In progress** | **Telemetry:** `LAKSH_USE_CANONICAL_JOINTS=1` → `telemetry.canonical_joint_path` (2D angle parity vs legacy at key frames). **Next:** optional metrics-from-canonical + frozen basketball manifest diff. |
| **P4** | Pending | LTX spike (separate budget + ADR). |

**Before every pose/preprocess PR:** Re-read §4 (denoise + multipass gates) and §5 (claim tiers).

**Quick verification (no GPU, no gym video required):**

```bash
make test-pose-core
make check-pose-readiness-strict   # or: make check-pose-readiness
python3 scripts/eval_pose_baseline.py --manifest evaluation/gym_manifest.csv --validate-only
# After a successful orchestrated eval, use orchestration_report.scorecard_header_suggested_command
# (from run_pose_ab_eval_compare.py / run_pose_isolation_ab_compare.py) for JSONL hashes.
```

**Doc index:** [POSE_EVALUATION_PROTOCOL.md](./POSE_EVALUATION_PROTOCOL.md) (metrics) · [GYM_EVAL_CAPTURE_AND_DATA.md](./GYM_EVAL_CAPTURE_AND_DATA.md) (clips) · [RESEARCH_PLAN_POSE_AND_LTX.md](./RESEARCH_PLAN_POSE_AND_LTX.md) (LTX + literature).

---

## 1. Dependency order (unchanged, enforced)

1. **Decode / timebase** — FFmpeg normalize when available; record `ffmpeg_preprocess_applied`.
2. **Optional person isolation** — detector + tracker when multi-person or loose framing dominates errors.
3. **2D pose backbone** — MediaPipe today; additional backends behind `app.pose.backends` with identical **contract** (see §2).
4. **Temporal refinement** — smoothing / filtering on **canonical** coordinates (not vendor raw).
5. **Downstream metrics** — basketball shot math, gym gates, rep logic (consume canonical API only).
6. **Generative media (LTX-class)** — **after** tracks are stable enough to condition on; labeled synthetic.

Skipping layers or merging them in one unmeasured PR is **out of scope** for a “research-grade” change.

---

## 2. Canonical representation & mapping (mandatory before new backends)

**Problem:** MediaPipe Pose Landmarker exposes a **33-landmark** topology (indices used in code, e.g. [app/pose/constants.py](../app/pose/constants.py)). Popular 2D models (RTMPose, ViTPose) often emit **COCO 17** (or WholeBody / Halpe variants). **Index order and joint definitions differ.** Silent “plug and play” mapping causes subtle bugs (wrong side, wrong elbow, inverted axes).

**Rule:** Downstream modules should eventually consume a **canonical joint dict** only (not raw vendor indices). **Implemented (P0):**

- Types: [app/pose/canonical.py](../app/pose/canonical.py) — `CanonicalJointName` (COCO-17 **names**), `JointObservation`, `CANONICAL_JOINT_SCHEMA_VERSION`.
- MediaPipe mapping: [app/pose/mapping_mediapipe.py](../app/pose/mapping_mediapipe.py) — `map_mediapipe_blazepose33_to_canonical()` (BlazePose 33 → canonical); tests in [tests/test_canonical_mapping.py](../tests/test_canonical_mapping.py).
- Coordinates: normalized image **x, y**, **z** passthrough, **visibility** in [0,1] (MediaPipe convention).

Per joint: `x`, `y` in **normalized image coordinates** [0,1] unless a future flag adds pixel mode. Joints enumerated once cover nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles (17 joints).

**Deliverable before merging a non-MediaPipe default:**

| Deliverable | Purpose |
|-------------|---------|
| **Mapping module** | MediaPipe 33 → canonical **done**; RTMPose / ONNX head → canonical **still required** when P1 lands. |
| **Contract tests** | Synthetic landmark tensors **done**; optional: short real clip fixture later. |
| **Provenance** | JSONL includes `canonical_joint_schema_version`, `canonical_joint_set`, `canonical_mapping_id` (see [provenance.py](../app/pose/provenance.py)); future backends add `checkpoint_id`, `onnx_opset`, etc. |

**Bump** `pose_baseline_schema_version` and/or `CANONICAL_JOINT_SCHEMA_VERSION` when canonical joint set or coordinate convention changes (current baseline JSONL schema: **1.2.0**).

---

## 3. Inference modes (offline vs online)

Pick **explicitly**; they may use **different** checkpoints or input resolution.

| Mode | Purpose | Typical stack |
|------|---------|----------------|
| **Offline eval** | Manifest A/B, research claims, CI nightly | Higher input size / heavier model / GPU OK; full-frame or large ROI |
| **Online / product** | User upload latency SLO | Smaller model, ROI crop, frame stride or capped max length |

**Rule:** Numbers quoted externally must state **mode + hardware + preprocess path**. Do not tune offline-only settings and imply the app matches without measurement.

---

## 4. Preprocessing & denoise — hard gates

### 4.1 Learned or heavy denoise

**Default:** **Off** until proven.

**Gate to enable or change denoise:** A PR must include **manifest-backed** comparison on the **same** clips: before/after table of at least `detection_rate`, `pose_usable_heuristic` rate (gym), and **basketball** stability metrics (e.g. variance of key angles / event timing on repeated runs). If denoise **hurts** easy clips, it does not ship globally.

**Rationale:** Restoration models can **shift textures** and **blur** structures; keypoint networks are not guaranteed to improve.

### 4.2 Multipass / “best of” preprocess (gamma, denoise variants)

`KinematicAnalyzer` and `--multipass` eval **select** the variant that maximizes an internal utility. That can **inflate** benchmarks relative to a **single fixed** deploy pipeline.

**Rule:**

- Report **both** `single_pass` and `multipass_best` (or explicitly document **only** the mode that matches production).
- If production uses **single pass**, do not cite multipass-only numbers in product copy.

---

## 5. Labeling budget & what you are allowed to claim

| Tier | Data | Legitimate claims |
|------|------|-------------------|
| **L0** | No manual labels | “**Pipeline B vs A** on our manifest: higher detection continuity / lower jerk / stabler downstream metrics” |
| **L1** | Sparse 2D keyframes (e.g. 50–200) on wrists/elbows/shoulders for basketball subset | **PCK@δ** or per-joint error bands on those keyframes + L0 |
| **L2** | Dense or mocap-backed | Stronger biomechanical statements; still bounded by single-view limits |

**Without L1+,** avoid language like “ground truth accuracy” or “SOTA on basketball pose.” **With L0 only,** “improved robustness on internal reel” is accurate.

---

## 6. Basketball evaluation parity (mirror gym discipline)

Gym Phase A already has manifests, JSONL, provenance, and strict CLI semantics. Basketball should **not** rely on ad hoc one-off runs.

**Target state:**

| Element | Gym (today) | Basketball (target) |
|---------|-------------|---------------------|
| Manifest | `evaluation/gym_manifest.csv` | [evaluation/manifest.template.csv](../evaluation/manifest.template.csv) / `manifest.csv` already define basketball clips; optional rename later for clarity only |
| Row output | `pose_baseline.jsonl` rows + provenance | Same **pattern**: per-clip JSON with preprocess flags + backend id |
| Compare | Diff two JSONL / summaries | Script or Makefile target: **same clips**, **diff** metrics and validation flags |
| Regression | `--strict-manifest` optional | Same idea for `expect_*` once curated |

Until that exists, any backbone comparison should still **fix a file list** and store outputs under `evaluation/` (gitignored binaries, versioned manifests).

---

## 7. Implementation phases (concrete)

| Phase | Scope | Exit criterion |
|-------|--------|----------------|
| **P0** | Canonical joint enum + MediaPipe→canonical mapping + provenance fields + tests | **Done** (wire into live frame loop = P3) |
| **P1** | RTMPose via **rtmlib** on **offline** `eval_pose_baseline.py --backend rtmpose` | **P1a shipped**; **P1b** = A/B tables + optional pinned model digest |
| **P2** | Optional **person ROI** (detector + tracker) in front of both backends on **hard** subset | Measurable drop in `multiple_people_detected` failures or wrong-subject rate on labeled rows |
| **P3** | Wire **canonical** stream into basketball `KinematicAnalyzer` behind feature flag | Frozen basketball manifest: metric distribution report vs MediaPipe-only |
| **P4** | LTX spike (separate ADR + budget); conditioning from **stabilized** 2D or rendered skeleton | Human-reviewed reel; synthetic disclaimer |

**Synthetic / biomechanics finetuning** (e.g. denser keypoints, OpenCapBench-style research): **Phase P3+ or separate program** — requires dataset, training budget, and ethics review. **Not** part of P1.

---

## 8. Risk register (explicit mitigations)

| Risk | Mitigation |
|------|------------|
| **Regression on easy clips** | Always run **smoke** set + **hard** set; block merge if easy-set metrics drop beyond tolerance. |
| **License / redistribution** | Record checkpoint **license** in provenance; comply with MMPose / HF model cards. |
| **Platform variance (macOS vs Linux CI)** | Declare **reference environment** for published numbers; run ONNX on same EP when comparing backends. |
| **Multipass vs deploy mismatch** | §4.2 reporting rule. |
| **Overstating accuracy** | §5 claim tiers. |

---

## 9. Self-assessment (re-grade after revision)

| Criterion | Grade | Note |
|-----------|-------|------|
| Logical ordering of dependencies | **A** | Decode → isolate → pose → time → metrics → gen |
| Honesty about monocular limits | **A** | Canonical API does not imply full 3D truth |
| Measurability | **A−** | L1 labeling still a **team action**; plan defines tiers |
| Ops / product alignment | **A−** | Offline vs online explicit; multipass caveat explicit |
| Completeness for coding | **A** | P1a–P1b landed; **P2** ROI/tracker and **P3** KinematicAnalyzer wiring remain |

**Overall: A** — run **P1b** on your clip bank, then pursue **A+** with L1 labels + published tables.

---

## 10. Related documents

- [RESEARCH_PLAN_POSE_AND_LTX.md](./RESEARCH_PLAN_POSE_AND_LTX.md) — LTX-2.3, literature, phased strategy  
- [POSE_EVALUATION_PROTOCOL.md](./POSE_EVALUATION_PROTOCOL.md) — gym metric definitions  
- [GYM_EVAL_CAPTURE_AND_DATA.md](./GYM_EVAL_CAPTURE_AND_DATA.md) — clip capture & internal data  
- [adr/0001-phase-a-mediapipe-baseline.md](./adr/0001-phase-a-mediapipe-baseline.md) — current Phase A backbone decision  
- [VIDEO_ANALYSIS_LIMITATIONS.md](./VIDEO_ANALYSIS_LIMITATIONS.md) — user-facing failure modes  
