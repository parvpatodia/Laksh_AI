# Validation

## Why this document exists

The pytest regression test (`tests/test_physics_regression.py`) compares the
analyzer's output against a **golden value generated from the analyzer's own
prior output**. That proves the pipeline has not *drifted* (self-consistency).
It does **not** prove the numbers are *accurate* against real-world ground truth.

Accurate markerless biomechanics normally needs multiple calibrated cameras
(e.g. [OpenCap](https://www.opencap.ai/), peer-reviewed at ~3.85° joint-angle
MAE with two synchronized phones). This project uses a single uncalibrated phone,
so its numbers are coarse by construction. This document records the one honest
accuracy check a solo project can run: **hand-labelled concurrent validity** on a
small set of frames.

## What is and isn't validated

| Metric | Validatable here? | Notes |
|---|---|---|
| `knee_angle`, `elbow_angle` | **Yes** | Real 3-point goniometry; measurable by hand on a clean side view. |
| `shot_arc_deg` | Partially | Wrist-trajectory proxy, not ball trajectory. |
| `release_velocity_mps` ("Release Power") | **No** | A 2D pixel-ratio proxy with no physical ground truth — not a true velocity. |
| `hip_rotation_deg` | **No** | Depth-axis (yaw) from one camera; low confidence, treated as noise. |
| `kinetic_sync_ms`, `fluidity_score`, `balance_index` | **No** | Proxies without a hand-measurable ground truth. |

## Protocol

1. Choose ~5–10 shot frames with the relevant joint clearly visible (clean side view).
2. Hand-measure the true joint angle (shoulder–elbow–wrist for elbow, hip–knee–ankle
   for knee) with ImageJ, Kinovea, or a printed protractor.
3. Run the analyzer on the same clips; read its reported angle for each frame.
4. Record both in a CSV: `clip,frame,joint,manual_deg,predicted_deg`.
5. Run:

   ```bash
   python evaluation/manual_validation.py evaluation/validation_labels.csv
   ```

## Results

> Fill this in after running the script. Example layout:

| Joint | N frames | MAE (°) | Max abs err (°) |
|---|---|---|---|
| elbow | _TBD_ | _TBD_ | _TBD_ |
| knee  | _TBD_ | _TBD_ | _TBD_ |

**Honest caveats:** small sample; single-camera side view; results apply only to
clean, well-framed clips. Knee/elbow are the trustworthy joints; release power and
hip rotation are explicitly *not* accuracy-validated and are labelled as proxies
in the product.
