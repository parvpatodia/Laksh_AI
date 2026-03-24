# Golden Video Guide — Regression Testing for Laksh.ai

> **Purpose**: Create a reference video for CI regression tests. Same video → same metrics = pipeline stability. Also useful for demo consistency and investor validation.

---

## What Is a Golden Video?

A **golden video** is a short, high-quality clip of a single jump shot that we run through the physics engine on every deploy. If the output metrics stay within expected ranges, the pipeline is stable. If they drift, something broke.

---

## How to Create One

### 1. Recording Setup

| Requirement | Why |
|-------------|-----|
| **Single shooter** | Multi-person triggers confidence penalty; golden test should be deterministic |
| **45° front-offset** | Matches recommended recording angle; minimizes arc distortion |
| **Landscape, 720p+** | Our validation flags warn on low res |
| **30 fps or higher** | Low FPS blurs fast motions |
| **Good lighting** | Avoid backlit; pose visibility must be >50% |
| **Full body visible** | Dip → release → follow-through in frame |

### 2. Content

- **1–2 clean jump shots** (free throw or mid-range)
- **5–15 seconds total** — long enough for event detection, short enough for fast CI
- **No cuts** — single continuous take

### 3. Options for Sourcing

| Source | Effort | Quality | Notes |
|--------|--------|---------|-------|
| **Record yourself** | Low | Good | Use phone on tripod; follow setup above |
| **Teammate/friend** | Low | Good | Same setup; one person in frame |
| **Public domain / CC footage** | Medium | Variable | Ensure license allows use; may need to trim to one shot |
| **Stock video site** | Medium | High | Search "basketball shot side view" or "jump shot"; verify license |
| **Lab / mocap session** | High | Best | If you partner with a sports science lab for validation |

### 4. Quick Recording Checklist

1. Phone on tripod or stable surface
2. Record from 45° angle (not pure side, not straight-on)
3. One person, one shot, full body
4. Export as MP4 (H.264), 720p or 1080p, 30fps
5. Trim to 5–15 seconds if the original is longer

---

## Where to Put It

```
tests/
  fixtures/
    golden_shot.mp4   # Your reference video (add this file)
    .gitkeep         # Keeps fixtures/ in git when folder is empty
```

**Git**: Add `tests/fixtures/golden_shot.mp4` to the repo if it's small (<5 MB). For larger files, use Git LFS or keep it local and document the path. The regression test will **skip** if the file is missing.

---

## Expected Metrics (Tolerance Ranges)

Once you have a golden video, run the analysis once and record the output. Use those values as the baseline. Example tolerances:

| Metric | Typical Range | Tolerance |
|--------|---------------|-----------|
| release_velocity_mps | 5–9 | ±0.5 |
| shot_arc_deg | 35–65 | ±5 |
| knee_angle | 130–175 | ±8 |
| elbow_angle | 150–180 | ±8 |
| kinetic_sync_ms | 100–400 | ±30 |
| fluidity_score | 50–99 | ±10 |
| hip_rotation_deg | -20 to 20 | ±5 |
| balance_index | 60–99 | ±10 |

**First run**: Analyze the video (via API or a small script). Create `tests/fixtures/golden_expected.json`:

```json
{
  "baseline": {
    "release_velocity_mps": 7.2,
    "shot_arc_deg": 48.5,
    "knee_angle": 152.0,
    "elbow_angle": 168.0,
    "kinetic_sync_ms": 220.0,
    "hip_rotation_deg": 4.2,
    "balance_index": 85,
    "fluidity_score": 72
  },
  "tolerances": {
    "release_velocity_mps": 0.5,
    "shot_arc_deg": 5,
    "knee_angle": 8,
    "elbow_angle": 8,
    "kinetic_sync_ms": 30,
    "hip_rotation_deg": 5,
    "balance_index": 10,
    "fluidity_score": 10
  }
}
```

**How to capture baseline**: Run the app locally, upload your golden video via the dashboard, then copy the `stats` section from the API response. Add tolerances from the table above.

---

## Running the Regression Test

```bash
# With golden video in place
pytest tests/test_physics_regression.py -v

# Without golden video (test is skipped)
pytest tests/test_physics_regression.py -v
# SKIPPED - golden video not found
```

---

## CI Integration (Railway / GitHub Actions)

Once the golden video and expected values are committed:

```yaml
# .github/workflows/regression.yml (example)
- name: Run regression test
  run: pytest tests/test_physics_regression.py -v
```

This runs on every push, ensuring changes to `app/physics_engine.py` or `app/main.py` don't break the pipeline.

---

## Long-Term Use

- **Hackathons**: Same demo video → consistent, repeatable pitch
- **Investors**: "We run every deploy against a validated reference; metrics are reproducible"
- **Sports academies**: Reference clip for calibration; compare athlete clips to same baseline
