# 60-Second Live Demo Runbook

## Setup (do this BEFORE the showcase, not during)

```bash
cd /Users/parvpatodia/Laksh_AI
source .venv/bin/activate          # or: PATH=".venv/bin:$PATH"

# Verify the fixture exists
ls -lh evaluation/fixtures/demo_squat_frames.json

# Dry run: confirm it works, check output looks right
python scripts/analyze_gym_clip.py \
  --exercise-id back_squat \
  --frames-json evaluation/fixtures/demo_squat_frames.json \
  --pretty 2>&1 | head -60
```

Make sure you see `"n_reps"` > 0 in the summary. If you see 0 reps,
run `make test-pose-core` to confirm segmenter is healthy.

---

## Terminal setup for the showcase

- Font size: **18pt minimum**. Judges will be standing 1-2 meters away.
- Dark background (iTerm2 / Terminal dark theme).
- Run `clear` right before you step up.
- Have exactly one terminal window open. One tab. Nothing else visible.

Pre-stage the command (type it, don't hit Enter yet):

```
python scripts/analyze_gym_clip.py --exercise-id back_squat --frames-json evaluation/fixtures/demo_squat_frames.json --pretty
```

---

## The 60-second sequence

### T+0:00 - Say the setup
> "Here is the full pipeline running end-to-end, no edits, no shortcuts."

Show the command already typed in the terminal. Read it aloud:
> "Exercise: back squat. Input: pre-extracted pose frames.
>  This is the --frames-json flag - same result every time, no network, no GPU."

### T+0:05 - Hit Enter

*The script runs. On M-series Mac with .venv it takes < 2 seconds.*

### T+0:08 - Point to the top of the output
> "Two reps detected."

Point to `"reps": [...]` in the segment block - two entries.

### T+0:15 - Scroll to feature_vectors, point to first rep
> "Each rep has seven measured fields. Here - rep_duration_s, 2.0 seconds.
>  Status: valid. The segmenter was confident about this rep's boundaries."

Point to:
```json
"rep_duration_s": {
  "value": 2.0,
  "unit": "s",
  "status": "valid",
  "reason_codes": []
}
```

### T+0:25 - Point to tempo_ratio
> "Tempo ratio: how long the eccentric phase took vs the concentric.
>  1.7 here - the athlete controlled the descent."

### T+0:35 - Scroll to calibration block
> "Now here is the part I want you to look at."

Point to:
```json
"status": "no_reference_yet",
"range": null,
"evidence_status": "uncalibrated_v0"
```

> "The system measured 1.7. It does NOT say 1.7 is good.
>  Because we haven't labeled enough clips to know what good looks like
>  for this movement yet. That honesty is deliberate. It's in the schema."

### T+0:50 - Step back
> "When we have the labeled data, that 'no_reference_yet' becomes
>  'within_reference' or 'outside_reference' with a cited source.
>  Until then: the number is real. The verdict is not."

### T+1:00 - Done

---

## Fallback plan (if anything goes wrong)

**Script errors / import fails:**
```bash
# Check venv is active
which python   # should be .venv/bin/python

# Reinstall deps if needed
uv sync
```

**Output looks garbled / no reps:**
```bash
# Run the test suite to confirm pipeline is healthy
PATH=".venv/bin:$PATH" make test-pose-core
```

**Judges want to see a real video:**
> "The --video flag runs the same pipeline on any MP4 with MediaPipe.
>  The fixture here is deterministic for the demo so nothing changes
>  between rehearsal and presentation."

**"What if the model is wrong?"**
> "That is exactly why every field has a status. If MediaPipe gives us
>  bad landmarks, the missingness fraction rises and the rep gets flagged
>  degraded - not silently passed through as if it were good."

---

## Commands reference card (print this, keep in pocket)

```bash
# Main demo command
python scripts/analyze_gym_clip.py \
  --exercise-id back_squat \
  --frames-json evaluation/fixtures/demo_squat_frames.json \
  --pretty

# Alternative exercises (same fixture works for any cyclic_vertical)
python scripts/analyze_gym_clip.py --exercise-id front_squat \
  --frames-json evaluation/fixtures/demo_squat_frames.json --pretty

# Show calibration config is all honest uncalibrated
make verify-calibration-v0

# Show exercise taxonomy is SHA-pinned
make verify-exercise-v0

# Run full test suite
make test-pose-core
```
