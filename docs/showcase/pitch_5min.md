# 5-Minute Pitch Script - Laksh.ai
## Research Showcase - Northeastern University

**Total time: 5:00**
**Pace guide: ~130 words/min. Practice until each block lands on its timestamp.**

---

### [0:00 - 0:30] HOOK

"Think about the last time you went to the gym.
Maybe you squatted. Maybe you benched.
And unless you had a coach standing right there, you had no idea
whether your form was actually good - or just felt good.

Coaches don't scale. One good coach, maybe 10 athletes.
One billion gym members. The math doesn't work.

So what if your phone could watch your rep and tell you something useful?"

*(pause 1 second)*

---

### [0:30 - 1:00] THE HARD PROBLEM

"There are apps that try to do this. Most of them have a dirty secret:
they hardcode ideal angles. 'Your knee should be at 90 degrees.'
Based on what? Which population? What evidence?

The number is invented. The system looks confident. The user trusts it.
That is not AI. That is theater.

Laksh.ai starts from a different place.
We build the measurement infrastructure first - and we are honest about
what we haven't measured yet."

---

### [1:00 - 1:45] THE SYSTEM - WHAT WE BUILT

"We built what I call the measurement spine. Four layers.

Layer one: a frozen exercise taxonomy. Twelve compound movements -
squat, deadlift, bench press, pull-up, plank - each one defines
what joint to watch and how to read a rep signal. No ideal angles.
Just: here is what this movement looks like mechanically.

Layer two: a rep segmenter. It takes a video, runs MediaPipe to get
skeleton landmarks, and detects where each rep starts and ends.
Deterministic - same video, same result, every time.

Layer three: a per-rep feature vector. Seven measurements per rep.
Total duration. Eccentric phase. Concentric phase. Tempo ratio.
Range of motion. Visibility quality. Missingness fraction.
Every field carries its own status: valid, degraded, or unknown.

Layer four: a calibration config. This is the honest part."

---

### [1:45 - 2:15] THE HONEST PART

"The calibration config is a versioned JSON file. Right now, every
entry says 'uncalibrated_v0' with empty reference ranges.

Why would we ship that? Because the alternative is worse.
The alternative is making up numbers.

And I've made this policy impossible to accidentally violate:
the validator literally rejects a config entry that has
'uncalibrated_v0' and non-empty ranges in the same entry.
You cannot silently hardcode an ideal band and call it calibrated.
The code won't let you.

When we have labeled data - when a biomechanist has reviewed
clips and said this tempo ratio range is good for a back squat -
that entry flips to 'cited' with a pointer to the eval run
that justified the numbers. No evidence, no range. That's the contract."

*(pause)*

---

### [2:15 - 3:15] LIVE DEMO

*[Switch to terminal]*

"Let me show you the whole pipeline in under 30 seconds."

```
python scripts/analyze_gym_clip.py \
  --exercise-id back_squat \
  --frames-json evaluation/fixtures/demo_squat_frames.json \
  --pretty
```

*[While it runs, narrate:]*
"This is taking pre-extracted pose frames - a synthetic squat - through
the full stack. Segmenter finds the reps. Feature extractor measures
each one. Calibration layer looks up any reference ranges."

*[When output appears, point to:]*

"Two reps detected. Each one has a rep_duration_s - valid.
tempo_ratio_ecc_over_con - valid, value is 2.0, meaning the athlete
took twice as long to lower as to lift. Classic eccentric control.

And here -" *(point to calibration block)*
"- status: no_reference_yet. The system knows its number. It does not
pretend to know if that number is good. That is the right posture."

---

### [3:15 - 3:45] WHY THIS MATTERS FOR RESEARCH

"This architecture is interesting for three reasons.

First: the per-field status taxonomy - valid, degraded, unknown -
mirrors how we'd want any scientific measurement to behave.
A degraded measurement is still informative. An unknown is not zero.

Second: the calibration contract decouples measurement from evaluation.
You can deploy the measurement spine now and add reference ranges later
when you have the evidence. Most systems conflate these two things.

Third: 187 unit tests, all deterministic, no GPU required for the fast
subset. The entire measurement spine runs in nine seconds on my laptop.
That matters for reproducibility."

---

### [3:45 - 4:30] TECHNICAL DEPTH (for technical judges)

"The rep segmenter uses scipy.signal.find_peaks on a 1D signal extracted
from the skeleton. For a squat, that signal is the hip's y-coordinate.
For a bench press, it's the elbow flexion angle computed from
a shoulder-elbow-wrist triplet. Duration holds like planks get a
stability proxy instead of cyclic peaks - the variance of a body-midline
signal over time.

The feature extractor tolerates both in-memory pose output -
enum keys, frozen dataclasses - and serialized JSON -
string keys, plain dicts - without a conversion step at every call site.
That matters when you're building evaluation pipelines that process
thousands of clips from disk.

The parity gate - ADR 0002 Phase C - uses a Tukey fence p90 outlier
detector to decide when the canonical joint path is production-ready
to replace the legacy coordinate path in KinematicAnalyzer."

---

### [4:30 - 5:00] CLOSE

"So what's next?

Labeled rep boundary annotations - IoU and F1 to measure whether the
segmenter actually finds the right frames. Once we have those,
the calibration entries graduate from 'uncalibrated_v0' to 'cited',
and the system starts giving real feedback grounded in real evidence.

Laksh.ai is not trying to replace a coach.
It's trying to be the thing between the camera and the coach -
a measurement layer that is honest, testable, and reproducible.

The code is on GitHub. The poster has the architecture.
I'm happy to go deep on any of the four layers."

*(step back, smile, stop)*

---

## TIMING REFERENCE

| Segment | Duration | Cumulative |
|---|---|---|
| Hook | 0:30 | 0:30 |
| Hard problem | 0:30 | 1:00 |
| System overview | 0:45 | 1:45 |
| Honest calibration | 0:30 | 2:15 |
| Live demo | 1:00 | 3:15 |
| Research contribution | 0:30 | 3:45 |
| Technical depth | 0:45 | 4:30 |
| Close | 0:30 | 5:00 |

## REHEARSAL NOTES

- Slow down on "valid, degraded, unknown" - say each word separately.
- Do not read the JSON output. Point to specific keys and narrate.
- "That is not AI. That is theater." - pause after this line.
- "The code won't let you." - let it land, then continue.
- Practice the demo command until you can type it without looking.
- If the demo fails: "This is why we have the --frames-json flag -
  reproducibility first." (it will not fail; the fixture is deterministic)
