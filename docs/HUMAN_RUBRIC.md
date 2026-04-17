# Human coaching rubric (blinded)

**Purpose.** F1 / IoU are necessary but not sufficient for coaching quality.
A rubric over 20-30 fixed clips catches the gap between "numerically
correct" and "a coach would say this." Versioned, blinded, re-runnable.

Roadmap reference:
[product-grade_laksh_roadmap](product-grade_laksh_roadmap_05e7df02.plan.md)
Phase D -- "Human evaluation for coaching".

## When to run

- Before every release that changes **metrics**, **thresholds**, **LLM
  prompts**, or **narrative templates**.
- When a new backbone graduates out of shadow (ADR 0002 / 0003).
- Never more than once a week per clip set -- ratings drift with memory if
  reviewers see the same clip and same output repeatedly.

## Clip set

- **20-30 clips** fixed for a rubric version. Add clips to a new rubric
  version (`v2`, `v3`, ...); do not mutate `v1`.
- Cover: clean side view, clean front view, multi-person, partial
  occlusion, one poor-lighting, one VFR / rotation-mismatch.
- Store the clip set IDs in
  [evaluation/gym_manifest.csv](../evaluation/gym_manifest.csv) with
  `tags` including `rubric_v1` (or current version).

## Blinding

- Reviewer sees: **clip + rendered feedback text + rendered overlay**.
- Reviewer does NOT see: backbone name, prompt version, git commit,
  predicted confidence, whether this is candidate vs baseline.
- Two systems are compared side-by-side with labels `A` / `B` randomized
  per clip by a secret map held by the release manager.

## Rubric (per clip, per system)

Score each axis 1-5. Anchors are explicit so scores are comparable across
reviewers.

| Axis | 1 | 3 | 5 |
|---|---|---|---|
| **Clarity** | Reader re-reads to parse | Clear after one read, some jargon | Immediately actionable; no jargon |
| **Correctness** | Contradicts visible motion | Mostly right; one minor error | All claims verifiable in overlay / metrics |
| **Grounding** | Invents numbers / joints | Cites metrics but not linked to text | Every claim maps to a JSON field the system computed |
| **Actionability** | Vague ("work on form") | One generic cue | 1-3 specific cues with what/when/why |
| **Tone** | Patronising or alarming | Neutral | Encouraging without inflating |

**Total (per clip, per system):** sum of five axes, 5-25.

## Automatic grounding check (pair with the human rubric)

Before a reviewer opens the clip, a script must confirm every numeric claim
in the narrative text maps to a computed field in the analysis JSON. If not,
the clip is flagged for rewrite -- the reviewer's time is not spent debating
a hallucination.

Proposed check path (future, not in this doc): `scripts/grounding_check.py`
parses the narrative for numbers + joint names, confirms each appears in
the emitted `analysis.*` fields (with tolerance). Hallucination rate =
unmatched claims / total claims. Target: <5% on L1.

## Reporting

For each rubric run store:

- Rubric version + clip set hash
- Reviewer ID (pseudonymised if external)
- Per-clip, per-system table of 5 axes
- Paired differences (`system_A_total - system_B_total` per clip)
- Wilcoxon signed-rank p-value if n >= 20

Attach the report filename to the scorecard header (see
`scripts/build_scorecard.py`) when reporting claim tier changes.

## Non-goals of this rubric

- Not a replacement for rep F1 / IoU. Those catch segmentation bugs the
  rubric will NOT see.
- Not a proxy for user satisfaction. Users rate the product; reviewers
  rate the coaching text.
- Not self-grading -- LLMs should not rate their own output against this
  rubric. Use them for the grounding check only.
