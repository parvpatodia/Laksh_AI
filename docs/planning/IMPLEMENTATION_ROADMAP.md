# Laksh.ai Implementation Roadmap — Reliability, Personalization & Sports-Agnostic

> **PhD / Research Scientist approach**: Build foundations for honesty, trust, and generalization. Every user—any sport, any level—gets genuine, actionable feedback.

---

## Phase 1: Foundations (Implement Now)

| # | Feature | Purpose | Status |
|---|---------|---------|--------|
| 1 | **Clip selection** | User chooses which part of video to analyze → multi-shot, long videos | Done |
| 2 | **Recording best-practices** | Onboarding tips before analysis → better input quality | Done |
| 3 | **Confidence explanation** | "Why is my confidence X?" with actionable reasons | Done |
| 4 | **Actionable validation** | Each warning has a clear "How to fix" | Done |
| 5 | **Sport selection** | Basketball first; structure for tennis, golf, etc. | Done |
| 6 | **Athlete profile** | Name, sport, level → personalized feedback tone | Done |
| 7 | **Video recommendations** | "For better accuracy: record 45° angle, single person" | Done |

---

## Phase 2: Sports-Agnostic Foundation

| # | Feature | Purpose |
|---|---------|---------|
| 8 | **Sport config module** | Event names, key joints, metrics schema per sport |
| 9 | **Basketball config** | Current logic as first sport |
| 10 | **Placeholder configs** | Tennis, Golf, Soccer (stub for future) |

---

## Phase 3: Advanced Accuracy

| # | Feature | Purpose |
|---|---------|---------|
| 11 | **Ball tracking** | YOLO → true release velocity, arc |
| 12 | **RTMPose option** | Higher-accuracy pose when GPU available |
| 13 | **Uncertainty ranges** | ± on metrics where estimable |

---

## Phase 4: Video Recommendations & Drills

| # | Feature | Purpose |
|---|---------|---------|
| 14 | **Recording tips** | Contextual based on validation (e.g. "Use landscape") |
| 15 | **Drill suggestions** | "Improve knee flexion: wall sits, leg press" |
| 16 | **Export annotated video** | Share with coach |

---

## Design Principles

1. **Honest first** — Never overstate accuracy. Show confidence and limitations.
2. **Actionable** — Every warning or tip tells the user what to do.
3. **Personalized** — Feedback tone adapts to sport and level.
4. **Extensible** — Sport configs let us add tennis, golf without rewriting core.
