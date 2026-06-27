# Skill: Refactoring

## When to Activate

Activate when:
- Developer explicitly asks for refactoring
- Code smells are identified during a review (duplication, long functions,
  deep nesting, god objects)
- A feature is hard to implement because the existing structure resists it
- Tech debt items in GOALS.md or TODO.md reference structural improvements
- After a bug fix reveals that the surrounding code is fragile

NEVER activate during a bug fix. Fix first, then refactor in a separate step.

## Core Principles

1. **Refactoring changes structure, not behavior.** If tests pass before and
   after, you refactored correctly. If behavior changed, you did something
   else — call it what it is.

2. **Tests are your safety net.** Never refactor code that has no tests.
   Write tests first (characterization tests that capture current behavior),
   then refactor. If you refactor without tests, you are just editing.

3. **Small steps, always green.** Each refactoring step should keep the code
   in a working state. If you refactor for an hour and the code is broken the
   entire time, you've lost the ability to isolate which change broke it.
   Commit after each step.

4. **Name the refactoring.** Martin Fowler catalogued refactorings for a
   reason. If you can name what you're doing (Extract Method, Move Field,
   Replace Conditional with Polymorphism), you can explain it, reason about
   its safety, and teach it. If you can't name it, you're probably doing
   something riskier than you think.

5. **Wait for the third instance.** The Rule of Three: the first time you see
   duplication, note it. The second time, note it again. The third time,
   refactor. Abstracting after the first instance leads to premature
   abstractions that don't generalize correctly.

## Patterns and Techniques

### Safe Refactoring Sequence

1. **Verify tests exist** for the code you're about to touch. If not, write
   characterization tests first.
2. **Run tests** to confirm they pass (green baseline).
3. **Make one structural change** (extract, move, rename, inline).
4. **Run tests** again. Green? Commit. Red? Revert the step, investigate.
5. **Repeat** until the target structure is reached.

### Common Refactorings

**Extract Function**: A block of code inside a larger function does one
identifiable thing. Pull it into its own function with a descriptive name.
The original function becomes a sequence of named steps — readable like prose.

**Replace Magic Values with Named Constants**: `if (status === 3)` → 
`if (status === ORDER_STATUS.SHIPPED)`. The code now documents itself.

**Simplify Conditionals**: Deep nesting (`if > if > if > if`) is a readability
killer. Techniques: early returns (guard clauses), extracting condition logic
into named boolean variables, replacing conditional chains with lookup
tables or polymorphism.

**Decompose God Objects**: A class or module doing 15 things needs to be
split into cohesive pieces. Identify clusters of methods that use the same
subset of data — each cluster is a candidate for extraction into its own
module.

**Replace Inheritance with Composition**: If a subclass overrides most of
its parent's behavior, the inheritance relationship is misleading. Extract
the shared behavior into a composable piece (a function, a mixin, a
strategy object).

### When NOT to Refactor

- **The code works and will not be touched again.** Refactoring code that
  is stable, correct, and not in an active development path is waste.
- **Under time pressure for a deadline.** Refactoring introduces risk.
  If you're shipping tomorrow, ship the working code and refactor next sprint.
- **Without tests.** Full stop. Write tests first or accept the risk explicitly.
- **Across a major version boundary.** If you're about to upgrade a framework
  or language version, refactor after the upgrade, not before — the upgrade
  may invalidate your refactoring choices.

## Checklist

Before considering a refactoring complete:

- [ ] Tests existed before starting (or characterization tests were written)
- [ ] All tests still pass after refactoring
- [ ] No behavioral changes were introduced (check git diff carefully)
- [ ] The refactoring can be named (Extract Method, Move Module, etc.)
- [ ] Each intermediate step was committed separately
- [ ] The code is genuinely more readable or more maintainable (not just different)
- [ ] No new abstractions were introduced that serve only one use case

## Anti-Patterns

- **Refactoring and adding features simultaneously**: Two concerns, two PRs.
  Mixing them makes both harder to review and debug.
- **Abstracting for hypothetical future requirements**: "We might need this
  to support multiple databases someday" — unless that's in GOALS.md, don't
  build the abstraction.
- **Renaming everything to be "cleaner"**: Mass renames create huge diffs that
  are hard to review and trigger merge conflicts for everyone. Rename
  strategically, not compulsively.
- **Moving code between files without reason**: Reorganizing file structure
  is a valid refactoring, but only when the current structure causes real
  confusion or violates a clear convention. "I prefer a different folder
  layout" is not sufficient justification.

## Learning Notes

**For the developer**: The best refactoring book is still Martin Fowler's
"Refactoring" (2nd edition, 2018). The key insight is that refactoring is
not a phase — it's a continuous practice woven into daily development.
You don't schedule "refactoring sprints." You refactor the code you're
touching, as you touch it, in small steps. The codebase gets cleaner
as a side effect of working in it.
