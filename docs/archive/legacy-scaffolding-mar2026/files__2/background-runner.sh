#!/usr/bin/env bash

# =============================================================================
# Background Agent Runner for Claude Code
# =============================================================================
#
# PURPOSE:
#   Invokes Claude Code on a schedule to perform automated tasks:
#   health checks, code review, research, and strategic suggestions.
#
# PREREQUISITES:
#   - Claude Code installed: npm install -g @anthropic-ai/claude-code
#   - ANTHROPIC_API_KEY set in environment (or via Claude Code auth)
#   - Project has CLAUDE.md (copy agent-rules.md to CLAUDE.md in project root)
#   - Project has GOALS.md filled out
#
# USAGE:
#   ./background-runner.sh <project-path> <task-type>
#
#   Task types:
#     health-check  — Run tests, linter, build; report to HEALTH_LOG.md
#     research      — Research improvements based on GOALS.md
#     review        — Review recent commits for quality/security
#     suggest       — Generate prioritized next-action recommendations
#     end-of-day    — Full summary + prompt developer to push
#
# SCHEDULING (cron examples):
#   # Health check every 2 hours during work hours
#   0 */2 9-18 * * 1-5  /path/to/background-runner.sh /path/to/project health-check
#
#   # Research once daily at 2 AM (off-peak)
#   0 2 * * *  /path/to/background-runner.sh /path/to/project research
#
#   # Code review once daily at 7 AM (before you start working)
#   0 7 * * 1-5  /path/to/background-runner.sh /path/to/project review
#
#   # Suggestions every Monday morning
#   0 8 * * 1  /path/to/background-runner.sh /path/to/project suggest
#
#   # End-of-day summary at 5:30 PM on weekdays
#   30 17 * * 1-5  /path/to/background-runner.sh /path/to/project end-of-day
#
# =============================================================================

set -euo pipefail

# --- Configuration ---
PROJECT_PATH="${1:?Usage: $0 <project-path> <task-type>}"
TASK_TYPE="${2:?Usage: $0 <project-path> <task-type>}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
LOG_DIR="${PROJECT_PATH}/.agent-logs"

# Validate project path
if [ ! -d "$PROJECT_PATH" ]; then
    echo "Error: Project path does not exist: $PROJECT_PATH"
    exit 1
fi

# Validate Claude Code is installed
if ! command -v claude &> /dev/null; then
    echo "Error: Claude Code is not installed. Run: npm install -g @anthropic-ai/claude-code"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# --- Task Prompts ---

health_check_prompt() {
    cat <<'PROMPT'
You are running in AUTOMATED MODE. Do not modify application code.

Perform a health check on this project:

1. Run the test suite. Record pass/fail counts and any failures.
2. Run the linter. Record any warnings or errors.
3. Run the build command. Record success or failure.
4. Check for any dependency vulnerabilities (if a lock file exists, look for
   known issues).
5. Check git status for uncommitted changes.

Write your findings to HEALTH_LOG.md in this format:

## Health Check — [current date and time]

**Tests**: [pass count]/[total] passing. [List failures if any]
**Linter**: [clean / N warnings / N errors]. [List critical issues if any]
**Build**: [success / failure]. [Error details if failed]
**Dependencies**: [clean / N vulnerabilities]. [List critical ones if any]
**Uncommitted changes**: [none / list of files]
**Overall status**: [HEALTHY / DEGRADED / BROKEN]
**Action required**: [none / list of things the developer should address]

If anything is BROKEN (tests fail, build fails), also write the critical findings
to URGENT.md so the developer sees them immediately.
PROMPT
}

research_prompt() {
    cat <<'PROMPT'
You are running in AUTOMATED MODE. Do not modify application code.

Read GOALS.md and RESEARCH_LOG.md (if they exist) to understand the project's
direction and what has already been researched.

Identify ONE knowledge gap or improvement opportunity that is:
- Relevant to the current milestone in GOALS.md
- Not already covered in RESEARCH_LOG.md
- Actionable (the developer could act on it within a few hours)

Conduct focused research on this topic using available tools (web search,
documentation lookup, GitHub exploration).

Append your findings to RESEARCH_LOG.md following the format specified in the
agent rules (Section 11.3).

Keep the research focused. One well-researched topic is worth more than five
shallow ones.
PROMPT
}

review_prompt() {
    cat <<'PROMPT'
You are running in AUTOMATED MODE. Do not modify application code.

Review the git log for the last 24 hours of commits (or since the last review
if REVIEW_LOG.md has a previous entry).

For each commit or group of related commits, evaluate:
1. Code quality: Is the code clear, well-structured, and maintainable?
2. Potential bugs: Are there edge cases, null checks, or error handling gaps?
3. Security: Are there any secrets, SQL injection vectors, XSS risks, or
   other vulnerabilities?
4. Performance: Are there obvious performance issues (N+1 queries, unnecessary
   re-renders, missing indexes)?
5. Test coverage: Were tests added or updated for the changes?

Write your findings to REVIEW_LOG.md in this format:

## Code Review — [current date and time]

**Commits reviewed**: [hash range or count]
**Overall quality**: [Good / Acceptable / Needs attention]

**Findings**:
- [SEVERITY: low/medium/high] [file:line] [description of issue]
- ...

**Positive notes**: [things done well — reinforcement matters]
**Recommendations**: [prioritized list of improvements]

Be fair. If the code is good, say so. Do not manufacture issues.
PROMPT
}

suggest_prompt() {
    cat <<'PROMPT'
You are running in AUTOMATED MODE. Do not modify application code.

Read GOALS.md, RESEARCH_LOG.md, REVIEW_LOG.md, HEALTH_LOG.md, and the
current state of the codebase.

Produce a prioritized list of recommended next actions, written to SUGGESTIONS.md:

## Suggested Next Actions — [current date and time]

### High Priority (unblocks current milestone)
1. [action] — [why] — [estimated effort: S/M/L]

### Medium Priority (improves quality or velocity)
1. [action] — [why] — [estimated effort: S/M/L]

### Low Priority (nice to have, no urgency)
1. [action] — [why] — [estimated effort: S/M/L]

### Strategic Notes
[Any observations about project direction, risks, or opportunities that
don't fit into a single action item]

Base priorities on:
1. What directly unblocks the current milestone in GOALS.md
2. What reduces risk (bugs, security, tech debt that compounds)
3. What improves developer velocity
4. What improves end-user experience
PROMPT
}

end_of_day_prompt() {
    cat <<'PROMPT'
You are running in AUTOMATED MODE. Do not modify application code.

Generate an end-of-day summary. This will be the last thing the developer sees
before deciding whether to push changes.

1. Run `git status` and `git diff --stat` to see all uncommitted changes.
2. Run `git log --oneline` for today's commits.
3. Run tests to confirm current state is clean.

Write to END_OF_DAY.md (overwrite previous — this is always "today's" summary):

## End of Day Summary — [current date]

### Today's Commits
[List of commits with one-line descriptions]

### Uncommitted Changes
[List of modified files and what they contain]

### Test Status
[All passing / N failures — details]

### Build Status
[Clean / Broken — details]

### Ready to Push?
[YES — all clean / NO — here is what needs attention first]

### What to Push
```
[Exact git commands the developer should run, e.g.:]
git add -A
git commit -m "feat(dashboard): add date range filter"
git push origin feat/dashboard-filters
```

### Tomorrow's Suggested Starting Point
[Based on GOALS.md and today's work, what should the developer tackle first
tomorrow?]

IMPORTANT: End with a clear call to action. The developer should be able to
glance at this file and know exactly whether to push and what to do next.
PROMPT
}

# --- Execute Task ---

echo "[$TIMESTAMP] Running task: $TASK_TYPE on $PROJECT_PATH"

cd "$PROJECT_PATH"

case "$TASK_TYPE" in
    health-check)
        PROMPT=$(health_check_prompt)
        ;;
    research)
        PROMPT=$(research_prompt)
        ;;
    review)
        PROMPT=$(review_prompt)
        ;;
    suggest)
        PROMPT=$(suggest_prompt)
        ;;
    end-of-day)
        PROMPT=$(end_of_day_prompt)
        ;;
    *)
        echo "Error: Unknown task type: $TASK_TYPE"
        echo "Valid types: health-check, research, review, suggest, end-of-day"
        exit 1
        ;;
esac

# Run Claude Code with the task prompt
# --print: output to stdout (non-interactive)
# --max-turns: limit autonomous iterations to prevent runaway sessions
claude --print --max-turns 10 "$PROMPT" 2>&1 | tee "${LOG_DIR}/${TASK_TYPE}_${TIMESTAMP//[: ]/_}.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -ne 0 ]; then
    echo "[$TIMESTAMP] Task $TASK_TYPE failed with exit code $EXIT_CODE"
    echo "[$TIMESTAMP] FAILED: $TASK_TYPE (exit $EXIT_CODE)" >> "${LOG_DIR}/failures.log"
else
    echo "[$TIMESTAMP] Task $TASK_TYPE completed successfully"
fi

exit $EXIT_CODE
