#!/usr/bin/env bash

# =============================================================================
# Sub-Agent Orchestrator for Claude Code
# =============================================================================
#
# PURPOSE:
#   Orchestrates multi-step tasks by chaining specialized Claude Code invocations.
#   Each "sub-agent" is a Claude Code call with a focused prompt and specific
#   skill context. Output from one step feeds into the next.
#
# ARCHITECTURE:
#   This is NOT true multi-agent parallelism. It is sequential task decomposition:
#
#   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
#   │ Planner  │ →  │ Worker 1 │ →  │ Worker N │ →  │ Reviewer │
#   │ (Claude) │    │ (Claude) │    │ (Claude) │    │ (Claude) │
#   └──────────┘    └──────────┘    └──────────┘    └──────────┘
#       Plan          Execute         Execute         Validate
#   (read-only)    (may edit code) (may edit code)   (read-only)
#
#   Each box is a separate Claude Code invocation with its own context window.
#   The Planner produces a structured plan. Workers execute one step each.
#   The Reviewer validates the combined result. Outputs chain through files.
#
# WHY SEQUENTIAL AND NOT PARALLEL:
#   Code changes are not parallelizable in most cases — step 2 depends on
#   what step 1 wrote. Parallel agents editing the same files produce conflicts.
#   Sequential execution with shared filesystem state is safer and debuggable.
#
# USAGE:
#   ./orchestrator.sh <project-path> <task-description>
#
# EXAMPLES:
#   ./orchestrator.sh . "Add user authentication with JWT, including signup,
#     login, logout, and protected route middleware"
#
#   ./orchestrator.sh . "Refactor the payment module to support multiple
#     payment providers (Stripe and PayPal)"
#
#   ./orchestrator.sh . "Perform a comprehensive security audit and fix
#     all critical issues"
#
# =============================================================================

set -euo pipefail

PROJECT_PATH="${1:?Usage: $0 <project-path> <task-description>}"
TASK="${2:?Usage: $0 <project-path> <task-description>}"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
WORK_DIR="${PROJECT_PATH}/.agent-work/${TIMESTAMP}"

# Validate
if [ ! -d "$PROJECT_PATH" ]; then
    echo "Error: Project path does not exist: $PROJECT_PATH"
    exit 1
fi

if ! command -v claude &> /dev/null; then
    echo "Error: Claude Code is not installed."
    exit 1
fi

cd "$PROJECT_PATH"
mkdir -p "$WORK_DIR"

echo "============================================"
echo "  Sub-Agent Orchestrator"
echo "  Task: ${TASK:0:80}..."
echo "  Work dir: $WORK_DIR"
echo "============================================"
echo ""

# ===========================================================================
# PHASE 1: PLANNER
# ===========================================================================
# The planner reads the codebase and task, produces a structured execution plan.
# It DOES NOT modify any code. Output is a JSON-ish plan file.

echo "[Phase 1/3] Planning..."

PLANNER_PROMPT=$(cat <<PROMPT
You are the PLANNER agent. You do NOT modify code. You produce a plan.

TASK: ${TASK}

Instructions:
1. Read the codebase structure and relevant files to understand current state.
2. Read GOALS.md and PROJECT_CONTEXT.md if they exist.
3. Break the task into 2-5 sequential steps. Each step should be:
   - Small enough to complete in one focused session
   - Independent enough that it can be verified before moving to the next step
   - Ordered so that later steps can build on earlier ones
4. For each step, identify:
   - What files will be touched
   - What the step accomplishes
   - How to verify the step is complete (specific test or check)
   - Which skill is most relevant (testing, security, performance, refactoring, or general)

Write your plan to ${WORK_DIR}/plan.md in this exact format:

# Execution Plan

## Task
${TASK}

## Current State Assessment
[2-3 sentences on what exists now relevant to this task]

## Steps

### Step 1: [title]
- **Files**: [list of files to touch]
- **Action**: [what to do]
- **Verify**: [how to confirm it worked]
- **Skill**: [testing/security/performance/refactoring/general]

### Step 2: [title]
...

## Risks
[Anything that could go wrong or needs developer attention]

## Estimated Total Effort
[S/M/L/XL]

Do NOT write any code. Only produce the plan file.
PROMPT
)

claude --print --max-turns 10 "$PLANNER_PROMPT" 2>&1 | tee "${WORK_DIR}/planner.log"

if [ ! -f "${WORK_DIR}/plan.md" ]; then
    echo ""
    echo "ERROR: Planner did not produce a plan file."
    echo "Check ${WORK_DIR}/planner.log for details."
    exit 1
fi

echo ""
echo "[Phase 1/3] Plan created: ${WORK_DIR}/plan.md"
echo ""
cat "${WORK_DIR}/plan.md"
echo ""

# ===========================================================================
# PHASE 2: WORKER EXECUTION
# ===========================================================================
# Execute each step from the plan. Each step is a separate Claude Code call
# with the plan as context.

echo "============================================"
echo "[Phase 2/3] Executing steps..."
echo "============================================"
echo ""

# Count steps in the plan
STEP_COUNT=$(grep -c "^### Step" "${WORK_DIR}/plan.md" || echo "0")

if [ "$STEP_COUNT" -eq 0 ]; then
    echo "ERROR: No steps found in plan."
    exit 1
fi

echo "Found $STEP_COUNT steps to execute."
echo ""

for ((i=1; i<=STEP_COUNT; i++)); do
    echo "--------------------------------------------"
    echo "[Step $i/$STEP_COUNT] Executing..."
    echo "--------------------------------------------"

    WORKER_PROMPT=$(cat <<PROMPT
You are a WORKER agent executing step $i of a ${STEP_COUNT}-step plan.

Read the full plan at ${WORK_DIR}/plan.md.

Execute ONLY step $i. Do not work on other steps.

After completing the step:
1. Run the verification described in the plan for this step.
2. Write a brief status report to ${WORK_DIR}/step_${i}_status.md containing:
   - What you did
   - What files you changed
   - Verification result (pass/fail)
   - Any issues encountered
3. If verification fails, attempt to fix the issue (up to 2 retries).
   If it still fails, write the failure details and stop.

Follow the core operating protocol from CLAUDE.md:
- Read before writing
- One change at a time
- Verify after every change
PROMPT
    )

    claude --print --max-turns 15 "$WORKER_PROMPT" 2>&1 | tee "${WORK_DIR}/step_${i}.log"

    if [ -f "${WORK_DIR}/step_${i}_status.md" ]; then
        echo ""
        echo "[Step $i/$STEP_COUNT] Status:"
        cat "${WORK_DIR}/step_${i}_status.md"
        echo ""

        # Check if the step reported failure
        if grep -qi "fail" "${WORK_DIR}/step_${i}_status.md"; then
            echo "WARNING: Step $i reported a failure. Continuing to review phase."
            echo "The reviewer will assess whether to proceed."
            break
        fi
    else
        echo "WARNING: Step $i did not produce a status file."
        echo "Check ${WORK_DIR}/step_${i}.log"
    fi
done

# ===========================================================================
# PHASE 3: REVIEWER
# ===========================================================================
# The reviewer does NOT modify code. It validates the combined result,
# checks for issues, and produces a final report.

echo ""
echo "============================================"
echo "[Phase 3/3] Review..."
echo "============================================"
echo ""

REVIEWER_PROMPT=$(cat <<PROMPT
You are the REVIEWER agent. You do NOT modify code. You validate and report.

A multi-step task was just executed. Here is the context:

- Original task: ${TASK}
- Plan: Read ${WORK_DIR}/plan.md
- Step statuses: Read ${WORK_DIR}/step_*_status.md files

Your job:
1. Run the full test suite. Record results.
2. Run the linter. Record results.
3. Review the git diff of all changes made during this session.
4. Check for:
   - Does the combined work actually accomplish the original task?
   - Are there any regressions (tests that passed before but fail now)?
   - Are there security concerns in the new code?
   - Is the code quality consistent with the rest of the codebase?
   - Were any anti-patterns introduced?

Write your review to ${WORK_DIR}/review.md in this format:

# Task Review

## Original Task
${TASK}

## Completion Status
[Complete / Partial / Failed]

## Test Results
[All passing / N failures — list them]

## Code Quality Assessment
[Good / Acceptable / Needs revision — specifics]

## Security Notes
[Any concerns, or "No issues found"]

## Regressions
[None found / List of regressions]

## Changes Summary
[Brief summary of all files changed and why]

## Recommendation
[APPROVE — ready to commit / REVISE — list what needs fixing / REJECT — explain why]

## Developer Action Items
[Numbered list of things the developer should review or decide on]

Also copy the final review to the project root as ORCHESTRATOR_REVIEW.md
so the developer sees it easily.

Be honest. If the work is incomplete or has issues, say so plainly.
PROMPT
)

claude --print --max-turns 10 "$REVIEWER_PROMPT" 2>&1 | tee "${WORK_DIR}/reviewer.log"

echo ""
echo "============================================"
echo "  Orchestration Complete"
echo "============================================"
echo ""
echo "Work directory: $WORK_DIR"
echo "  plan.md          — Execution plan"
echo "  step_*_status.md — Individual step reports"
echo "  review.md        — Final review"
echo ""

if [ -f "${WORK_DIR}/review.md" ]; then
    echo "Final review:"
    echo ""
    cat "${WORK_DIR}/review.md"
fi

echo ""
echo "All changes are uncommitted. Review the diff with:"
echo "  git diff"
echo ""
echo "If satisfied, commit with a descriptive message."
echo "If not, revert with: git checkout ."
