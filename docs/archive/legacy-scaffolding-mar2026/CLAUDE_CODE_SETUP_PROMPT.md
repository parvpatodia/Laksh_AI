Set up the full agent architecture for this project using the files I've placed in the project root. Do each step in order and verify before moving to the next.

## Step 1: Create directory structure

```
.cursor/rules/skills/
.claude/commands/
.claude/skills/
.agent-logs/
```

## Step 2: Move core agent rules

- Copy `agent-rules.md` to `.cursor/rules/agent-rules.md` (if not already there)
- Copy `agent-rules.md` to `CLAUDE.md` in the project root (this is what you, Claude Code, read)
- Keep the original `agent-rules.md` in the root as a reference copy

## Step 3: Move skill files to both Cursor and Claude Code locations

These 4 files should go to BOTH `.cursor/rules/skills/` AND `.claude/skills/`:
- `testing.md`
- `security.md`
- `performance.md`
- `refactoring.md`

## Step 4: Move slash command files

These 3 files go to `.claude/commands/`:
- `security-audit.md`
- `suggest-next.md`
- `improve-tests.md`

## Step 5: Make scripts executable

```bash
chmod +x background-runner.sh setup-schedule.sh orchestrator.sh
```

## Step 6: Move guide files to a docs folder

Create a `.agent-docs/` directory and move these reference files there so they don't clutter the root:
- `SETUP_GUIDE.md`
- `SKILLS_SETUP_GUIDE.md`
- `SKILLS_README.md` (if it exists)

## Step 7: Update .gitignore

Append these lines to `.gitignore` (create it if it doesn't exist), but only add lines that aren't already present:

```
# Agent output files
.agent-logs/
.agent-work/
.agent-docs/
HEALTH_LOG.md
REVIEW_LOG.md
RESEARCH_LOG.md
SUGGESTIONS.md
END_OF_DAY.md
URGENT.md
ORCHESTRATOR_REVIEW.md
SECURITY_AUDIT.md
```

## Step 8: Verify the final structure

Run `find . -name "*.md" -path "*cursor*" -o -name "*.md" -path "*claude*" | sort` and show me the result. Also show `ls -la *.sh` to confirm scripts are executable.

The correct final structure should look like:

```
.cursor/rules/agent-rules.md
.cursor/rules/skills/testing.md
.cursor/rules/skills/security.md
.cursor/rules/skills/performance.md
.cursor/rules/skills/refactoring.md
.claude/commands/security-audit.md
.claude/commands/suggest-next.md
.claude/commands/improve-tests.md
.claude/skills/testing.md
.claude/skills/security.md
.claude/skills/performance.md
.claude/skills/refactoring.md
CLAUDE.md
PROJECT_CONTEXT.md
GOALS.md
background-runner.sh (executable)
setup-schedule.sh (executable)
orchestrator.sh (executable)
```

## Step 9: Clean up

Remove the original copies from the project root that have been moved:
- Remove `testing.md` from root (now in skills dirs)
- Remove `security.md` from root (now in skills dirs)
- Remove `performance.md` from root (now in skills dirs)
- Remove `refactoring.md` from root (now in skills dirs)
- Remove `security-audit.md` from root (now in commands dir)
- Remove `suggest-next.md` from root (now in commands dir)
- Remove `improve-tests.md` from root (now in commands dir)
- Do NOT remove: `agent-rules.md` (reference copy), `PROJECT_CONTEXT.md`, `GOALS.md`, `CLAUDE.md`, or any `.sh` files

## Step 10: Report

After everything is done, show me:
1. The full directory tree of `.cursor/`, `.claude/`, and `.agent-docs/`
2. Confirmation that CLAUDE.md exists at root
3. Confirmation that all 3 shell scripts are executable
4. Any issues you encountered

Do NOT modify the contents of any file. Only move, copy, and organize.
