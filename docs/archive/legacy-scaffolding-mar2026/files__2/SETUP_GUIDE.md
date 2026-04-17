# Setup Guide — Agent Rules for Cursor & Claude Code

## How This Works (The Mental Model)

These files are not "prompts" in the casual sense. They are **persistent context** that
gets loaded into the AI model's working memory at the start of every interaction. Think of
them as the agent's "operating system" — they shape how it reasons, acts, and communicates
across your entire session.

The quality of these files directly determines agent behavior quality. A vague instruction
file produces vague work. A precise, structured instruction file produces methodical,
expert-level work.

---

## Option A: Cursor Rules (Recommended Starting Point)

### Step 1: Create the rules directory

In your project root:

```bash
mkdir -p .cursor/rules
```

### Step 2: Copy the agent rules

Copy `agent-rules.md` into your project:

```bash
cp agent-rules.md .cursor/rules/agent-rules.md
```

### Step 3: Add project context

Copy `PROJECT_CONTEXT.md` to your project root and fill it out:

```bash
cp PROJECT_CONTEXT.md ./PROJECT_CONTEXT.md
```

Edit it with your project's specific details — stack, conventions, current focus.

### Step 4: Use Cursor in Agent Mode

- Open Cursor
- Use **Cmd+I** (or Ctrl+I) to open the Composer
- Select **Agent** mode (not "Normal" or "Edit")
- The agent will now follow the rules defined in `.cursor/rules/`

### How Cursor Rules Load

- Files in `.cursor/rules/` are automatically loaded into context for every agent interaction.
- You can have multiple rule files (e.g., `agent-rules.md`, `code-style.md`, `testing-rules.md`).
- The `.cursorrules` file at the project root also works but is the older format.
  The `.cursor/rules/` directory is the current recommended approach.

---

## Option B: Claude Code (For Heavier Autonomous Tasks)

### Step 1: Install Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

Requires Node.js 18+.

### Step 2: Create CLAUDE.md

In your project root, create `CLAUDE.md` with the contents of `agent-rules.md`:

```bash
cp agent-rules.md ./CLAUDE.md
```

### Step 3: Add project context

Same as above — fill out `PROJECT_CONTEXT.md` in the project root.
Update the "Project Context Loading" section in `CLAUDE.md` to reference it.

### Step 4: Run Claude Code

```bash
cd your-project
claude
```

Claude Code reads `CLAUDE.md` automatically from the project root.

---

## Option C: Both Together

Use Cursor for interactive, in-IDE work (feature building, quick fixes, exploratory coding)
and Claude Code for larger autonomous tasks (refactoring a module, writing test suites,
migrating code).

Both read markdown instruction files. Keep them in sync, or maintain separate files
if you want different behavior for interactive vs autonomous work.

---

## Per-Project Customization

The agent adapts based on `PROJECT_CONTEXT.md`. Here are examples of how the persona
section changes agent behavior:

**For a React/Next.js web app:**
> Act as a senior frontend architect specializing in React performance, accessibility,
> and component design. Prioritize user experience, Core Web Vitals, and semantic HTML.
> When choosing between approaches, favor the one with better accessibility.

**For an ML pipeline:**
> Act as a senior ML engineer with deep expertise in PyTorch, experiment tracking,
> and production ML systems. Prioritize reproducibility, data pipeline correctness,
> and model evaluation rigor. Never skip validation steps.

**For a backend API:**
> Act as a senior backend engineer specializing in distributed systems, API design,
> and database performance. Think about failure modes, idempotency, and observability
> in every decision. Favor explicit error handling over implicit behavior.

**For a CLI tool:**
> Act as a senior systems programmer focused on developer experience, POSIX conventions,
> and composable tool design. Prioritize clear error messages, predictable behavior,
> and zero-surprise defaults.

---

## Maintenance

- Update `PROJECT_CONTEXT.md` as your project evolves (new stack decisions, shifting focus).
- Update `GOALS.md` as milestones are completed or priorities shift.
- The agent rules file rarely needs changes — it encodes process, not project specifics.
- If you find the agent consistently making a type of mistake, add it to the
  Anti-Patterns table in the rules file.

---

## Background Automation Setup (Claude Code Required)

This enables the agent to run on a schedule — health checks, code reviews, research,
and end-of-day push prompts — without you actively interacting with it.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Your Machine (macOS launchd / Linux cron)              │
│                                                         │
│  Schedule:                                              │
│    Every 3h  →  background-runner.sh health-check       │
│    7 AM      →  background-runner.sh review             │
│    2 AM      →  background-runner.sh research           │
│    Monday 8AM→  background-runner.sh suggest            │
│    5:30 PM   →  background-runner.sh end-of-day         │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  background-runner.sh                             │  │
│  │  - Selects task prompt based on argument          │  │
│  │  - Invokes Claude Code in non-interactive mode    │  │
│  │  - Claude Code reads CLAUDE.md + GOALS.md         │  │
│  │  - Writes findings to log files (never touches    │  │
│  │    application code)                              │  │
│  │  - Optionally sends desktop notification          │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Output files (checked by developer at session start):  │
│    HEALTH_LOG.md    — test/build/lint status             │
│    REVIEW_LOG.md    — code quality findings              │
│    RESEARCH_LOG.md  — improvement research               │
│    SUGGESTIONS.md   — prioritized next actions           │
│    END_OF_DAY.md    — push readiness + git commands      │
│    URGENT.md        — critical issues (if any)           │
└─────────────────────────────────────────────────────────┘
```

### Quick Start

```bash
# 1. Make sure Claude Code is installed and authenticated
npm install -g @anthropic-ai/claude-code
claude  # Run once to authenticate

# 2. Copy files to your project
cp agent-rules.md /path/to/your/project/CLAUDE.md
cp GOALS.md /path/to/your/project/GOALS.md
cp background-runner.sh /path/to/your/project/
cp setup-schedule.sh /path/to/your/project/

# 3. Fill out GOALS.md in your project

# 4. Test it manually first
cd /path/to/your/project
chmod +x background-runner.sh
./background-runner.sh . health-check

# 5. If the test works, set up the schedule
chmod +x setup-schedule.sh
./setup-schedule.sh /path/to/your/project
```

### What Happens at 5:30 PM

The end-of-day task generates `END_OF_DAY.md` which contains:

- Summary of all commits made today
- List of uncommitted changes
- Test and build status
- Exact git commands to run if everything is clean
- Suggested starting point for tomorrow

You open the file, review it, and either run the push commands or hold off.
The agent never pushes for you — it prepares everything and asks for your approval.

### Adding Desktop Notifications (macOS)

To get a native macOS notification when the end-of-day summary is ready,
add this to the end of the `end-of-day` case in `background-runner.sh`:

```bash
osascript -e 'display notification "End-of-day summary ready. Check END_OF_DAY.md" with title "Agent: Ready to Push?" sound name "Glass"'
```

### Limitations to Be Aware Of

- **Claude Code does not have persistent internet access by default.** The
  `research` task depends on Claude Code having web search capability. If your
  Claude Code setup does not include web search, the research task will be
  limited to analyzing your local codebase and documentation.
- **Background tasks do not modify your code.** This is by design. Automated
  code changes without human review are dangerous. The agent writes findings
  and suggestions; you decide what to act on.
- **API costs.** Each background invocation uses Claude API tokens. A health
  check is cheap (short context, minimal output). Research can be more expensive.
  Monitor your usage.
- **Rate limits.** If you run multiple projects, stagger the schedules to avoid
  hitting API rate limits simultaneously.

### Gitignore Additions

Add these to your `.gitignore` to keep agent output out of version control
(unless you want it tracked):

```
.agent-logs/
HEALTH_LOG.md
REVIEW_LOG.md
RESEARCH_LOG.md
SUGGESTIONS.md
END_OF_DAY.md
URGENT.md
```
