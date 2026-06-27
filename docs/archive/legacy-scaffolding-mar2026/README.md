# Legacy scaffolding — archived 2026-04-17

These files are preserved historical artifacts from the March 2026 agent-workflow
setup attempt. They are **superseded** by:

- Global Claude Code system at `~/.claude/CLAUDE.md` + `~/.claude/commands/`
- Background agents at `~/claude-agents/` (project-scanner, slurm-monitor, repo-analyzer)
- Current root `CLAUDE.md` (18KB, dated Mar 25 2026) — the live project-specific rules

## Inventory

| File | Origin | Why archived |
|---|---|---|
| `CLAUDE.md.suggested` | Apr 17 2026 scaffold template | Stale pip template; current root `CLAUDE.md` is the source of truth |
| `CLAUDE_CODE_SETUP_PROMPT.md` | Mar 25 2026 | One-time setup instructions, already applied |
| `agent-rules.md` | Mar 25 2026 "Agent Operating System" | Duplicate content now lives in root `CLAUDE.md` |
| `files__1/orchestrator.sh` | Mar 26 2026 | Legacy shell orchestrator, replaced by LaunchAgent plists in `~/Library/LaunchAgents/com.parvpatodia.*` |
| `files__2/SETUP_GUIDE.md`, `background-runner.sh`, `setup-schedule.sh` | Mar 26 2026 | Pre-LaunchAgent scheduler. Superseded. |

Kept for historical reference. Do not re-introduce any of these into the live
agent pipeline without removing their successors first.
