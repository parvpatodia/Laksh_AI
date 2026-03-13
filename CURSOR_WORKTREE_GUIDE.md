# Cursor Worktree Apply: Diagnosis & Permanent Fix

## What’s Happening

When you click **Apply** in Cursor:

1. Cursor reads the active worktree
2. Merges changes into your main repo
3. Writes files into your workspace

Your Apply hangs or fails because of:

| Issue | Why It Happens |
|-------|----------------|
| **Hanging for minutes** | You have 7 worktrees (~35MB each, pose models). Cursor tries to reconcile them and copy large files. |
| **EROFS (read-only)** | Cursor builds paths like `/Desktop/...` instead of `/Users/you/Desktop/...`. |
| **Missing .sh / config.yaml** | Apply expects these files; when they’re missing, it fails. |

## Stopping the Current Apply

1. Press **Escape** to dismiss the Apply / Review panel
2. Or **Cmd+Shift+P** → “Close Panel”
3. If it’s still frozen: **quit Cursor** (Cmd+Q), then reopen

## Permanent Fix (3 Parts)

### 1. Prune Old Worktrees

Cursor has created 7 worktrees; only `npa` is needed. Run:

```bash
cd /Users/parvpatodia/Desktop/Apex.ai

# Remove 6 unused worktrees
git worktree remove ~/.cursor/worktrees/Apex.ai/bxh --force 2>/dev/null || true
git worktree remove ~/.cursor/worktrees/Apex.ai/ivt --force 2>/dev/null || true
git worktree remove ~/.cursor/worktrees/Apex.ai/jek --force 2>/dev/null || true
git worktree remove ~/.cursor/worktrees/Apex.ai/kzv --force 2>/dev/null || true
git worktree remove ~/.cursor/worktrees/Apex.ai/mbo --force 2>/dev/null || true
git worktree remove ~/.cursor/worktrees/Apex.ai/zyr --force 2>/dev/null || true
```

Or use: `./scripts/cleanup-worktrees.sh`

### 2. Open via Projects Path

Open the project from `~/projects/Apex.ai` instead of `~/Desktop/Apex.ai` so the EROFS path bug is avoided.

- **File → Open Folder** → `/Users/parvpatodia/projects/Apex.ai`

### 3. Use Manual Apply Instead of Cursor’s Button

When Apply hangs or fails, use:

```bash
./scripts/apply-worktree.sh
```

Or **Cmd+Shift+P** → “Tasks: Run Task” → “Apply Worktree”

## Recommended Workflow

1. Use **Local** mode for agents when possible (no worktree, direct edits).
2. For “Run in worktree”: after the agent finishes, use `./scripts/apply-worktree.sh` instead of Apply.
3. Keep only 1–2 worktrees; run `./scripts/cleanup-worktrees.sh` after major work.
