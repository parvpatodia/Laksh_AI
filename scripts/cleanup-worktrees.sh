#!/usr/bin/env bash
# Remove old Cursor worktrees to stop Apply hanging and free ~250MB disk.
# Run after closing Cursor (or when no agent is running). Safe to run anytime.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "Current worktrees:"
git worktree list
echo ""
echo "Removing worktrees (keeps main repo and npa if it has unmerged changes)..."
# Remove worktrees that are detached; keep npa if it has a branch we care about
for wt in ~/.cursor/worktrees/Apex.ai/*/; do
  [ -d "$wt" ] || continue
  name=$(basename "$wt")
  [ "$name" = "npa" ] && echo "  Keeping npa" && continue
  echo "  Removing $name..."
  git worktree remove "$wt" --force 2>/dev/null || true
done
echo "Done. Run 'git worktree list' to verify."
