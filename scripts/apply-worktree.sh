#!/usr/bin/env bash
# Apply worktree changes when Cursor's Apply button fails (EROFS/Desktop path bug).
# Run: ./scripts/apply-worktree.sh   or from Cursor: Cmd+Shift+P -> "Tasks: Run Task" -> "Apply Worktree"
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "Merging worktree branch 'npa' into $(git branch --show-current)..."
if git merge npa -m "Merge worktree npa"; then
  echo "Apply complete."
else
  echo "Merge had conflicts. Resolve them, then: git add -A && git commit"
  exit 1
fi
