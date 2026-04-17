"""
Shell command lines for ``eval_scorecard_header.py`` (P1b / P2 archival discipline).

Orchestration scripts attach the result to JSON ``orchestration_report`` so scorecard
bundles stay copy-pasteable from a successful run.
"""
from __future__ import annotations

import shlex
from pathlib import Path


def suggest_eval_scorecard_header_command(
    *,
    repo_root: Path,
    manifest_path: Path,
    jsonl_paths: list[Path],
    python_exe: str = "python3",
) -> str:
    """
    Return a single line to run from **repo root** (paths under ``repo_root`` are
    emitted repo-relative for readability).
    """
    root = repo_root.resolve()

    def _rel(p: Path) -> str:
        pr = p.resolve()
        try:
            return str(pr.relative_to(root))
        except ValueError:
            return str(pr)

    tokens: list[str] = [
        python_exe,
        "scripts/eval_scorecard_header.py",
        "--manifest",
        _rel(manifest_path),
    ]
    for jp in jsonl_paths:
        tokens.extend(["--jsonl", _rel(jp)])

    return " ".join(shlex.quote(t) for t in tokens)
