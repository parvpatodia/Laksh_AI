"""Suggested eval_scorecard_header command (P1b / P2 archival)."""

from pathlib import Path

from app.pose.scorecard_command import suggest_eval_scorecard_header_command


def test_suggest_command_repo_relative_paths(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "scripts").mkdir()
    (root / "evaluation").mkdir()
    manifest = root / "evaluation" / "gym_manifest.csv"
    manifest.write_text("x\n", encoding="utf-8")
    j1 = root / "evaluation" / "a.jsonl"
    j2 = root / "evaluation" / "b.jsonl"
    j1.write_text("{}\n", encoding="utf-8")
    j2.write_text("{}\n", encoding="utf-8")

    cmd = suggest_eval_scorecard_header_command(
        repo_root=root,
        manifest_path=manifest,
        jsonl_paths=[j1, j2],
        python_exe="/usr/bin/python3",
    )
    assert "evaluation/gym_manifest.csv" in cmd
    assert "evaluation/a.jsonl" in cmd
    assert "evaluation/b.jsonl" in cmd
    assert "--jsonl" in cmd
    assert cmd.startswith("/usr/bin/python3")


def test_suggest_command_quotes_spaces(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    sub = root / "eval dir"
    sub.mkdir()
    manifest = sub / "manifest.csv"
    manifest.write_text("x", encoding="utf-8")
    jl = sub / "out.jsonl"
    jl.write_text("{}\n", encoding="utf-8")
    cmd = suggest_eval_scorecard_header_command(
        repo_root=root,
        manifest_path=manifest,
        jsonl_paths=[jl],
        python_exe="python3",
    )
    assert "eval dir" in cmd
    assert "'" in cmd or '"' in cmd
