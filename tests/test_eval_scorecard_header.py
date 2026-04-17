"""Regression header script (stdlib + git optional)."""

import json
import subprocess
import sys
from pathlib import Path


def test_eval_scorecard_header_json(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    req = repo / "requirements.txt"
    req.write_text("pytest\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_scorecard_header.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--requirements", str(req)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["scorecard_schema_version"] == "1.1.0"
    assert data["requirements_txt_sha256"]
    assert len(data["requirements_txt_sha256"]) == 64
    assert data["interpreter"]


def test_eval_scorecard_header_jsonl_hashes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    req = repo / "requirements.txt"
    req.write_text("x\n", encoding="utf-8")
    jl = repo / "out.jsonl"
    jl.write_text('{"clip_id": "a", "ok": true}\n', encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_scorecard_header.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--requirements", str(req), "--jsonl", str(jl)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["scorecard_schema_version"] == "1.1.0"
    assert "pose_jsonl_artifacts" in data
    assert len(data["pose_jsonl_artifacts"]) == 1
    assert data["pose_jsonl_artifacts"][0]["sha256"]
    assert len(data["pose_jsonl_artifacts"][0]["sha256"]) == 64
