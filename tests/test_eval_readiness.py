"""eval_readiness static report (no inference)."""

import csv
from pathlib import Path

import app.pose.eval_readiness as eval_readiness_mod
from app.pose.eval_readiness import collect_eval_readiness


def test_collect_minimal_keys(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    def fake_default():
        return repo / "pose_landmarker_heavy.task"

    import app.pose.mediapipe_common as mc

    monkeypatch.setattr(mc, "default_model_path", fake_default)
    r = collect_eval_readiness(repo_root=repo)
    assert r["report_purpose"] == "static_readiness_no_inference"
    assert r.get("report_schema_version") == "1.2.0"
    assert r.get("interpreter")
    assert "mediapipe_gym_eval_minimal" in r
    assert "rtmpose_stack_imports_ok" in r
    assert isinstance(r["notes"], list)
    assert "probe_error" in r["opencv"]
    if r["opencv"]["import_ok"]:
        assert r["opencv"]["version"] is not None
        assert r["opencv"]["probe_error"] is None


def test_manifest_validation_ok(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    clips = repo / "evaluation" / "clips"
    clips.mkdir(parents=True)
    (clips / "x.mp4").write_bytes(b"x")
    manifest = repo / "evaluation" / "m.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id",
                "path",
                "tags",
                "notes",
                "exercise_id",
                "expect_pose_usable",
                "expect_min_detection_rate",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "clip_id": "c1",
                "path": "evaluation/clips/x.mp4",
                "tags": "",
                "notes": "",
                "exercise_id": "",
                "expect_pose_usable": "",
                "expect_min_detection_rate": "",
            }
        )
    r = collect_eval_readiness(manifest_path=manifest, repo_root=repo)
    assert r["gym_manifest"]["load_ok"] is True
    assert r["gym_manifest"]["files_missing"] == 0


def test_probe_error_on_subprocess_style_failure(tmp_path: Path, monkeypatch):
    """Failure detail must not be stuffed into ``version`` (schema 1.2.0)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    import app.pose.mediapipe_common as mc

    monkeypatch.setattr(mc, "default_model_path", lambda: repo / "pose_landmarker_heavy.task")

    real_import_ok = eval_readiness_mod._import_ok

    def fake_import_ok(name: str):
        if name == "onnxruntime":
            return False, "child killed by signal 11"
        return real_import_ok(name)

    monkeypatch.setattr(eval_readiness_mod, "_import_ok", fake_import_ok)
    r = collect_eval_readiness(repo_root=repo)
    assert r["onnxruntime"]["import_ok"] is False
    assert r["onnxruntime"]["version"] is None
    assert r["onnxruntime"]["probe_error"] == "child killed by signal 11"


def test_not_installed_probe_message(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    import app.pose.mediapipe_common as mc

    monkeypatch.setattr(mc, "default_model_path", lambda: repo / "pose_landmarker_heavy.task")

    real_import_ok = eval_readiness_mod._import_ok

    def fake_import_ok(name: str):
        if name == "rtmlib":
            return False, None
        return real_import_ok(name)

    monkeypatch.setattr(eval_readiness_mod, "_import_ok", fake_import_ok)
    r = collect_eval_readiness(repo_root=repo)
    assert r["rtmlib"]["import_ok"] is False
    assert r["rtmlib"]["version"] is None
    assert r["rtmlib"]["probe_error"] is not None
    assert "interpreter" in r["rtmlib"]["probe_error"].lower() or "module not found" in r[
        "rtmlib"
    ]["probe_error"].lower()
