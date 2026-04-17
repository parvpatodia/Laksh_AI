"""Gym manifest CSV loading (no OpenCV)."""
import csv
import logging
from pathlib import Path

import pytest

from app.pose.gym_manifest import (
    load_gym_manifest,
    parse_expect_min_detection_rate,
    summarize_manifest_path_status,
)


def test_load_and_validate_paths(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    clips = repo / "evaluation" / "clips"
    clips.mkdir(parents=True)
    vid = clips / "a.mp4"
    vid.write_bytes(b"fake")

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
                "path": "evaluation/clips/a.mp4",
                "tags": "t",
                "notes": "",
                "exercise_id": "",
                "expect_pose_usable": "",
                "expect_min_detection_rate": "",
            }
        )

    jobs = load_gym_manifest(manifest, repo)
    assert len(jobs) == 1
    assert jobs[0]["video_path"].is_file()

    stat = summarize_manifest_path_status(jobs)
    assert stat["files_present"] == 1
    assert stat["files_missing"] == 0


def test_duplicate_clip_id_raises(tmp_path: Path):
    repo = tmp_path / "repo"
    clips = repo / "evaluation" / "clips"
    clips.mkdir(parents=True)
    (clips / "a.mp4").write_bytes(b"x")
    (clips / "b.mp4").write_bytes(b"y")

    manifest = repo / "m.csv"
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
                "clip_id": "same",
                "path": "evaluation/clips/a.mp4",
                "tags": "",
                "notes": "",
                "exercise_id": "",
                "expect_pose_usable": "",
                "expect_min_detection_rate": "",
            }
        )
        w.writerow(
            {
                "clip_id": "same",
                "path": "evaluation/clips/b.mp4",
                "tags": "",
                "notes": "",
                "exercise_id": "",
                "expect_pose_usable": "",
                "expect_min_detection_rate": "",
            }
        )
    with pytest.raises(ValueError, match="duplicate clip_id"):
        load_gym_manifest(manifest, repo)


def test_empty_path_raises(tmp_path: Path):
    repo = tmp_path / "repo"
    manifest = repo / "m.csv"
    manifest.parent.mkdir(parents=True)
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
                "path": "",
                "tags": "",
                "notes": "",
                "exercise_id": "",
                "expect_pose_usable": "",
                "expect_min_detection_rate": "",
            }
        )
    with pytest.raises(ValueError, match="empty 'path'"):
        load_gym_manifest(manifest, repo)


def test_parse_expect_min_detection_rate_invalid_logs_and_returns_none(caplog):
    caplog.set_level(logging.WARNING)
    assert parse_expect_min_detection_rate("not-a-float") is None
    assert any("Invalid expect_min_detection_rate" in r.message for r in caplog.records)
    caplog.clear()
    assert parse_expect_min_detection_rate("1.5") is None
    assert any("outside [0,1]" in r.message for r in caplog.records)


def test_parse_expect_min_detection_rate_valid():
    assert parse_expect_min_detection_rate("0") == 0.0
    assert parse_expect_min_detection_rate("0.5") == 0.5
    assert parse_expect_min_detection_rate(" 1 ") == 1.0
