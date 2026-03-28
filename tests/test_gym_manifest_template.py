"""Gym Phase A manifest schema (pose baseline)."""
from pathlib import Path
import csv

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = PROJECT_ROOT / "evaluation" / "gym_manifest.template.csv"
REQUIRED_FIELDS = (
    "clip_id",
    "path",
    "tags",
    "notes",
    "exercise_id",
    "expect_pose_usable",
    "expect_min_detection_rate",
)


@pytest.mark.skipif(not TEMPLATE.exists(), reason="gym_manifest.template.csv missing")
def test_gym_manifest_template_schema():
    with TEMPLATE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames
        for col in REQUIRED_FIELDS:
            assert col in reader.fieldnames, f"missing column {col}"
        rows = list(reader)

    assert len(rows) >= 10, "template should list at least 10 gym scenario rows"

    tags_flat = ",".join((r.get("tags") or "") for r in rows)
    for t in (
        "squat",
        "deadlift",
        "phone_clean",
        "phone_low_light",
        "multi_person",
        "short_clip",
        "vfr_hevc",
    ):
        assert t in tags_flat, f"no row tagged with {t!r}"

    for r in rows:
        p = (r.get("path") or "").strip()
        assert p, "empty path"
        assert "evaluation/gym_clips/" in p or p.startswith("/"), "paths should be under evaluation/gym_clips/ or absolute"
