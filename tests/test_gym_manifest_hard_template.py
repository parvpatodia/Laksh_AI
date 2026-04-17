"""Hard-subset gym manifest template (P2) — same column contract as Phase A manifest."""

from pathlib import Path

import csv

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = PROJECT_ROOT / "evaluation" / "gym_manifest_hard.template.csv"
REQUIRED_FIELDS = (
    "clip_id",
    "path",
    "tags",
    "notes",
    "exercise_id",
    "expect_pose_usable",
    "expect_min_detection_rate",
)


@pytest.mark.skipif(not TEMPLATE.exists(), reason="gym_manifest_hard.template.csv missing")
def test_gym_manifest_hard_template_schema():
    with TEMPLATE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames
        for col in REQUIRED_FIELDS:
            assert col in reader.fieldnames, f"missing column {col}"
        rows = list(reader)

    assert len(rows) >= 1
    tags_flat = ",".join((r.get("tags") or "") for r in rows)
    assert "hard_subset" in tags_flat
    for r in rows:
        p = (r.get("path") or "").strip()
        assert p, "empty path"
        assert "evaluation/gym_clips/" in p or p.startswith("/")
