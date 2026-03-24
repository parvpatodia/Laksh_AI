"""Manifest template schema for evaluation/benchmark workflow."""
from pathlib import Path
import csv

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = PROJECT_ROOT / "evaluation" / "manifest.template.csv"
REQUIRED_FIELDS = (
    "clip_id",
    "path",
    "tags",
    "notes",
    "expect_analysis_mode",
    "expect_min_measured",
)


@pytest.mark.skipif(not TEMPLATE.exists(), reason="manifest.template.csv missing")
def test_manifest_template_exists_and_covers_categories():
    with TEMPLATE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames
        for col in REQUIRED_FIELDS:
            assert col in reader.fieldnames, f"missing column {col}"
        rows = list(reader)

    assert len(rows) >= 20, "template should list at least 20 clips per evaluation spec"

    tags_flat = ",".join((r.get("tags") or "") for r in rows)
    required_tags = (
        "phone_clean",
        "phone_low_light",
        "phone_far_subject",
        "yt_short_reencode",
        "broadcast_crop",
        "side_view_ft",
        "side_view_jumper",
        "multi_person",
        "occlusion",
        "vfr_hevc",
        "short_clip",
    )
    for t in required_tags:
        assert t in tags_flat, f"no row tagged with {t!r}"

    for r in rows:
        p = (r.get("path") or "").strip()
        assert p, "empty path"
        assert "evaluation/clips/" in p or p.startswith("/"), "paths should be under evaluation/clips/ or absolute"
