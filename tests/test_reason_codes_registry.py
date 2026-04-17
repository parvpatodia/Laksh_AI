"""Registry completeness for pose baseline reason_codes (Phase A)."""

from app.pose.reason_codes import (
    DETECTION_QUALITY_CODES,
    FAILURE_CODES,
    REASON_CODE_DESCRIPTIONS,
)


def test_all_catalog_codes_have_descriptions():
    for c in FAILURE_CODES | DETECTION_QUALITY_CODES:
        assert c in REASON_CODE_DESCRIPTIONS, f"missing description for {c!r}"
