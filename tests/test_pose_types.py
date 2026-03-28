from app.pose.types import merge_reason_codes


def test_merge_reason_codes_ordering():
    c = merge_reason_codes(0.5, 100, 0.5)
    assert isinstance(c, list)
    assert "very_low_detection" not in c


def test_merge_reason_codes_low_detection():
    c = merge_reason_codes(0.02, 50, 0.5)
    assert "very_low_detection" in c


def test_merge_reason_codes_short_clip():
    c = merge_reason_codes(1.0, 2, 0.9)
    assert "short_clip" in c
