"""P2 person ROI helpers (no full pose inference)."""

import numpy as np
import pytest

from app.pose.person_isolation import (
    PERSON_ISOLATION_MODE_HAAR_MIL_V1,
    HaarMilPersonIsolation,
    clamp_roi_xyxy,
    create_person_isolation,
    expand_xyxy,
    normalize_person_isolation_mode,
    unmap_normalized_xy_from_crop,
)


def test_normalize_mode_none():
    assert normalize_person_isolation_mode(None) is None
    assert normalize_person_isolation_mode("") is None
    assert normalize_person_isolation_mode("  ") is None


def test_normalize_mode_aliases():
    assert normalize_person_isolation_mode("haar_mil_v1") == PERSON_ISOLATION_MODE_HAAR_MIL_V1
    assert normalize_person_isolation_mode("HAAR-MIL-V1") == PERSON_ISOLATION_MODE_HAAR_MIL_V1


def test_normalize_mode_unknown():
    with pytest.raises(ValueError, match="Unknown person_isolation"):
        normalize_person_isolation_mode("yolov8")


def test_create_factory():
    assert create_person_isolation(None) is None
    assert isinstance(create_person_isolation("haar_mil_v1"), HaarMilPersonIsolation)


def test_unmap_identity_full_crop():
    xf, yf = unmap_normalized_xy_from_crop(0.5, 0.25, 0, 0, 320, 240, 320, 240)
    assert xf == pytest.approx(0.5)
    assert yf == pytest.approx(0.25)


def test_unmap_offset_crop():
    # Center of 100×100 crop at (50,40) inside 200×200 frame → (0.5, 0.45) full-frame norm
    xf, yf = unmap_normalized_xy_from_crop(0.5, 0.5, 50, 40, 100, 100, 200, 200)
    assert xf == pytest.approx(0.5)
    assert yf == pytest.approx(0.45)


def test_unmap_degenerate_returns_input():
    xf, yf = unmap_normalized_xy_from_crop(0.3, 0.7, 0, 0, 0, 100, 200, 200)
    assert (xf, yf) == (0.3, 0.7)


def test_clamp_roi():
    x0, y0, x1, y1 = clamp_roi_xyxy(-5, -5, 400, 300, 320, 240, min_side=48)
    assert 0 <= x0 < x1 <= 320
    assert 0 <= y0 < y1 <= 240
    assert x1 - x0 >= 48
    assert y1 - y0 >= 48


def test_expand_xyxy():
    x0, y0, x1, y1 = expand_xyxy(50, 50, 150, 150, 300, 300, margin=0.1)
    # margin * max(100,100) = 10
    assert x0 == 40 and y0 == 40 and x1 == 160 and y1 == 160


def test_haar_mil_reset_stats():
    iso = HaarMilPersonIsolation(redetect_every_n_frames=5)
    iso.frames_full_frame_fallback = 3
    iso.start_clip()
    assert iso.frames_full_frame_fallback == 0
    assert iso._tracker is None
    assert iso._frame_index == 0


def test_haar_not_run_every_frame_when_never_detects(monkeypatch):
    """If Haar never finds a box, do not call it on every frame (regression guard)."""
    iso = HaarMilPersonIsolation(redetect_every_n_frames=5)
    iso.start_clip()
    monkeypatch.setattr(iso, "_detect_largest", lambda bgr: None)
    bgr = np.zeros((120, 160, 3), dtype=np.uint8)
    for _ in range(20):
        iso.step(bgr)
    assert iso.haar_detection_attempts == 4  # frame_index 0, 5, 10, 15
    assert iso.frames_full_frame_fallback == 20
    assert iso.stats_dict(mode="haar_mil_v1")["redetect_events"] == iso.haar_detection_attempts
