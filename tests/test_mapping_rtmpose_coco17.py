"""COCO-17 pixel → canonical mapping (no rtmlib)."""
import numpy as np
import pytest

from app.pose.canonical import CanonicalJointName
from app.pose.mapping_rtmpose_coco17 import (
    canonical_to_gym_raw_row,
    coco17_pixels_to_canonical,
)


def test_coco17_pixels_to_canonical_normalizes():
    kp = np.zeros((17, 2), dtype=np.float64)
    kp[:, 0] = np.linspace(0, 100, 17)
    kp[:, 1] = 50.0
    sc = np.ones(17, dtype=np.float64) * 0.8
    c = coco17_pixels_to_canonical(kp, sc, image_width=100, image_height=100)
    assert c[CanonicalJointName.NOSE].x == pytest.approx(0.0)
    assert c[CanonicalJointName.NOSE].y == pytest.approx(0.5)
    assert c[CanonicalJointName.NOSE].visibility == pytest.approx(0.8)


def test_coco17_pixels_clips_to_unit_square():
    kp = np.array([[150.0, 75.0]] + [[0, 0]] * 16, dtype=np.float64)
    sc = np.ones(17, dtype=np.float64)
    c = coco17_pixels_to_canonical(kp, sc, image_width=100, image_height=100)
    assert c[CanonicalJointName.NOSE].x == pytest.approx(1.0)
    assert c[CanonicalJointName.NOSE].y == pytest.approx(0.75)


def test_coco17_wrong_shape_raises():
    with pytest.raises(ValueError):
        coco17_pixels_to_canonical(np.zeros((16, 2)), np.ones(17), image_width=10, image_height=10)


def test_canonical_to_gym_raw_row():
    c = coco17_pixels_to_canonical(
        np.ones((17, 2)) * 5, np.ones(17) * 0.5, image_width=10, image_height=10
    )
    row = canonical_to_gym_raw_row(c)
    assert row["left_wrist"].shape == (3,)
    assert np.isfinite(row["left_wrist"][0])

    row2 = canonical_to_gym_raw_row(None)
    assert np.all(np.isnan(row2["left_hip"]))
