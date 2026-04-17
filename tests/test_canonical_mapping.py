"""Canonical joint schema and MediaPipe 33 → COCO-17-name mapping (no MediaPipe import)."""
from types import SimpleNamespace

import pytest

from app.pose.canonical import (
    CANONICAL_JOINT_ORDER,
    CANONICAL_JOINT_SCHEMA_VERSION,
    CanonicalJointName,
    JointObservation,
    canonical_joint_dict_from_mapping,
)
from app.pose.mapping_mediapipe import (
    MEDIAPIPE_BLAZEPOSE33_TO_CANONICAL,
    map_mediapipe_blazepose33_to_canonical,
)


def _fake_landmarks33(**overrides: float) -> list[SimpleNamespace]:
    """33 landmarks; default x,y,z,visibility = 0; apply overrides by flat index key str."""
    out: list[SimpleNamespace] = []
    for _ in range(33):
        out.append(SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=1.0))
    for k, v in overrides.items():
        idx = int(k)
        out[idx] = SimpleNamespace(x=v, y=v + 0.01, z=0.02, visibility=0.99)
    return out


def test_schema_version_defined():
    assert CANONICAL_JOINT_SCHEMA_VERSION
    assert len(CANONICAL_JOINT_ORDER) == 17


def test_mapping_indices_cover_all_canonical_joints():
    assert set(MEDIAPIPE_BLAZEPOSE33_TO_CANONICAL.keys()) == set(CanonicalJointName)
    for idx in MEDIAPIPE_BLAZEPOSE33_TO_CANONICAL.values():
        assert 0 <= idx < 33


def test_map_none_and_wrong_length():
    assert map_mediapipe_blazepose33_to_canonical(None) is None
    assert map_mediapipe_blazepose33_to_canonical([]) is None
    assert map_mediapipe_blazepose33_to_canonical([SimpleNamespace(x=0, y=0, z=0, visibility=1)] * 32) is None


def test_map_roundtrip_known_indices():
    # Nose=0, left_hip=23, right_hip=24 with distinct values
    lms = _fake_landmarks33()
    lms[0] = SimpleNamespace(x=0.5, y=0.1, z=-0.01, visibility=0.9)
    lms[23] = SimpleNamespace(x=0.4, y=0.6, z=0.0, visibility=0.88)
    lms[24] = SimpleNamespace(x=0.6, y=0.61, z=0.0, visibility=0.87)
    m = map_mediapipe_blazepose33_to_canonical(lms)
    assert m is not None
    assert m[CanonicalJointName.NOSE] == JointObservation(0.5, 0.1, -0.01, 0.9)
    assert m[CanonicalJointName.LEFT_HIP].x == 0.4
    assert m[CanonicalJointName.RIGHT_HIP].x == 0.6


def test_frontal_invariant_left_hip_left_of_right_hip_in_image():
    """Synthetic frontal: subject's left hip appears on the right side of image → larger x."""
    lms = _fake_landmarks33()
    lms[23] = SimpleNamespace(x=0.55, y=0.5, z=0.0, visibility=1.0)
    lms[24] = SimpleNamespace(x=0.45, y=0.5, z=0.0, visibility=1.0)
    m = map_mediapipe_blazepose33_to_canonical(lms)
    assert m is not None
    assert m[CanonicalJointName.LEFT_HIP].x > m[CanonicalJointName.RIGHT_HIP].x


def test_canonical_joint_dict_json_shape():
    m = {
        CanonicalJointName.NOSE: JointObservation(0.1, 0.2, 0.0, 1.0),
    }
    d = canonical_joint_dict_from_mapping(m)
    assert d["nose"] == {"x": 0.1, "y": 0.2, "z": 0.0, "visibility": 1.0}


def test_visibility_fallback_presence_attr():
    lm = SimpleNamespace(x=0.0, y=0.0, z=0.0, presence=0.77)
    lms = [lm] + [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=1.0)] * 32
    m = map_mediapipe_blazepose33_to_canonical(lms)
    assert m is not None
    assert m[CanonicalJointName.NOSE].visibility == pytest.approx(0.77)
