"""P3 parity probe: canonical joint path telemetry (ADR 0002)."""

import numpy as np
import pytest

from app.physics_engine import _canonical_joint_path_telemetry
from app.pose.canonical import CanonicalJointName, JointObservation
from app.pose.kinematic_flags import use_canonical_joint_trace


def test_kinematic_flag_default_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LAKSH_USE_CANONICAL_JOINTS", raising=False)
    assert use_canonical_joint_trace() is False


def test_kinematic_flag_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LAKSH_USE_CANONICAL_JOINTS", "1")
    assert use_canonical_joint_trace() is True


def _fake_frame_right_90deg_elbow() -> dict:
    """Right arm in an L-shape: shoulder (0.5,0.5), elbow (0.5,0.6), wrist (0.55,0.6)."""

    def obs(x: float, y: float) -> JointObservation:
        return JointObservation(x=x, y=y, z=0.0, visibility=0.95)
    return {
        CanonicalJointName.RIGHT_SHOULDER: obs(0.5, 0.5),
        CanonicalJointName.RIGHT_ELBOW: obs(0.5, 0.60),
        CanonicalJointName.RIGHT_WRIST: obs(0.55, 0.60),
        CanonicalJointName.RIGHT_HIP: obs(0.52, 0.72),
        CanonicalJointName.RIGHT_KNEE: obs(0.52, 0.84),
        CanonicalJointName.RIGHT_ANKLE: obs(0.52, 0.95),
    }


def test_canonical_probe_produces_deltas_when_legacy_provided():
    fr = _fake_frame_right_90deg_elbow()
    fd = fr
    cpf = [None, fd, fr]  # dip=1, release=2 for length check
    ar = 1.0
    # Same triplets as legacy path would use from raw indices → hand-crafted legacy angles
    h2d = np.array([fr[CanonicalJointName.RIGHT_HIP].x, fr[CanonicalJointName.RIGHT_HIP].y, 0.9])
    k2d = np.array([fr[CanonicalJointName.RIGHT_KNEE].x, fr[CanonicalJointName.RIGHT_KNEE].y, 0.9])
    a2d = np.array([fr[CanonicalJointName.RIGHT_ANKLE].x, fr[CanonicalJointName.RIGHT_ANKLE].y, 0.9])
    s2d = np.array([fr[CanonicalJointName.RIGHT_SHOULDER].x, fr[CanonicalJointName.RIGHT_SHOULDER].y, 0.9])
    e2d = np.array([fr[CanonicalJointName.RIGHT_ELBOW].x, fr[CanonicalJointName.RIGHT_ELBOW].y, 0.9])
    w2d = np.array([fr[CanonicalJointName.RIGHT_WRIST].x, fr[CanonicalJointName.RIGHT_WRIST].y, 0.9])
    from app.physics_engine import _angle_2d_image

    k_leg = _angle_2d_image(h2d, k2d, a2d, ar)
    e_leg = _angle_2d_image(s2d, e2d, w2d, ar)
    assert k_leg is not None and e_leg is not None

    out = _canonical_joint_path_telemetry(
        cpf,
        "right",
        1,
        2,
        ar,
        float(e_leg),
        float(k_leg),
    )
    assert out is not None
    assert out.get("delta_elbow_vs_legacy_2d_deg") == pytest.approx(0.0, abs=0.02)
    assert out.get("delta_knee_vs_legacy_2d_deg") == pytest.approx(0.0, abs=0.02)


def test_canonical_probe_missing_joint_returns_error():
    fr = _fake_frame_right_90deg_elbow()
    fd = {k: v for k, v in fr.items() if k != CanonicalJointName.RIGHT_KNEE}
    cpf = [None, fd, fr]
    ar = 1.0
    h2d = np.array([0.5, 0.72, 0.9])
    k2d = np.array([0.5, 0.84, 0.9])
    a2d = np.array([0.5, 0.95, 0.9])
    s2d = np.array([0.5, 0.5, 0.9])
    e2d = np.array([0.5, 0.6, 0.9])
    w2d = np.array([0.55, 0.6, 0.9])
    from app.physics_engine import _angle_2d_image

    k_leg = _angle_2d_image(h2d, k2d, a2d, ar)
    e_leg = _angle_2d_image(s2d, e2d, w2d, ar)
    assert k_leg is not None and e_leg is not None

    out = _canonical_joint_path_telemetry(
        cpf,
        "right",
        1,
        2,
        ar,
        float(e_leg),
        float(k_leg),
    )
    assert out is not None
    assert out.get("error") == "missing_joint_in_canonical_frame"
