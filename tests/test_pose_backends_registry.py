"""Backend registry (no heavy inference)."""

from app.pose.backends import get_pose_backend


def test_get_mediapipe():
    b = get_pose_backend("mediapipe")
    assert b.name == "mediapipe"


def test_get_rtmpose():
    b = get_pose_backend("rtmpose")
    assert b.name == "rtmpose"


def test_unknown_backend():
    try:
        get_pose_backend("vitpose")
    except NotImplementedError as e:
        msg = str(e).lower()
        assert "vitpose" in msg or "supported" in msg or "unknown" in msg
    else:
        raise AssertionError("expected NotImplementedError")
