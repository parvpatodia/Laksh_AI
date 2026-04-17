"""Shared gym baseline metric helpers."""
import numpy as np

from app.pose.gym_baseline_metrics import aggregate_metrics, frame_has_pose, utility_score


def test_aggregate_metrics_empty():
    raw = {k: np.zeros((0, 3)) for k in ["left_hip", "right_hip", "left_shoulder", "right_shoulder", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_wrist", "right_wrist", "left_elbow", "right_elbow"]}
    n, dr, vd, va, disp = aggregate_metrics(raw, 0)
    assert n == 0 and dr == 0.0 and disp is None


def test_frame_has_pose_requires_both_hips():
    raw = {
        "left_hip": np.array([[0.4, 0.5, 1.0]]),
        "right_hip": np.array([[np.nan, np.nan, np.nan]]),
        "left_shoulder": np.array([[0.4, 0.3, 1.0]]),
        "right_shoulder": np.array([[0.6, 0.3, 1.0]]),
        "left_knee": np.array([[0.4, 0.7, 1.0]]),
        "right_knee": np.array([[0.6, 0.7, 1.0]]),
        "left_ankle": np.array([[0.4, 0.9, 1.0]]),
        "right_ankle": np.array([[0.6, 0.9, 1.0]]),
        "left_wrist": np.array([[0.35, 0.2, 1.0]]),
        "right_wrist": np.array([[0.65, 0.2, 1.0]]),
        "left_elbow": np.array([[0.37, 0.25, 1.0]]),
        "right_elbow": np.array([[0.63, 0.25, 1.0]]),
    }
    assert not frame_has_pose(raw, 0)


def test_utility_score_positive_with_pose():
    raw = {
        "left_hip": np.array([[0.4, 0.5, 1.0]]),
        "right_hip": np.array([[0.6, 0.5, 1.0]]),
        "left_shoulder": np.array([[0.4, 0.3, 1.0]]),
        "right_shoulder": np.array([[0.6, 0.3, 1.0]]),
        "left_knee": np.array([[0.4, 0.7, 1.0]]),
        "right_knee": np.array([[0.6, 0.7, 1.0]]),
        "left_ankle": np.array([[0.4, 0.9, 1.0]]),
        "right_ankle": np.array([[0.6, 0.9, 1.0]]),
        "left_wrist": np.array([[0.35, 0.2, 1.0]]),
        "right_wrist": np.array([[0.65, 0.2, 1.0]]),
        "left_elbow": np.array([[0.37, 0.25, 1.0]]),
        "right_elbow": np.array([[0.63, 0.25, 1.0]]),
    }
    assert utility_score(raw, 1) > 0
