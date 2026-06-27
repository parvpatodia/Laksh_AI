"""Smoke tests for scripts/subject_split_check.py (split leakage detector)."""
from __future__ import annotations

from scripts import subject_split_check as ssc


def _rows(pairs: list[tuple[str, str, str]]) -> list[dict]:
    """Helper: (clip_id, subject_id, split) -> row list."""
    return [
        {"clip_id": cid, "subject_id": sid, "split": split}
        for cid, sid, split in pairs
    ]


def test_clean_split() -> None:
    rows = _rows(
        [
            ("c1", "alice", "train"),
            ("c2", "alice", "train"),
            ("c3", "bob", "val"),
            ("c4", "carol", "test"),
        ]
    )
    report = ssc.check_splits(rows)
    assert report["leaked_subjects"] == {}
    assert report["bad_split_values"] == []
    assert report["split_coverage"] == {"train": 2, "val": 1, "test": 1}
    assert report["n_subjects"] == 3


def test_leak_detected_when_subject_crosses_splits() -> None:
    rows = _rows(
        [
            ("c1", "alice", "train"),
            ("c2", "alice", "test"),  # leak
            ("c3", "bob", "val"),
        ]
    )
    report = ssc.check_splits(rows)
    assert "alice" in report["leaked_subjects"]
    assert sorted(report["leaked_subjects"]["alice"]) == ["test", "train"]


def test_bad_split_value_collected() -> None:
    rows = _rows(
        [
            ("c1", "alice", "trian"),  # typo
            ("c2", "bob", "val"),
        ]
    )
    report = ssc.check_splits(rows)
    assert report["bad_split_values"] == [{"clip_id": "c1", "split": "trian"}]


def test_session_leak_with_flag() -> None:
    rows = [
        {"clip_id": "c1", "subject_id": "alice", "session_id": "s1", "split": "train"},
        {"clip_id": "c2", "subject_id": "alice", "session_id": "s1", "split": "val"},
    ]
    # subject "alice" also leaks — checked in same report.
    report = ssc.check_splits(rows, check_sessions=True)
    assert "s1" in report["leaked_sessions"]
    assert sorted(report["leaked_sessions"]["s1"]) == ["train", "val"]


def test_sessions_not_checked_without_flag() -> None:
    rows = [
        {"clip_id": "c1", "subject_id": "alice", "session_id": "s1", "split": "train"},
        {"clip_id": "c2", "subject_id": "bob", "session_id": "s1", "split": "val"},
    ]
    report = ssc.check_splits(rows, check_sessions=False)
    assert report["leaked_sessions"] == {}
    # subjects are fine (different subjects per split)
    assert report["leaked_subjects"] == {}
