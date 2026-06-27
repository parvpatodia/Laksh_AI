"""Tests for the dependency-inverted session store + record builder.

Covers the SQLite fallback (the always-available backend), the no-op Null
store, the backend factory's flag handling, and the envelope->record builder
that computes the leaderboard index and a measured-only fingerprint.
"""
import pytest

from app.persistence.models import build_session_record
from app.persistence.store import (
    NullSessionStore,
    SqliteSessionStore,
    get_session_store,
)


def _feat(value, status="valid"):
    return {"value": value, "unit": "x", "status": status, "reason_codes": []}


def _rep(i, status="valid", tempo=2.0, vis=0.95):
    return {
        "rep_index": i,
        "start_frame": i * 30,
        "end_frame": i * 30 + 29,
        "peak_frame": i * 30 + 15,
        "rep_status": status,
        "features": {
            "tempo_ratio_ecc_over_con": _feat(tempo),
            "primary_joints_min_visibility": _feat(vis),
            "rep_duration_s": _feat(1.5),
        },
    }


def _envelope(exercise_id="back_squat", reps=None, sha="abc123"):
    reps = reps if reps is not None else [_rep(0), _rep(1), _rep(2)]
    return {
        "sport_id": "gym",
        "exercise_id": exercise_id,
        "source": "frames_json",
        "fps": 30.0,
        "n_frames": 90,
        "provenance": {
            "git_commit_sha": sha,
            "pose_baseline_version": "1.2.0",
            "model": "none_frames_json",
        },
        "feature_vectors": reps,
    }


@pytest.fixture
def store(tmp_path):
    return SqliteSessionStore(db_path=tmp_path / "laksh.db")


# ---- record builder -------------------------------------------------------


def test_build_record_extracts_provenance_and_counts():
    rec = build_session_record(_envelope(), display_name="parv")
    assert rec.exercise_id == "back_squat"
    assert rec.sport_id == "gym"
    assert rec.git_commit_sha == "abc123"
    assert rec.pose_baseline_version == "1.2.0"
    assert rec.n_reps == 3
    assert rec.n_valid_reps == 3
    assert rec.display_name == "parv"
    assert rec.session_id  # auto-generated
    assert rec.form_index is not None and rec.form_index_status == "valid"


def test_fingerprint_contains_only_measured_values():
    reps = [_rep(0), _rep(1, status="unknown")]
    rec = build_session_record(_envelope(reps=reps))
    # 2 reps recorded in fingerprint, but counts reflect validity.
    assert len(rec.fingerprint) == 2
    assert rec.n_valid_reps == 1
    rep0 = next(r for r in rec.fingerprint if r.rep_index == 0)
    assert rep0.measured["tempo_ratio_ecc_over_con"] == 2.0


# ---- sqlite store ---------------------------------------------------------


def test_persist_then_leaderboard_roundtrip(store):
    rec = build_session_record(_envelope(), display_name="alice")
    sid = store.persist(rec)
    assert sid == rec.session_id
    board = store.leaderboard("back_squat")
    assert len(board) == 1
    assert board[0].display_name == "alice"
    assert board[0].rank == 1


def test_leaderboard_orders_by_form_index_desc(store):
    # High-consistency session vs noisy session -> higher index ranks first.
    high = build_session_record(_envelope(), display_name="high")
    noisy = build_session_record(
        _envelope(reps=[_rep(0, tempo=0.5), _rep(1, tempo=3.5), _rep(2, tempo=1.0)]),
        display_name="noisy",
    )
    store.persist(noisy)
    store.persist(high)
    board = store.leaderboard("back_squat")
    assert [e.display_name for e in board] == ["high", "noisy"]
    assert [e.rank for e in board] == [1, 2]


def test_leaderboard_filters_by_exercise(store):
    store.persist(build_session_record(_envelope(exercise_id="back_squat"), display_name="sq"))
    store.persist(build_session_record(_envelope(exercise_id="bench_press"), display_name="bp"))
    assert [e.display_name for e in store.leaderboard("back_squat")] == ["sq"]
    assert [e.display_name for e in store.leaderboard("bench_press")] == ["bp"]


def test_leaderboard_excludes_unscored_sessions(store):
    # A session with no valid reps has form_index=None and must not rank.
    store.persist(build_session_record(_envelope(reps=[_rep(0, status="unknown")]), display_name="ghost"))
    store.persist(build_session_record(_envelope(), display_name="real"))
    board = store.leaderboard("back_squat")
    assert [e.display_name for e in board] == ["real"]


def test_leaderboard_limit(store):
    for i in range(5):
        store.persist(build_session_record(_envelope(), display_name=f"u{i}"))
    assert len(store.leaderboard("back_squat", limit=3)) == 3


def test_sqlite_health_ok(store):
    assert store.health() is True


def test_persistence_survives_new_instance(tmp_path):
    db = tmp_path / "laksh.db"
    SqliteSessionStore(db_path=db).persist(build_session_record(_envelope(), display_name="persisted"))
    # A fresh instance on the same file sees the row.
    board = SqliteSessionStore(db_path=db).leaderboard("back_squat")
    assert board[0].display_name == "persisted"


# ---- null store + factory -------------------------------------------------


def test_null_store_is_noop():
    s = NullSessionStore()
    rec = build_session_record(_envelope())
    assert s.persist(rec) == rec.session_id
    assert s.leaderboard("back_squat") == []
    assert s.health() is True


def test_factory_defaults_to_sqlite(tmp_path):
    s = get_session_store(backend=None, db_path=tmp_path / "x.db")
    assert s.backend_name == "sqlite"


def test_factory_none_returns_null():
    assert get_session_store(backend="none").backend_name == "null"


def test_factory_insforge_without_adapter_falls_back_to_sqlite(tmp_path):
    # InsForge requested but adapter/key not wired -> safe SQLite fallback.
    s = get_session_store(backend="insforge", db_path=tmp_path / "x.db")
    assert s.backend_name == "sqlite"
