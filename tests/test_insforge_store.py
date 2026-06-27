"""Tests for the InsForge-backed SessionStore adapter.

Uses an injected fake transport so the request shaping (insert body, leaderboard
query params, ranking) is verified without a live key or network. The real REST
contract (POST/GET /api/database/records/{table}, Bearer auth, order/limit
filters) was confirmed live before this was written.
"""
from app.persistence.insforge_store import InsForgeSessionStore
from app.persistence.models import build_session_record


def _feat(v, s="valid"):
    return {"value": v, "unit": "x", "status": s, "reason_codes": []}


def _rep(i, status="valid", tempo=2.0, vis=0.95):
    return {
        "rep_index": i,
        "rep_status": status,
        "features": {
            "tempo_ratio_ecc_over_con": _feat(tempo),
            "primary_joints_min_visibility": _feat(vis),
            "rep_duration_s": _feat(1.5),
        },
    }


def _env(exercise_id="back_squat", reps=None, sha="abc"):
    reps = reps if reps is not None else [_rep(0), _rep(1), _rep(2)]
    return {
        "sport_id": "gym", "exercise_id": exercise_id, "source": "frames_json",
        "fps": 30.0, "n_frames": 90,
        "provenance": {"git_commit_sha": sha, "pose_baseline_version": "1.2.0", "model": "none_frames_json"},
        "feature_vectors": reps,
    }


class FakeTransport:
    def __init__(self, get_rows=None, get_status=200, post_status=201):
        self.calls = []
        self.get_rows = get_rows if get_rows is not None else []
        self.get_status = get_status
        self.post_status = post_status

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if method == "GET":
            return self.get_status, self.get_rows
        if method == "POST":
            return self.post_status, []
        return 204, None


def _store(transport):
    return InsForgeSessionStore("https://q9p6qk2x.us-west.insforge.app", "ik_test", transport=transport)


def test_persist_posts_session_then_reps():
    fake = FakeTransport()
    rec = build_session_record(_env(), display_name="alice")
    assert _store(fake).persist(rec) == rec.session_id

    posts = [c for c in fake.calls if c[0] == "POST"]
    sess = next(c for c in posts if "/api/database/records/sessions" in c[1])
    reps = next(c for c in posts if "/api/database/records/rep_results" in c[1])
    # bodies are JSON arrays (InsForge requires array even for one row)
    assert isinstance(sess[2], list) and sess[2][0]["session_id"] == rec.session_id
    assert sess[2][0]["form_index"] is not None
    assert isinstance(reps[2], list) and len(reps[2]) == 3
    assert reps[2][0]["session_id"] == rec.session_id


def test_persist_raises_on_session_insert_failure():
    fake = FakeTransport(post_status=500)
    rec = build_session_record(_env())
    try:
        _store(fake).persist(rec)
        raised = False
    except RuntimeError:
        raised = True
    assert raised  # endpoint wraps this best-effort; the store surfaces the failure


def test_leaderboard_ranks_and_builds_query():
    rows = [
        {"session_id": "a", "display_name": "high", "exercise_id": "back_squat", "form_index": 95.0,
         "form_index_status": "valid", "n_valid_reps": 3, "n_reps": 3, "created_at": "2026-06-27T00:00:00Z", "git_commit_sha": "x"},
        {"session_id": "b", "display_name": "low", "exercise_id": "back_squat", "form_index": 80.0,
         "form_index_status": "valid", "n_valid_reps": 3, "n_reps": 3, "created_at": "2026-06-27T00:01:00Z", "git_commit_sha": "x"},
    ]
    fake = FakeTransport(get_rows=rows)
    board = _store(fake).leaderboard("back_squat", limit=10)
    assert [e.display_name for e in board] == ["high", "low"]
    assert [e.rank for e in board] == [1, 2]
    get_path = next(c[1] for c in fake.calls if c[0] == "GET")
    assert "order=form_index.desc" in get_path
    assert "form_index=not.is.null" in get_path
    assert "exercise_id=eq.back_squat" in get_path
    assert "limit=10" in get_path


def test_leaderboard_returns_empty_on_error_not_500():
    # A read failure must degrade to [] (never crash the endpoint).
    assert _store(FakeTransport(get_status=503)).leaderboard("back_squat") == []


def test_health_reflects_status():
    assert _store(FakeTransport(get_status=200)).health() is True
    assert _store(FakeTransport(get_status=500)).health() is False
