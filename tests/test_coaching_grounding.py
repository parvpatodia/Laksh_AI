"""Tests for You.com-grounded coaching.

The grounding logic is verified here against a MOCKED You.com transport, so the
request/parse/citation-shaping path is fully exercised without a live key. The
honesty contract is pinned: the deterministic cue text is preserved and only
real, resolvable citations are attached -- detection is never LLM-invented.
"""
from app.coaching.grounding import GroundedCue, build_query, ground_faults
from app.coaching.you_search import Source, YouComClient

# A You.com /v1/search response in the documented shape.
_YOU_RESPONSE = {
    "results": {
        "web": [
            {
                "title": "NSCA eccentric tempo guidelines",
                "url": "https://example.org/nsca-tempo",
                "description": "Controlled eccentric loading recommendations.",
                "snippets": ["Aim for a 2-4s eccentric under load."],
            },
            {
                "title": "Squat tempo and hypertrophy",
                "url": "https://example.org/squat-tempo",
                "description": "Review of tempo prescriptions.",
                "snippets": ["Slower descents increase time under tension."],
            },
        ]
    },
    "metadata": {"search_uuid": "abc", "query": "x", "latency": 0.1},
}


def _transport(payload):
    def _t(url, headers, params):
        assert "X-API-Key" in headers
        assert "query" in params
        return payload

    return _t


def test_client_disabled_without_key():
    assert YouComClient(api_key=None).enabled is False
    assert YouComClient(api_key="k").enabled is True


def test_client_parses_web_results():
    c = YouComClient(api_key="k", transport=_transport(_YOU_RESPONSE))
    sources = c.search("back squat eccentric tempo", count=2)
    assert len(sources) == 2
    assert isinstance(sources[0], Source)
    assert sources[0].title == "NSCA eccentric tempo guidelines"
    assert sources[0].url.startswith("https://")
    assert sources[0].snippet


def test_client_returns_empty_when_disabled():
    # No transport call should happen without a key.
    c = YouComClient(api_key=None, transport=_transport(_YOU_RESPONSE))
    assert c.search("anything") == []


def test_build_query_uses_exercise_and_known_fault():
    q = build_query("back_squat", {"id": "tempo-fast-eccentric", "title": "Control the lowering phase"})
    assert "squat" in q.lower()
    assert "eccentric" in q.lower() or "tempo" in q.lower()


def test_ground_attaches_citations_and_preserves_cue():
    c = YouComClient(api_key="k", transport=_transport(_YOU_RESPONSE))
    faults = [{"id": "tempo-fast-eccentric", "title": "Control the lowering phase", "cue": "Slow your descent."}]
    cues = ground_faults("back_squat", faults, client=c, max_citations=2)
    assert len(cues) == 1
    cue = cues[0]
    assert isinstance(cue, GroundedCue)
    assert cue.grounded is True
    assert cue.cue == "Slow your descent."  # deterministic text preserved verbatim
    assert len(cue.citations) == 2
    assert all(cit.url.startswith("https://") for cit in cue.citations)
    assert "you_com_grounded" in cue.reason_codes


def test_ground_falls_back_without_key():
    c = YouComClient(api_key=None)
    faults = [{"id": "tempo-fast-eccentric", "title": "t", "cue": "Slow descent."}]
    cues = ground_faults("back_squat", faults, client=c)
    assert cues[0].grounded is False
    assert cues[0].citations == []
    assert cues[0].cue == "Slow descent."  # cue still shown (honest fallback)
    assert "grounding_unavailable_no_key" in cues[0].reason_codes


def test_ground_handles_no_sources_found():
    c = YouComClient(api_key="k", transport=_transport({"results": {"web": []}}))
    cues = ground_faults("back_squat", [{"id": "x", "title": "t", "cue": "c"}], client=c)
    assert cues[0].grounded is False
    assert "no_sources_found" in cues[0].reason_codes


def test_ground_handles_search_exception():
    def _boom(url, headers, params):
        raise RuntimeError("network down")

    c = YouComClient(api_key="k", transport=_boom)
    cues = ground_faults("back_squat", [{"id": "x", "title": "t", "cue": "c"}], client=c)
    assert cues[0].grounded is False
    assert "you_com_search_failed" in cues[0].reason_codes
