"""Tests for the v1 provenance builder's git-SHA sourcing.

Regression guard: the deploy bakes the commit SHA into the ``GIT_COMMIT_SHA``
env var (Dockerfile ``ENV GIT_COMMIT_SHA`` + ``make fly-deploy --build-arg
GIT_COMMIT_SHA=...``). The v1 provenance reader must honor that exact name so
every canonical report carries a resolvable commit SHA. A prior version read a
non-existent ``LAKSH_GIT_SHA``, fell through to ``git rev-parse`` (which fails
in the container), and shipped ``git_commit_sha: null``.
"""
import pytest

from app.api.v1 import provenance as prov

_HEX40 = 40


@pytest.fixture(autouse=True)
def _isolate_sha_cache(monkeypatch):
    """Each test starts and ends with a cleared @cache and no SHA env vars."""
    monkeypatch.delenv("GIT_COMMIT_SHA", raising=False)
    monkeypatch.delenv("LAKSH_GIT_SHA", raising=False)
    prov._git_commit_sha.cache_clear()
    yield
    prov._git_commit_sha.cache_clear()


def test_reads_git_commit_sha_env_var(monkeypatch):
    """The Dockerfile-baked GIT_COMMIT_SHA must be the source of truth."""
    monkeypatch.setenv("GIT_COMMIT_SHA", "abc123def456")
    prov._git_commit_sha.cache_clear()
    assert prov._git_commit_sha() == "abc123def456"


def test_truncates_to_40_chars(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT_SHA", "f" * 50)
    prov._git_commit_sha.cache_clear()
    assert prov._git_commit_sha() == "f" * _HEX40


def test_unknown_sentinel_is_not_reported_as_sha(monkeypatch):
    """The Dockerfile ARG defaults to 'unknown'; that must not leak as a SHA."""
    monkeypatch.setenv("GIT_COMMIT_SHA", "unknown")
    prov._git_commit_sha.cache_clear()
    assert prov._git_commit_sha() != "unknown"


def test_legacy_laksh_git_sha_still_accepted(monkeypatch):
    """Backward compatibility: LAKSH_GIT_SHA works when GIT_COMMIT_SHA is unset."""
    monkeypatch.setenv("LAKSH_GIT_SHA", "legacy0sha0value")
    prov._git_commit_sha.cache_clear()
    assert prov._git_commit_sha() == "legacy0sha0value"


def test_build_provenance_carries_sha(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT_SHA", "deadbeefcafe")
    prov._git_commit_sha.cache_clear()
    block = prov.build_provenance()
    assert block.git_commit_sha == "deadbeefcafe"
