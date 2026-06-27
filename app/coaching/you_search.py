"""Thin client for the You.com Search API.

REF: https://you.com/docs/api-reference/search/v1-search
    GET {base}/v1/search  with header X-API-Key
    params: query, count, freshness (day|week|month|year|YYYY-MM-DDtoYYYY-MM-DD),
            livecrawl
    response: {"results": {"web": [{title, url, description, snippets[]}], ...}}

The transport is injectable so the parse/citation path is testable without a
live key or network. The public API surface is small on purpose: ``enabled``
and ``search``.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

#: Documented base host. Overridable via env for staging or version pinning.
_DEFAULT_BASE = os.environ.get("YOU_API_BASE", "https://ydc-index.io")

#: (url, headers, params) -> parsed JSON dict.
Transport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]


@dataclass
class Source:
    """One retrieved web source."""

    title: str
    url: str
    snippet: str = ""


class YouComClient:
    """Minimal You.com Search client. Disabled (no-op) when no API key is set."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = _DEFAULT_BASE,
        livecrawl: Optional[str] = None,
        transport: Optional[Transport] = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("YOU_API_KEY")
        self._base = base_url.rstrip("/")
        self._livecrawl = livecrawl
        self._transport = transport or self._http_get

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def search(
        self, query: str, *, count: int = 5, freshness: Optional[str] = None
    ) -> list[Source]:
        """Return up to ``count`` web sources for ``query``. Empty if disabled."""
        if not self.enabled:
            return []
        params: dict[str, Any] = {"query": query, "count": count}
        if freshness:
            params["freshness"] = freshness
        if self._livecrawl:
            params["livecrawl"] = self._livecrawl
        headers = {"X-API-Key": str(self._api_key)}
        data = self._transport(f"{self._base}/v1/search", headers, params)
        return self._parse(data)

    @staticmethod
    def _parse(data: Any) -> list[Source]:
        """Defensively extract web hits across documented/observed response shapes."""
        if not isinstance(data, dict):
            return []
        results = data.get("results")
        hits: list[Any]
        if isinstance(results, dict):
            hits = results.get("web") or []
        elif isinstance(results, list):
            hits = results
        elif isinstance(data.get("hits"), list):
            hits = data["hits"]
        else:
            hits = []

        out: list[Source] = []
        for h in hits:
            if not isinstance(h, dict):
                continue
            url = h.get("url") or ""
            if not url:
                continue
            snips = h.get("snippets")
            snippet = (
                str(snips[0])
                if isinstance(snips, list) and snips
                else str(h.get("description") or "")
            )
            out.append(Source(title=str(h.get("title") or url), url=str(url), snippet=snippet))
        return out

    def _http_get(self, url: str, headers: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
        qs = urllib.parse.urlencode(params, doseq=True)
        req = urllib.request.Request(f"{url}?{qs}", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 -- fixed https host
            return json.loads(resp.read())
