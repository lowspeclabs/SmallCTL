from __future__ import annotations

import asyncio

import pytest

from smallctl.search_server.app import SearchServerError, SearchServerService
from smallctl.search_server.citations import now_iso
from smallctl.search_server.config import SearchServerConfig
from smallctl.search_server.fetch import FetchedDocument
from smallctl.search_server.models import CitationSource, WebFetchRequest, WebFetchResponse, WebSearchRequest, WebSearchResult
from smallctl.search_server.providers import SearxNGProvider, _parse_duckduckgo_html, normalize_search_results


def _make_service(tmp_path, **config_overrides) -> SearchServerService:
    config = SearchServerConfig(cache_path=str(tmp_path / "search-cache.sqlite3"), **config_overrides)
    return SearchServerService.create(config)


def test_search_service_cache_hit_and_expiry(monkeypatch, tmp_path) -> None:
    calls: list[str] = []
    now = [1000.0]

    async def fake_search_with_providers(request, *, config):
        del config
        calls.append(request.query)
        return (
            [
                WebSearchResult(
                    result_id="webres-cache-1",
                    title="Cache Result",
                    url="https://example.com/story",
                    canonical_url="https://example.com/story",
                    domain="example.com",
                    snippet="cached snippet",
                    provider="duckduckgo",
                    rank=1,
                )
            ],
            [],
            "duckduckgo",
            "none",
        )

    monkeypatch.setattr("smallctl.search_server.app.search_with_providers", fake_search_with_providers)
    monkeypatch.setattr("smallctl.search_server.cache.time.time", lambda: now[0])

    service = _make_service(tmp_path, search_ttl_seconds=5, negative_ttl_seconds=2)
    request = WebSearchRequest(query="latest example", recency_days=7)

    first = asyncio.run(service.search(request))
    second = asyncio.run(service.search(request))
    now[0] += 6
    third = asyncio.run(service.search(request))

    assert first.recency_support == "none"
    assert first.recency_enforced is False
    assert any("could not enforce recency_days=7" in item for item in first.warnings)
    assert second.results[0].result_id == "webres-cache-1"
    assert len(calls) == 2
    assert third.results[0].title == "Cache Result"


def test_search_service_marks_strict_recency_without_warning(monkeypatch, tmp_path) -> None:
    async def fake_search_with_providers(request, *, config):
        del config
        return (
            [
                WebSearchResult(
                    result_id="webres-recency-1",
                    title="Fresh Result",
                    url="https://example.com/fresh",
                    canonical_url="https://example.com/fresh",
                    domain="example.com",
                    snippet="fresh snippet",
                    provider="brave",
                    rank=1,
                )
            ],
            [],
            "brave",
            "strict",
        )

    monkeypatch.setattr("smallctl.search_server.app.search_with_providers", fake_search_with_providers)
    service = _make_service(tmp_path)

    response = asyncio.run(service.search(WebSearchRequest(query="fresh news", recency_days=3)))

    assert response.recency_support == "strict"
    assert response.recency_enforced is True
    assert response.warnings == []


def test_fetch_service_negative_cache_expires(monkeypatch, tmp_path) -> None:
    calls = {"count": 0}
    now = [2000.0]

    async def fake_fetch_document(request, *, config, url):
        del config, request, url
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary upstream failure")
        response = WebFetchResponse(
            source_id="websrc-cache-1",
            url="https://example.com/story",
            canonical_url="https://example.com/story",
            domain="example.com",
            title="Recovered Story",
            byline=None,
            published_at=None,
            fetched_at=now_iso(),
            content_type="text/html",
            content_sha256="hash",
            text_excerpt="excerpt",
            char_start=1,
            char_end=7,
            warnings=[],
            artifact_id=None,
            untrusted_text="excerpt",
        )
        citation = CitationSource(
            source_id=response.source_id,
            url=response.url,
            canonical_url=response.canonical_url,
            domain=response.domain,
            fetched_at=response.fetched_at,
            content_sha256=response.content_sha256,
            extractor="stub",
            provider="stub",
            title=response.title,
        )
        return FetchedDocument(
            response=response,
            full_text="excerpt",
            citation=citation,
            extractor="stub",
            content_type="text/html",
            status_code=200,
            warnings=[],
        )

    monkeypatch.setattr("smallctl.search_server.app.fetch_document", fake_fetch_document)
    monkeypatch.setattr("smallctl.search_server.cache.time.time", lambda: now[0])

    service = _make_service(tmp_path, negative_ttl_seconds=2)
    request = WebFetchRequest(url="https://example.com/story", max_chars=120)

    with pytest.raises(RuntimeError, match="temporary upstream failure"):
        asyncio.run(service.fetch(request))
    with pytest.raises(SearchServerError, match="temporary upstream failure"):
        asyncio.run(service.fetch(request))

    now[0] += 3
    response, full_text, citation = asyncio.run(service.fetch(request))

    assert calls["count"] == 2
    assert response.source_id == "websrc-cache-1"
    assert full_text == "excerpt"
    assert citation.extractor == "stub"


def test_normalize_search_results_dedupes_canonical_urls() -> None:
    results = [
        WebSearchResult(
            result_id="webres-1",
            title="Primary",
            url="https://example.com/story?ref=one",
            canonical_url="https://example.com/story",
            domain="example.com",
            snippet="one",
            provider="stub",
            rank=1,
        ),
        WebSearchResult(
            result_id="webres-2",
            title="Duplicate",
            url="https://example.com/story?ref=two",
            canonical_url="https://example.com/story",
            domain="example.com",
            snippet="two",
            provider="stub",
            rank=2,
        ),
    ]

    normalized = normalize_search_results(results, limit=10)

    assert [item.result_id for item in normalized] == ["webres-1"]


def test_searxng_recency_support_comes_from_config() -> None:
    provider = SearxNGProvider()

    assert provider.recency_support(SearchServerConfig(searxng_recency_support="strict")) == "strict"
    assert provider.recency_support(SearchServerConfig(searxng_recency_support="none")) == "none"
    assert provider.recency_support(SearchServerConfig(searxng_recency_support="unexpected")) == "best_effort"


def test_search_server_config_normalizes_searxng_recency_support() -> None:
    config = SearchServerConfig.from_mapping({"searxng_recency_support": " STRICT "})

    assert config.searxng_recency_support == "strict"


def test_parse_duckduckgo_html_unwraps_redirect_urls() -> None:
    html = """
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fforecast%3Fday%3D1&amp;rut=abc">
      Example Forecast
    </a>
    <a class="result__snippet">Detailed weather forecast</a>
    """

    results = _parse_duckduckgo_html(
        html,
        request=WebSearchRequest(query="forecast", limit=5),
    )

    assert len(results) == 1
    assert results[0].url == "https://example.com/forecast?day=1"
    assert results[0].canonical_url == "https://example.com/forecast?day=1"
    assert results[0].domain == "example.com"
