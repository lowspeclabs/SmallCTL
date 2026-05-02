from __future__ import annotations

import asyncio
import http.client
import json
import threading

import pytest

from smallctl.search_server.app import SearchServerError, SearchServerRuntime, SearchServerService
from smallctl.search_server.citations import now_iso
from smallctl.search_server.config import SearchServerConfig
from smallctl.search_server.models import CitationSource, WebFetchRequest, WebFetchResponse, WebSearchRequest, WebSearchResponse, WebSearchResult

_TOKEN_HEADER = "X-Smallctl-Search-Token"


def test_search_runtime_health_endpoint_is_local_and_requires_auth() -> None:
    runtime = SearchServerRuntime(config=SearchServerConfig())
    try:
        runtime.ensure_started()
        host, port = runtime.health()["bind_host"], runtime.health()["bind_port"]

        unauth_conn = http.client.HTTPConnection(host, port, timeout=2)
        unauth_conn.request("GET", "/health")
        unauth_response = unauth_conn.getresponse()
        unauth_payload = json.loads(unauth_response.read().decode("utf-8"))
        unauth_conn.close()

        auth_conn = http.client.HTTPConnection(host, port, timeout=2)
        auth_conn.request("GET", "/health", headers={_TOKEN_HEADER: runtime.token})
        auth_response = auth_conn.getresponse()
        auth_payload = json.loads(auth_response.read().decode("utf-8"))
        auth_conn.close()

        assert host == "127.0.0.1"
        assert int(port) > 0
        assert unauth_response.status == 401
        assert "invalid search service token" in unauth_payload["error"].lower()
        assert auth_response.status == 200
        assert auth_payload["bind_host"] == "127.0.0.1"
        assert int(auth_payload["bind_port"]) == int(port)
        assert auth_payload["version"] == runtime.version
    finally:
        runtime.stop()


def test_search_runtime_rejects_non_local_bind_host() -> None:
    runtime = SearchServerRuntime(config=SearchServerConfig(bind_host="0.0.0.0"))
    with pytest.raises(SearchServerError, match="127.0.0.1"):
        runtime.ensure_started()


def test_search_runtime_client_calls_flow_through_daemon(monkeypatch) -> None:
    seen: dict[str, object] = {}
    main_thread_id = threading.get_ident()

    async def fake_search(self, request: WebSearchRequest) -> WebSearchResponse:
        seen["search_query"] = request.query
        seen["search_thread_id"] = threading.get_ident()
        return WebSearchResponse(
            query=request.query,
            provider="stub",
            results=[
                WebSearchResult(
                    result_id="webres-daemon-1",
                    title="Daemon Result",
                    url="https://example.com/story",
                    canonical_url="https://example.com/story",
                    domain="example.com",
                    snippet="stub snippet",
                    provider="stub",
                    rank=1,
                )
            ],
            warnings=[],
        )

    async def fake_fetch(self, request: WebFetchRequest) -> tuple[WebFetchResponse, str, CitationSource]:
        seen["fetch_url"] = request.url
        seen["fetch_thread_id"] = threading.get_ident()
        response = WebFetchResponse(
            source_id="websrc-daemon-1",
            url=str(request.url or "https://example.com/story"),
            canonical_url=str(request.url or "https://example.com/story"),
            domain="example.com",
            title="Daemon Result",
            byline=None,
            published_at=None,
            fetched_at=now_iso(),
            content_type="text/html",
            content_sha256="hash",
            text_excerpt="stub excerpt",
            char_start=1,
            char_end=12,
            warnings=[],
            artifact_id=None,
            untrusted_text="stub excerpt",
        )
        citation = CitationSource(
            source_id="websrc-daemon-1",
            url=response.url,
            canonical_url=response.canonical_url,
            domain=response.domain,
            fetched_at=response.fetched_at,
            content_sha256=response.content_sha256,
            extractor="stub",
            provider="stub",
            title=response.title,
        )
        return response, "stub full text", citation

    monkeypatch.setattr(SearchServerService, "search", fake_search)
    monkeypatch.setattr(SearchServerService, "fetch", fake_fetch)

    runtime = SearchServerRuntime(config=SearchServerConfig())
    try:
        search_response = asyncio.run(
            runtime.search(
                WebSearchRequest(query="latest example"),
                token=runtime.token,
            )
        )
        fetch_response, full_text, citation = asyncio.run(
            runtime.fetch(
                WebFetchRequest(url="https://example.com/story", max_chars=200),
                token=runtime.token,
            )
        )

        assert search_response.provider == "stub"
        assert search_response.results[0].result_id == "webres-daemon-1"
        assert fetch_response.source_id == "websrc-daemon-1"
        assert full_text == "stub full text"
        assert citation.provider == "stub"
        assert seen["search_query"] == "latest example"
        assert seen["fetch_url"] == "https://example.com/story"
        assert seen["search_thread_id"] != main_thread_id
        assert seen["fetch_thread_id"] != main_thread_id
    finally:
        runtime.stop()
