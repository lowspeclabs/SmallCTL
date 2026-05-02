from __future__ import annotations

import asyncio
import http.client
import json
import secrets
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency path
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from .cache import SearchCache
from .citations import now_iso
from .config import SearchServerConfig
from .fetch import fetch_document
from .models import CitationSource, WebFetchRequest, WebFetchResponse, WebSearchRequest, WebSearchResponse, WebSearchResult
from .providers import search_with_providers

_RUNTIME_VERSION = "2"
_TOKEN_HEADER = "X-Smallctl-Search-Token"


class SearchServerError(RuntimeError):
    def __init__(self, message: str, *, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


class SearchServerAuthError(SearchServerError):
    pass


@dataclass(slots=True)
class SearchServerService:
    config: SearchServerConfig
    cache: SearchCache
    result_index: dict[str, WebSearchResult] = field(default_factory=dict)
    started_at: str = field(default_factory=now_iso)

    @classmethod
    def create(cls, config: SearchServerConfig) -> "SearchServerService":
        return cls(config=config, cache=SearchCache(config.resolved_cache_path()))

    def health(self) -> dict[str, Any]:
        return {
            "ok": bool(self.config.enabled),
            "started_at": self.started_at,
            "provider_order": list(self.config.providers),
            "cache_path": str(self.cache.path),
        }

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        if not self.config.enabled:
            raise SearchServerError("Web search is disabled.")
        if not request.query:
            raise SearchServerError("Query is required.")
        key = request.cache_key()
        cached = self.cache.get(kind="search", key=key)
        if cached is not None:
            results = [WebSearchResult(**item) for item in cached.payload.get("results", [])]
            for result in results:
                self.result_index[result.result_id] = result
            return WebSearchResponse(
                query=cached.payload.get("query", request.query),
                provider=cached.payload.get("provider", "cache"),
                results=results,
                recency_requested_days=cached.payload.get("recency_requested_days"),
                recency_enforced=bool(cached.payload.get("recency_enforced", False)),
                recency_support=str(cached.payload.get("recency_support", "unknown") or "unknown"),
                warnings=list(cached.payload.get("warnings", [])),
            )

        results, warnings, provider_name, recency_support = await search_with_providers(request, config=self.config)
        for result in results:
            self.result_index[result.result_id] = result
        recency_enforced = bool(request.recency_days) and recency_support == "strict"
        recency_warning = _recency_warning(
            requested_days=request.recency_days,
            provider_name=provider_name,
            recency_support=recency_support,
        )
        response = WebSearchResponse(
            query=request.query,
            provider=provider_name,
            results=results,
            recency_requested_days=request.recency_days,
            recency_enforced=recency_enforced,
            recency_support=recency_support,
            warnings=warnings + ([recency_warning] if recency_warning else []),
        )
        self.cache.set(
            kind="search",
            key=key,
            payload=response.to_dict(),
            ttl_seconds=self.config.negative_ttl_seconds if not results else self.config.search_ttl_seconds,
            negative=not bool(results),
        )
        return response

    async def fetch(self, request: WebFetchRequest) -> tuple[WebFetchResponse, str, CitationSource]:
        if not self.config.enabled:
            raise SearchServerError("Web search is disabled.")
        if bool(request.url) == bool(request.result_id):
            raise SearchServerError("Provide exactly one of url or result_id.")
        if request.max_chars > self.config.max_fetch_chars:
            raise SearchServerError(f"max_chars cannot exceed {self.config.max_fetch_chars}.")

        target_url = request.url
        source: WebSearchResult | None = None
        if request.result_id:
            source = self.result_index.get(request.result_id)
            if source is None:
                raise SearchServerError(f"Unknown result_id: {request.result_id}")
            target_url = source.url

        assert target_url is not None
        key = request.cache_key()
        cached = self.cache.get(kind="fetch", key=key)
        if cached is not None:
            if cached.negative:
                raise SearchServerError(str(cached.payload.get("error") or "Cached web fetch failure."))
            response = WebFetchResponse(**cached.payload["response"])
            citation = CitationSource(**cached.payload["citation"])
            return response, str(cached.payload.get("full_text", "")), citation

        try:
            fetched = await fetch_document(request, config=self.config, url=target_url)
        except Exception as exc:
            self.cache.set(
                kind="fetch",
                key=key,
                payload={
                    "error": str(exc),
                    "url": target_url,
                    "result_id": request.result_id,
                },
                ttl_seconds=self.config.negative_ttl_seconds,
                negative=True,
            )
            raise
        response = fetched.response
        full_text = fetched.full_text
        citation = fetched.citation
        if source is not None:
            response.source_id = source.result_id
            citation.source_id = source.result_id
            response.url = source.url
            response.canonical_url = source.canonical_url
            response.domain = source.domain
        self.cache.set(
            kind="fetch",
            key=key,
            payload={
                "response": response.to_dict(),
                "citation": citation.to_dict(),
                "full_text": full_text,
                "status_code": fetched.status_code,
                "extractor": fetched.extractor,
                "content_type": fetched.content_type,
            },
            ttl_seconds=self.config.fetch_ttl_seconds,
            negative=False,
        )
        return response, full_text, citation


class _SearchHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], runtime: "SearchServerRuntime") -> None:
        self.runtime = runtime
        super().__init__(server_address, _SearchRequestHandler)


class _SearchRequestHandler(BaseHTTPRequestHandler):
    server_version = "smallctl-search/0.1"

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover - keep daemon quiet
        del format, args
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._write_json(404, {"error": "Not found."})
            return
        try:
            runtime = self._runtime()
            runtime._assert_token(self.headers.get(_TOKEN_HEADER, ""))
            self._write_json(200, runtime._daemon_health())
        except SearchServerAuthError as exc:
            self._write_json(401, {"error": str(exc), "metadata": exc.metadata})
        except SearchServerError as exc:
            self._write_json(400, {"error": str(exc), "metadata": exc.metadata})
        except Exception as exc:  # pragma: no cover - defensive
            self._write_json(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/search", "/fetch"}:
            self._write_json(404, {"error": "Not found."})
            return
        try:
            runtime = self._runtime()
            runtime._assert_token(self.headers.get(_TOKEN_HEADER, ""))
            payload = self._read_json()
            if self.path == "/search":
                request = WebSearchRequest(**payload)
                response = asyncio.run(runtime.service.search(request))
                self._write_json(200, response.to_dict())
                return
            request = WebFetchRequest(**payload)
            response, full_text, citation = asyncio.run(runtime.service.fetch(request))
            self._write_json(
                200,
                {
                    "response": response.to_dict(),
                    "full_text": full_text,
                    "citation": citation.to_dict(),
                },
            )
        except SearchServerAuthError as exc:
            self._write_json(401, {"error": str(exc), "metadata": exc.metadata})
        except SearchServerError as exc:
            self._write_json(400, {"error": str(exc), "metadata": exc.metadata})
        except Exception as exc:
            self._write_json(500, {"error": str(exc)})

    def _runtime(self) -> "SearchServerRuntime":
        server = self.server
        if not isinstance(server, _SearchHTTPServer):  # pragma: no cover - defensive
            raise SearchServerError("Search server is not available.")
        return server.runtime

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise SearchServerError("Invalid JSON payload.") from exc
        if not isinstance(payload, dict):
            raise SearchServerError("JSON payload must be an object.")
        return payload

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True, default=str).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class SearchServerRuntime:
    def __init__(self, *, config: SearchServerConfig) -> None:
        self.config = config
        self.token = secrets.token_urlsafe(32)
        self.service = SearchServerService.create(config)
        self.started_at = now_iso()
        self.version = _RUNTIME_VERSION
        self._stopped = False
        self._lock = threading.RLock()
        self._server: _SearchHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._base_url: str | None = None

    @property
    def base_url(self) -> str:
        self.ensure_started()
        assert self._base_url is not None
        return self._base_url

    def health(self) -> dict[str, Any]:
        with self._lock:
            host, port = self._bound_address()
            return self.service.health() | {
                "ok": not self._stopped and self._thread is not None and self._thread.is_alive(),
                "token_required": True,
                "version": self.version,
                "bind_host": host,
                "bind_port": port,
            }

    def ensure_started(self) -> None:
        with self._lock:
            if self._stopped:
                self._restart_locked()
                return
            if self._server is None or self._thread is None or not self._thread.is_alive():
                self._start_locked()
                return
            health = self._probe_health_locked()
            if not health.get("ok") or str(health.get("version") or "") != self.version:
                self._restart_locked()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    async def get_health(self, *, token: str) -> dict[str, Any]:
        self._assert_token(token)
        return await self._request_json("GET", "/health")

    async def search(self, request: WebSearchRequest, *, token: str) -> WebSearchResponse:
        self._assert_token(token)
        payload = await self._request_json("POST", "/search", request.to_dict())
        return WebSearchResponse(
            query=str(payload.get("query") or ""),
            provider=str(payload.get("provider") or ""),
            results=[WebSearchResult(**item) for item in payload.get("results", [])],
            recency_requested_days=payload.get("recency_requested_days"),
            recency_enforced=bool(payload.get("recency_enforced", False)),
            recency_support=str(payload.get("recency_support", "unknown") or "unknown"),
            warnings=list(payload.get("warnings", [])),
        )

    async def fetch(self, request: WebFetchRequest, *, token: str) -> tuple[WebFetchResponse, str, CitationSource]:
        self._assert_token(token)
        payload = await self._request_json("POST", "/fetch", request.to_dict())
        response = WebFetchResponse(**payload["response"])
        citation = CitationSource(**payload["citation"])
        return response, str(payload.get("full_text", "")), citation

    def _assert_token(self, token: str) -> None:
        if token != self.token:
            raise SearchServerAuthError("Invalid search service token.")

    async def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if httpx is None:
            raise SearchServerError("Dependency missing: httpx")
        self.ensure_started()
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(
                    timeout=float(self.config.timeout_seconds),
                    follow_redirects=False,
                    trust_env=False,
                ) as client:
                    response = await client.request(
                        method,
                        f"{self.base_url}{path}",
                        json=payload,
                        headers={_TOKEN_HEADER: self.token},
                    )
                data = response.json()
                if response.status_code >= 400:
                    raise SearchServerError(str(data.get("error") or f"Search service HTTP {response.status_code}"))
                if not isinstance(data, dict):
                    raise SearchServerError("Search service returned a non-object payload.")
                return data
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                with self._lock:
                    self._restart_locked()
                if attempt == 0:
                    continue
                raise SearchServerError(f"Search service unavailable: {exc}") from exc
        if last_error is not None:  # pragma: no cover - defensive
            raise SearchServerError(f"Search service unavailable: {last_error}") from last_error
        raise SearchServerError("Search service request failed.")

    def _daemon_health(self) -> dict[str, Any]:
        host, port = self._bound_address()
        return self.service.health() | {
            "ok": not self._stopped,
            "token_required": True,
            "version": self.version,
            "bind_host": host,
            "bind_port": port,
        }

    def _validate_bind_host(self) -> str:
        host = str(self.config.bind_host or "127.0.0.1").strip() or "127.0.0.1"
        if host != "127.0.0.1":
            raise SearchServerError("Search daemon bind_host must be 127.0.0.1.")
        return host

    def _bound_address(self) -> tuple[str, int]:
        if self._server is None:
            return self._validate_bind_host(), int(self.config.bind_port or 0)
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def _start_locked(self) -> None:
        self._stop_locked()
        host = self._validate_bind_host()
        port = int(self.config.bind_port or 0)
        self._stopped = False
        self._server = _SearchHTTPServer((host, port), self)
        self._base_url = f"http://{host}:{int(self._server.server_address[1])}"
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="smallctl-search-daemon",
            daemon=True,
        )
        self._thread.start()
        health = self._probe_health_locked()
        if not health.get("ok"):
            self._stop_locked()
            raise SearchServerError("Search daemon failed to start.")

    def _restart_locked(self) -> None:
        self._stop_locked()
        self._start_locked()

    def _stop_locked(self) -> None:
        server = self._server
        thread = self._thread
        self._stopped = True
        self._server = None
        self._thread = None
        self._base_url = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _probe_health_locked(self) -> dict[str, Any]:
        if self._base_url is None:
            return {"ok": False}
        host, port = self._bound_address()
        try:
            conn = http.client.HTTPConnection(host, port, timeout=max(1, int(self.config.timeout_seconds)))
            conn.request("GET", "/health", headers={_TOKEN_HEADER: self.token})
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()
        except Exception:
            return {"ok": False}
        if response.status != 200 or not isinstance(payload, dict):
            return {"ok": False}
        return payload


_RUNTIMES: dict[str, SearchServerRuntime] = {}


def get_search_runtime(owner: Any, *, config: SearchServerConfig | None = None) -> SearchServerRuntime:
    state = getattr(owner, "state", owner)
    cwd = str(getattr(state, "cwd", "") or "").strip() or str(Path.cwd())
    key = f"{cwd}|{str(getattr(state, 'thread_id', '') or '')}"
    runtime = _RUNTIMES.get(key)
    if runtime is None:
        runtime = SearchServerRuntime(config=config or SearchServerConfig.from_harness(owner))
        _RUNTIMES[key] = runtime
    elif config is not None and runtime.config != config:
        runtime.stop()
        runtime = SearchServerRuntime(config=config)
        _RUNTIMES[key] = runtime
    runtime.ensure_started()
    return runtime


def _recency_warning(*, requested_days: int | None, provider_name: str, recency_support: str) -> str | None:
    if not requested_days:
        return None
    if recency_support == "strict":
        return None
    if recency_support == "best_effort":
        return f"{provider_name} treated recency_days={requested_days} as best-effort only and may include older results."
    if recency_support == "none":
        return f"{provider_name} could not enforce recency_days={requested_days}; treat recency as unfiltered."
    return f"{provider_name} did not report whether recency_days={requested_days} was enforced."
