from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as_optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


@dataclass(slots=True)
class WebSearchRequest:
    query: str
    domains: list[str] | None = None
    recency_days: int | None = None
    limit: int = 5
    sort: str = "relevance"

    def __post_init__(self) -> None:
        self.query = str(self.query or "").strip()
        self.domains = [str(item).strip() for item in (self.domains or []) if str(item).strip()] or None
        self.recency_days = None if self.recency_days in (None, "") else max(1, int(self.recency_days))
        self.limit = max(1, int(self.limit or 1))
        self.sort = str(self.sort or "relevance").strip().lower() or "relevance"

    def cache_key(self) -> str:
        domains = ",".join(sorted(self.domains or []))
        recency = "" if self.recency_days is None else str(self.recency_days)
        return f"{self.sort}|{recency}|{self.limit}|{domains}|{self.query.lower()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "domains": list(self.domains) if self.domains else None,
            "recency_days": self.recency_days,
            "limit": self.limit,
            "sort": self.sort,
        }


@dataclass(slots=True)
class WebSearchResult:
    result_id: str
    title: str
    url: str
    canonical_url: str
    domain: str
    snippet: str
    published_at: str | None = None
    provider: str = ""
    rank: int = 0

    def __post_init__(self) -> None:
        self.result_id = str(self.result_id or "").strip()
        self.title = str(self.title or "").strip()
        self.url = str(self.url or "").strip()
        self.canonical_url = str(self.canonical_url or self.url).strip()
        self.domain = str(self.domain or "").strip()
        self.snippet = str(self.snippet or "").strip()
        self.published_at = _as_optional_str(self.published_at)
        self.provider = str(self.provider or "").strip()
        self.rank = int(self.rank or 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "title": self.title,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "domain": self.domain,
            "snippet": self.snippet,
            "published_at": self.published_at,
            "provider": self.provider,
            "rank": self.rank,
        }


@dataclass(slots=True)
class WebSearchResponse:
    query: str
    provider: str
    results: list[WebSearchResult] = field(default_factory=list)
    recency_requested_days: int | None = None
    recency_enforced: bool = False
    recency_support: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "provider": self.provider,
            "results": [result.to_dict() for result in self.results],
            "recency_requested_days": self.recency_requested_days,
            "recency_enforced": self.recency_enforced,
            "recency_support": self.recency_support,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class WebFetchRequest:
    url: str | None = None
    result_id: str | None = None
    max_chars: int = 12000
    extract_mode: str = "article"

    def __post_init__(self) -> None:
        self.url = _as_optional_str(self.url)
        self.result_id = _as_optional_str(self.result_id)
        self.max_chars = max(1, int(self.max_chars or 1))
        self.extract_mode = str(self.extract_mode or "article").strip().lower() or "article"

    def cache_key(self) -> str:
        if self.result_id:
            return f"result:{self.result_id}|{self.max_chars}|{self.extract_mode}"
        return f"url:{self.url or ''}|{self.max_chars}|{self.extract_mode}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "result_id": self.result_id,
            "max_chars": self.max_chars,
            "extract_mode": self.extract_mode,
        }


@dataclass(slots=True)
class WebFetchResponse:
    source_id: str
    url: str
    canonical_url: str
    domain: str
    title: str
    byline: str | None
    published_at: str | None
    fetched_at: str
    content_type: str
    content_sha256: str
    text_excerpt: str
    char_start: int
    char_end: int
    artifact_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    untrusted_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "domain": self.domain,
            "title": self.title,
            "byline": self.byline,
            "published_at": self.published_at,
            "fetched_at": self.fetched_at,
            "content_type": self.content_type,
            "content_sha256": self.content_sha256,
            "text_excerpt": self.text_excerpt,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "artifact_id": self.artifact_id,
            "warnings": list(self.warnings),
            "untrusted_text": self.untrusted_text,
        }


@dataclass(slots=True)
class CitationSource:
    source_id: str
    url: str
    canonical_url: str
    domain: str
    fetched_at: str
    published_at: str | None = None
    content_sha256: str = ""
    extractor: str = ""
    provider: str = ""
    artifact_id: str | None = None
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "domain": self.domain,
            "fetched_at": self.fetched_at,
            "published_at": self.published_at,
            "content_sha256": self.content_sha256,
            "extractor": self.extractor,
            "provider": self.provider,
            "artifact_id": self.artifact_id,
            "title": self.title,
        }


@dataclass(slots=True)
class FetchContentSlice:
    text: str
    char_start: int
    char_end: int
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "truncated": self.truncated,
        }
