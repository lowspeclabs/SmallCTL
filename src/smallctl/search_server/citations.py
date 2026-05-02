from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(str(url or "").strip())
    netloc = parsed.hostname or ""
    if parsed.port and parsed.port not in {80, 443}:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme.lower(), netloc.lower(), parsed.path or "/", parsed.query, ""))


def domain_from_url(url: str) -> str:
    parsed = urlsplit(str(url or "").strip())
    return (parsed.hostname or "").lower()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def content_sha256(content: str | bytes) -> str:
    raw = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(raw).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(str(part or "") for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def make_result_id(*, provider: str, canonical_url: str, rank: int) -> str:
    return _stable_id("webres", provider, canonical_url, str(rank))


def make_source_id(*, provider: str, canonical_url: str, fetched_at: str) -> str:
    return _stable_id("websrc", provider, canonical_url, fetched_at)


@dataclass(slots=True)
class CitationEnvelope:
    source_id: str
    result_id: str
    canonical_url: str
    fetched_at: str
    provider: str
