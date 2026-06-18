from __future__ import annotations

from typing import Protocol

from .config import SearchServerConfig
from .models import WebSearchRequest, WebSearchResult


class SearchProvider(Protocol):
    name: str

    def is_enabled(self, config: SearchServerConfig) -> bool: ...

    def recency_support(self, config: SearchServerConfig) -> str: ...

    async def search(self, request: WebSearchRequest, config: SearchServerConfig) -> list[WebSearchResult]: ...
