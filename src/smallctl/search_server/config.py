from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SearchServerConfig:
    enabled: bool = True
    providers: list[str] = field(default_factory=lambda: ["brave", "searxng", "duckduckgo"])
    default_limit: int = 5
    max_limit: int = 10
    default_fetch_chars: int = 12000
    max_fetch_chars: int = 20000
    timeout_seconds: int = 15
    allowed_ports: tuple[int, ...] = (80, 443)
    allow_private_network_targets: tuple[str, ...] = ()
    max_redirects: int = 5
    max_searches_per_run: int = 6
    max_fetches_per_run: int = 4
    max_total_fetched_chars: int = 30000
    search_ttl_seconds: int = 900
    fetch_ttl_seconds: int = 3600
    negative_ttl_seconds: int = 120
    bind_host: str = "127.0.0.1"
    bind_port: int = 0
    cache_path: str | None = None
    brave_api_key: str | None = None
    brave_api_endpoint: str = "https://api.search.brave.com/res/v1/web/search"
    searxng_url: str | None = None
    searxng_recency_support: str = "best_effort"
    duckduckgo_url: str = "https://html.duckduckgo.com/html/"
    user_agent: str = "smallctl-web-search/0.1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None = None) -> "SearchServerConfig":
        raw = dict(data or {})
        if "providers" in raw and raw["providers"] not in (None, ""):
            providers_value = raw["providers"]
            if isinstance(providers_value, str):
                providers_iterable = [item.strip() for item in providers_value.split(",")]
            else:
                providers_iterable = providers_value
            raw["providers"] = [str(item).strip().lower() for item in providers_iterable if str(item).strip()]
        if "allowed_ports" in raw and raw["allowed_ports"] not in (None, ""):
            allowed_ports_value = raw["allowed_ports"]
            if isinstance(allowed_ports_value, str):
                allowed_ports_iterable = [item.strip() for item in allowed_ports_value.split(",")]
            else:
                allowed_ports_iterable = allowed_ports_value
            raw["allowed_ports"] = tuple(int(item) for item in allowed_ports_iterable)
        if "allow_private_network_targets" in raw and raw["allow_private_network_targets"] not in (None, ""):
            allow_private_targets_value = raw["allow_private_network_targets"]
            if isinstance(allow_private_targets_value, str):
                allow_private_targets_iterable = [item.strip() for item in allow_private_targets_value.split(",")]
            else:
                allow_private_targets_iterable = allow_private_targets_value
            raw["allow_private_network_targets"] = tuple(
                str(item).strip().lower()
                for item in allow_private_targets_iterable
                if str(item).strip()
            )
        if "cache_path" in raw and raw["cache_path"] not in (None, ""):
            raw["cache_path"] = str(raw["cache_path"])
        if "searxng_recency_support" in raw and raw["searxng_recency_support"] not in (None, ""):
            raw["searxng_recency_support"] = str(raw["searxng_recency_support"]).strip().lower()
        return cls(**raw)

    @classmethod
    def from_harness(cls, harness: Any) -> "SearchServerConfig":
        config = getattr(harness, "config", None)
        cwd = str(getattr(getattr(harness, "state", None), "cwd", "") or "").strip() or None
        cache_path = None
        if cwd:
            cache_path = str(Path(cwd).resolve() / ".smallctl" / "search_cache.sqlite3")
        payload = {"cache_path": cache_path}
        if config is not None:
            for key in (
                "enabled",
                "providers",
                "default_limit",
                "max_limit",
                "default_fetch_chars",
                "max_fetch_chars",
                "timeout_seconds",
                "allowed_ports",
                "allow_private_network_targets",
                "max_redirects",
                "max_searches_per_run",
                "max_fetches_per_run",
                "max_total_fetched_chars",
                "search_ttl_seconds",
                "fetch_ttl_seconds",
                "negative_ttl_seconds",
                "bind_host",
                "bind_port",
                "brave_api_key",
                "brave_api_endpoint",
                "searxng_url",
                "searxng_recency_support",
                "duckduckgo_url",
                "user_agent",
            ):
                if hasattr(config, key):
                    payload[key] = getattr(config, key)
        return cls.from_mapping(payload)

    def resolved_cache_path(self) -> Path:
        if self.cache_path:
            return Path(self.cache_path).expanduser().resolve()
        return (Path.cwd() / ".smallctl" / "search_cache.sqlite3").resolve()
