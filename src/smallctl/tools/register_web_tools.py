from __future__ import annotations

from typing import Any, Awaitable, Callable

from . import web


def register_web_tools(
    *,
    register: Callable[[list[Any]], None],
    make_registration: Callable[..., Any],
    inject_state_and_harness: Callable[[Callable[..., Awaitable[dict[str, Any]]]], Callable[..., Awaitable[dict[str, Any]]]],
    network_read_profile: str,
) -> None:
    register(
        [
            make_registration(
                name="web_search",
                description="Search the public web for current or recent information and return normalized ranked results.",
                schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "domains": {"type": "array", "items": {"type": "string"}},
                        "recency_days": {"type": "integer"},
                        "limit": {"type": "integer", "default": 5},
                        "sort": {"type": "string", "enum": ["relevance", "recency"], "default": "relevance"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_harness(web.web_search),
                category="web",
                risk="network_read",
                allowed_modes={"chat", "loop", "planning"},
                profiles={network_read_profile},
            ),
            make_registration(
                name="web_fetch",
                description="Fetch a selected web result by returned fetch/result ID or safe URL, extract bounded readable text, and preserve provenance.",
                schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "result_id": {"type": "string"},
                        "fetch_id": {"type": "string"},
                        "max_chars": {"type": "integer", "default": 12000},
                        "extract_mode": {"type": "string", "enum": ["article", "text"], "default": "article"},
                    },
                    "additionalProperties": False,
                },
                handler=inject_state_and_harness(web.web_fetch),
                category="web",
                risk="network_read",
                allowed_modes={"chat", "loop", "planning"},
                profiles={network_read_profile},
            ),
        ]
    )
