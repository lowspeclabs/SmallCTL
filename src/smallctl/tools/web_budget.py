from __future__ import annotations

from typing import Any

from ..search_server.app import SearchServerError
from ..search_server.config import SearchServerConfig


def mark_web_fetch_budget_exhausted(state: Any, error: str) -> None:
    lowered = str(error or "").strip().lower()
    if "web fetch" not in lowered or "budget exhausted" not in lowered:
        return
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    scratchpad["_web_fetch_budget_exhausted"] = {
        "error": str(error),
        "terminal": True,
    }


def budget_state(state: Any) -> dict[str, Any]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    budget = scratchpad.get("_web_budget")
    if not isinstance(budget, dict):
        budget = {"searches_used": 0, "fetches_used": 0, "total_fetched_chars": 0}
        scratchpad["_web_budget"] = budget
    return budget


def budget_remaining(state: Any, config: SearchServerConfig) -> dict[str, int]:
    budget = budget_state(state)
    return {
        "searches_remaining": max(0, config.max_searches_per_run - int(budget.get("searches_used", 0))),
        "fetches_remaining": max(0, config.max_fetches_per_run - int(budget.get("fetches_used", 0))),
        "chars_remaining": max(0, config.max_total_fetched_chars - int(budget.get("total_fetched_chars", 0))),
    }


def ensure_budget(state: Any, *, config: SearchServerConfig, action: str, chars: int = 0) -> None:
    budget = budget_state(state)
    if action == "search":
        used = int(budget.get("searches_used", 0))
        if used >= config.max_searches_per_run:
            raise SearchServerError(
                f"Web search budget exhausted for this run ({used}/{config.max_searches_per_run} searches used)."
            )
        budget["searches_used"] = used + 1
    elif action == "fetch":
        used = int(budget.get("fetches_used", 0))
        if used >= config.max_fetches_per_run:
            raise SearchServerError(
                f"Web fetch budget exhausted for this run ({used}/{config.max_fetches_per_run} fetches used)."
            )
        budget["fetches_used"] = used + 1
    elif action == "fetch_chars":
        used = int(budget.get("total_fetched_chars", 0))
        if used + int(chars) > config.max_total_fetched_chars:
            remaining = max(0, config.max_total_fetched_chars - used)
            raise SearchServerError(
                f"Web fetch character budget exhausted for this run. "
                f"Requested {int(chars)} chars but only {remaining} chars remain "
                f"({used}/{config.max_total_fetched_chars} total chars used)."
            )
        budget["total_fetched_chars"] = used + int(chars)
