from __future__ import annotations

from typing import Any


def resolve_fetch_selector(
    state: Any,
    *,
    url: str | None,
    result_id: str | None,
    fetch_id: str | None,
) -> tuple[str | None, str | None, str | None, list[str]]:
    """Resolve which fetch target to use when multiple are provided."""
    requested_url = str(url or "").strip() or None
    requested_result_id = str(result_id or "").strip() or None
    requested_fetch_id = str(fetch_id or "").strip() or None
    selector_count = sum(1 for value in (requested_url, requested_result_id, requested_fetch_id) if value)
    warnings: list[str] = []
    if selector_count == 0:
        return None, None, None, warnings
    if selector_count > 1:
        if requested_fetch_id:
            requested_url = None
            requested_result_id = None
            warnings.append("Warning: Multiple target arguments provided. Using fetch_id and ignoring others.")
        elif requested_result_id:
            requested_url = None
            requested_fetch_id = None
            warnings.append("Warning: Multiple target arguments provided. Using result_id and ignoring others.")
        else:
            requested_result_id = None
            requested_fetch_id = None
            warnings.append("Warning: Multiple target arguments provided. Using url and ignoring others.")
    return requested_url, requested_result_id, requested_fetch_id, warnings
