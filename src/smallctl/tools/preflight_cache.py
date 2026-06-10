from __future__ import annotations

import hashlib
from typing import Any

from ..state_schema import PreflightResult


def _make_preflight_cache_key(
    *,
    tool_name: str,
    host: str,
    user: str,
    preflight_type: str,
) -> str:
    """Build a canonical cache key for a preflight check."""
    parts = [
        str(tool_name or "").strip().lower(),
        str(host or "").strip().lower(),
        str(user or "").strip().lower(),
        str(preflight_type or "").strip().lower(),
    ]
    return "|".join(parts)


def _make_evidence_hash(*pieces: str) -> str:
    """Build a short evidence hash from ordered string pieces."""
    payload = "\n".join(str(p) for p in pieces)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def get_preflight_result(
    state: Any,
    *,
    tool_name: str,
    host: str,
    user: str,
    preflight_type: str,
    expected_evidence_hash: str = "",
) -> PreflightResult | None:
    """Return a cached preflight result if present and evidence matches."""
    if state is None:
        return None
    cache = getattr(state, "preflight_cache", None)
    if not isinstance(cache, dict):
        return None
    key = _make_preflight_cache_key(
        tool_name=tool_name,
        host=host,
        user=user,
        preflight_type=preflight_type,
    )
    result = cache.get(key)
    if not isinstance(result, PreflightResult):
        return None
    if expected_evidence_hash and result.evidence_hash != expected_evidence_hash:
        return None
    return result


def set_preflight_result(
    state: Any,
    *,
    tool_name: str,
    host: str,
    user: str,
    preflight_type: str,
    passed: bool,
    evidence_hash: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Store a preflight result in the unified cache."""
    if state is None:
        return
    if not hasattr(state, "preflight_cache") or state.preflight_cache is None:
        state.preflight_cache = {}
    key = _make_preflight_cache_key(
        tool_name=tool_name,
        host=host,
        user=user,
        preflight_type=preflight_type,
    )
    state.preflight_cache[key] = PreflightResult(
        preflight_type=preflight_type,
        host=str(host or ""),
        user=str(user or ""),
        passed=passed,
        evidence_hash=evidence_hash,
        metadata=dict(metadata or {}),
    )
    if hasattr(state, "touch"):
        state.touch()


def invalidate_preflight_cache(
    state: Any,
    *,
    host: str | None = None,
    user: str | None = None,
    preflight_type: str | None = None,
    all_for_host: bool = False,
) -> None:
    """Remove matching entries from the preflight cache.

    Parameters match by substring; pass ``None`` to match any value.
    ``all_for_host`` removes every entry whose host matches.
    """
    if state is None:
        return
    cache = getattr(state, "preflight_cache", None)
    if not isinstance(cache, dict):
        return
    to_remove: list[str] = []
    for key, result in cache.items():
        if not isinstance(result, PreflightResult):
            to_remove.append(key)
            continue
        if all_for_host and host is not None:
            if result.host == host:
                to_remove.append(key)
                continue
        matched = True
        if host is not None and result.host != host:
            matched = False
        if user is not None and result.user != user:
            matched = False
        if preflight_type is not None and result.preflight_type != preflight_type:
            matched = False
        if matched:
            to_remove.append(key)
    for key in to_remove:
        cache.pop(key, None)
    if to_remove and hasattr(state, "touch"):
        state.touch()


def invalidate_deb822_on_sources_change(state: Any, path: str) -> None:
    """Invalidate deb822 preflight cache when an apt sources file is modified."""
    if state is None:
        return
    normalized = str(path or "").strip()
    if "/etc/apt/sources.list" in normalized or "/etc/apt/sources.list.d/" in normalized:
        invalidate_preflight_cache(state, preflight_type="deb822")


def invalidate_preflight_on_phase_change(state: Any, previous_phase: str, current_phase: str) -> None:
    """Invalidate preflight cache when phase transitions from execute/repair back to explore."""
    if state is None:
        return
    if previous_phase in {"execute", "repair"} and current_phase == "explore":
        invalidate_preflight_cache(state)
