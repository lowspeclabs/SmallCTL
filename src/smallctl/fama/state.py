from __future__ import annotations

from typing import Any

from .signals import (
    ActiveMitigation,
    get_fama_state,
    mitigation_from_dict,
    mitigation_to_dict,
)


def active_mitigations(state: Any) -> list[ActiveMitigation]:
    payload = get_fama_state(state)
    normalized: list[ActiveMitigation] = []
    for item in payload.get("active_mitigations", []):
        if not isinstance(item, dict):
            continue
        mitigation = mitigation_from_dict(item)
        if mitigation is not None:
            normalized.append(mitigation)
    return normalized


def activate_mitigations(
    state: Any,
    mitigations: list[ActiveMitigation],
    *,
    max_active: int = 2,
) -> list[ActiveMitigation]:
    if not mitigations:
        return []
    by_name = {item.name: item for item in active_mitigations(state)}
    changed: list[ActiveMitigation] = []
    for mitigation in mitigations:
        previous = by_name.get(mitigation.name)
        if previous is None or previous.expires_after_step != mitigation.expires_after_step:
            changed.append(mitigation)
        by_name[mitigation.name] = mitigation
    ordered = sorted(by_name.values(), key=lambda item: (item.priority, item.activated_step, item.name))
    limit = max(1, int(max_active or 1))
    get_fama_state(state)["active_mitigations"] = [mitigation_to_dict(item) for item in ordered[-limit:]]
    return changed


def expire_mitigations(state: Any, *, step: int) -> list[ActiveMitigation]:
    kept: list[ActiveMitigation] = []
    expired: list[ActiveMitigation] = []
    for mitigation in active_mitigations(state):
        if int(step) > mitigation.expires_after_step:
            expired.append(mitigation)
        else:
            kept.append(mitigation)
    get_fama_state(state)["active_mitigations"] = [mitigation_to_dict(item) for item in kept]
    return expired


def clear_mitigations(state: Any, names: set[str], *, reason: str) -> list[ActiveMitigation]:
    del reason
    normalized_names = {str(name).strip() for name in names if str(name).strip()}
    kept: list[ActiveMitigation] = []
    cleared: list[ActiveMitigation] = []
    for mitigation in active_mitigations(state):
        if mitigation.name in normalized_names:
            cleared.append(mitigation)
        else:
            kept.append(mitigation)
    get_fama_state(state)["active_mitigations"] = [mitigation_to_dict(item) for item in kept]
    return cleared


def active_mitigation_names(state: Any) -> set[str]:
    return {item.name for item in active_mitigations(state)}
