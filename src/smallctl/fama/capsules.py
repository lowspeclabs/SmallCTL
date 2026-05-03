from __future__ import annotations

from typing import Any

from ..context.policy import estimate_text_tokens
from .signals import current_step
from .state import active_mitigations, expire_mitigations


CAPSULE_TEXT: dict[str, str] = {
    "done_gate": "Before task_complete, satisfy the latest verifier/acceptance evidence or call task_fail with the blocker.",
    "acceptance_checklist_capsule": "Use the acceptance checklist and latest verifier result as the finish gate.",
    "remote_scope_capsule": "Remote scope is active; use SSH tools for remote paths and verify remotely before finishing.",
    "micro_plan_capsule": "Break the next move into one concrete action, expected evidence, and stop condition.",
    "evidence_reuse_capsule": "Use visible evidence before repeating the same read or command.",
    "write_session_recovery_capsule": "Resume the active write session with its required next section/tool before other edits.",
    "outline_only_recovery": "If blocked on a large write, outline the next section instead of rewriting the whole target.",
}


def render_fama_capsules(state: Any, *, token_budget: int = 180) -> list[str]:
    config = _scratch_config(state)
    if not _enabled(config):
        return []
    if token_budget <= 0:
        return []

    budget = max(1, int(config.get("capsule_token_budget") or token_budget or 180))
    budget = min(budget, max(1, int(token_budget)))
    expire_mitigations(state, step=current_step(state))
    lines: list[str] = []
    seen: set[str] = set()
    used_tokens = 0
    mitigations = sorted(active_mitigations(state), key=lambda item: (item.priority, item.activated_step, item.name))
    for mitigation in mitigations:
        line = CAPSULE_TEXT.get(mitigation.name)
        if not line or line in seen:
            continue
        line_tokens = estimate_text_tokens(line)
        if used_tokens + line_tokens > budget:
            continue
        lines.append(line)
        seen.add(line)
        used_tokens += line_tokens
        if len(lines) >= 5:
            break
    return lines


def _scratch_config(state: Any) -> dict[str, Any]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return {}
    payload = scratchpad.get("_fama_config")
    return payload if isinstance(payload, dict) else {}


def _enabled(config: dict[str, Any]) -> bool:
    if "enabled" in config and not bool(config.get("enabled")):
        return False
    return str(config.get("mode") or "lite").strip().lower() != "off"
