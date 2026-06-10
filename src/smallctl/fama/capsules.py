from __future__ import annotations

from typing import Any

from ..context.policy import estimate_text_tokens
from .signals import current_step
from .state import active_mitigations, expire_mitigations


CAPSULE_TEXT: dict[str, str] = {
    "done_gate": "Before task_complete, satisfy the latest verifier/acceptance evidence or call task_fail with the blocker.",
    "acceptance_checklist_capsule": "Use the acceptance checklist and latest verifier result as the finish gate.",
    "zero_test_recovery_capsule": "Latest test evidence discovered 0 tests; add real test classes/functions and rerun that test command before finishing.",
    "remote_scope_capsule": "Remote scope is active; use SSH tools for remote paths and verify remotely before finishing.",
    "remote_verification_pending_capsule": "A remote mutation is pending verification; read back the remote path/state instead of rediscovering local paths.",
    "micro_plan_capsule": "Break the next move into one concrete action, expected evidence, and stop condition.",
    "evidence_reuse_capsule": "Use visible evidence before repeating the same read or command.",
    "tool_exposure_narrowing": "Do not repeat the same tool call; use prior output or switch to one different tool/action that can create new evidence.",
    "evidence_gathering_needed": "Gather a bounded read-only ToolPlan evidence pass before patching or finishing.",
    "evidence_gathering_needed_hard_route": "",
    "write_session_recovery_capsule": "Resume the active write session with its required next section/tool before other edits.",
    "outline_only_recovery": "If blocked on a large write, outline the next section instead of rewriting the whole target.",
    "repair_debug_scaffold": "REPAIR MODE: You have already read the failing file. The bugs are in code you wrote. Do NOT read the file again. Emit ONE mutation (file_patch/file_write/ast_patch) this turn, then run the verifier.",
    "preflight_contradiction_capsule": "A preflight validation passed but the gate is still blocking. Do not retry the same validator; escalate or ask for human guidance.",
    "repeated_remote_installer_failure_capsule": "The remote installer has failed repeatedly. Verify the remote environment state (apt sources, DNS, python3), repair any broken state, and only then retry.",
    "preexisting_state_as_success_capsule": "Distinguish 'state already existed' from 'I caused the state'. Verify that your actions produced the intended outcome, not that it was already present.",
    "repeated_shell_failure_collapsed": "Multiple identical shell failures were collapsed in the transcript. Do NOT repeat the same command. Use the preserved last occurrence as evidence, diagnose the root cause, and try a fundamentally different approach.",
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
    mitigation_names = {m.name for m in mitigations}
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
    # Inject repair-phase debug scaffold when in repair with stuck signals
    state_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    if state_phase == "repair" and mitigation_names & {"done_gate", "micro_plan_capsule", "tool_exposure_narrowing"}:
        scaffold = CAPSULE_TEXT.get("repair_debug_scaffold")
        if scaffold and scaffold not in seen:
            scaffold_tokens = estimate_text_tokens(scaffold)
            if used_tokens + scaffold_tokens <= budget and len(lines) < 5:
                lines.append(scaffold)
                seen.add(scaffold)
                used_tokens += scaffold_tokens
    # Inject repeated-shell-failure collapsed nudge when the assembler has compacted failures
    recent_messages = getattr(state, "recent_messages", None) or []
    if isinstance(recent_messages, list):
        for msg in recent_messages:
            content = str(getattr(msg, "content", "") or "").strip()
            if "repeated shell_exec failures with identical error collapsed" in content:
                collapse_capsule = CAPSULE_TEXT.get("repeated_shell_failure_collapsed")
                if collapse_capsule and collapse_capsule not in seen:
                    collapse_tokens = estimate_text_tokens(collapse_capsule)
                    if used_tokens + collapse_tokens <= budget and len(lines) < 5:
                        lines.append(collapse_capsule)
                        seen.add(collapse_capsule)
                        used_tokens += collapse_tokens
                break

    # Track empty streak for health monitoring
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        if not lines:
            scratchpad["_fama_empty_streak"] = int(scratchpad.get("_fama_empty_streak", 0) or 0) + 1
        else:
            scratchpad["_fama_empty_streak"] = 0
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


def fama_capsule_health_warning(state: Any) -> str | None:
    """Return a warning if FAMA capsules have been empty for 3+ consecutive prompts."""
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    streak = int(scratchpad.get("_fama_empty_streak", 0) or 0)
    if streak < 3:
        return None
    return f"no mitigations have been rendered for {streak} consecutive prompts"


def fama_fallback_recovery_guidance(state: Any) -> list[str]:
    """Provide fallback recovery guidance when FAMA capsules are empty for 3+ prompts."""
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return []
    streak = int(scratchpad.get("_fama_empty_streak", 0) or 0)
    if streak < 3:
        return []
    failure_class = str(getattr(state, "last_failure_class", "") or "").strip().lower()
    lines: list[str] = []
    if "path" in failure_class or failure_class == "wrong_path":
        lines.append("Path failure detected but no FAMA mitigation active. Verify the exact path exists and retry.")
    elif "verifier" in failure_class:
        lines.append("Verifier failure detected but no FAMA mitigation active. Read the failing output and patch one narrow cause.")
    elif "backend" in failure_class:
        lines.append("Backend stream failure detected. Retry with a smaller, explicit next action.")
    else:
        lines.append("No FAMA mitigation is active. Break the next move into one concrete action, expected evidence, and stop condition.")
    return lines
