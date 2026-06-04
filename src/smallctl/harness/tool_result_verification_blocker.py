from __future__ import annotations

import hashlib
from typing import Any

from ..models.tool_result import ToolEnvelope
from .tool_result_verification_constants import _INTERACTIVE_PROMPT_RE, _REMOTE_APPLICATION_BLOCKERS
from .tool_result_verification_helpers import snip_text as _snip_text


def _extract_latest_execution_blocker(
    *,
    tool_name: str,
    command: str,
    target: str,
    exit_code: Any,
    failure_class: str,
    stdout: str,
    stderr: str,
    error: str,
    verdict: str,
) -> dict[str, Any] | None:
    if tool_name not in {"shell_exec", "ssh_exec"} or verdict == "pass":
        return None
    combined = "\n".join(str(part or "").strip() for part in (error, stderr, stdout) if str(part or "").strip())
    if not combined:
        return None
    blocker_class = ""
    salient = ""
    for candidate_class, pattern in _REMOTE_APPLICATION_BLOCKERS:
        match = pattern.search(combined)
        if match:
            blocker_class = candidate_class
            salient = _snip_text(match.group(0), limit=280)
            break
    if not blocker_class and _INTERACTIVE_PROMPT_RE.search(combined):
        blocker_class = "interactive_prompt"
        match = _INTERACTIVE_PROMPT_RE.search(combined)
        salient = _snip_text(match.group(0) if match else combined, limit=280)
    if not blocker_class:
        return None
    signature_seed = "|".join([tool_name, command, blocker_class, salient.lower()])
    return {
        "tool": tool_name,
        "command": command,
        "target": target,
        "exit_code": exit_code,
        "blocker_class": blocker_class,
        "failure_class": failure_class,
        "salient_error": salient,
        "is_interactive_prompt": blocker_class == "interactive_prompt",
        "signature": hashlib.sha1(signature_seed.encode("utf-8")).hexdigest()[:16],
    }


def _store_latest_execution_blocker(state: Any, blocker: dict[str, Any]) -> None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    prior = scratchpad.get("_latest_execution_blocker")
    prior_signature = str(prior.get("signature") or "") if isinstance(prior, dict) else ""
    new_signature = str(blocker.get("signature") or "")
    scratchpad["_latest_execution_blocker"] = blocker
    if prior_signature and new_signature and prior_signature != new_signature:
        counters = state.stagnation_counters if isinstance(getattr(state, "stagnation_counters", None), dict) else {}
        counters["no_progress"] = 0
        counters["repeat_command"] = 0
        state.stagnation_counters = counters
        scratchpad["_repair_last_failure_signature"] = new_signature
