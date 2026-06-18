from __future__ import annotations

import json
import re
from typing import Any

from ..state import json_safe_value

_COMPLETION_FACT_MARKERS = ("[COMPLETED]", "[DONE]", "[SUCCESS]", "task complete", "successfully removed", "successfully uninstalled")
_TERMINAL_PROSE_STRONG_MARKERS = (
    "task complete",
    "task completed",
    "task is essentially complete",
    "the task is complete",
    "final summary",
    "final answer",
    "completed successfully",
)
_FENCED_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def working_memory_signals_completion(harness: Any) -> bool:
    blocker = harness.state.scratchpad.get("_unresolved_missing_input_file")
    if isinstance(blocker, dict) and str(blocker.get("path") or "").strip():
        return False
    if harness.state.scratchpad.get("_task_complete"):
        return True
    facts = getattr(getattr(harness.state, "working_memory", None), "known_facts", None)
    if isinstance(facts, list):
        facts_text = " ".join(str(f) for f in facts).lower()
        if any(m.lower() in facts_text for m in _COMPLETION_FACT_MARKERS):
            return True
    return False


def extract_completion_message(harness: Any, hidden_tool_calls: list[Any]) -> str:
    for pending in hidden_tool_calls:
        if pending.tool_name == "task_complete":
            args = pending.args if isinstance(pending.args, dict) else {}
            msg = str(args.get("message") or "").strip()
            if msg:
                return msg
    last_text = str(getattr(harness.state, "_last_assistant_text", "") or "").strip()
    if not last_text:
        facts = getattr(getattr(harness.state, "working_memory", None), "known_facts", None)
        if isinstance(facts, list) and facts:
            last_text = str(facts[-1]).strip()
    return last_text[:500] if last_text else "Task completed (force-finalized after repeated task_complete blocks)."


def latest_verifier_allows_terminal_recovery(harness: Any) -> bool:
    verdict_fn = getattr(harness.state, "current_verifier_verdict", None)
    verdict = verdict_fn() if callable(verdict_fn) else getattr(harness.state, "last_verifier_verdict", None)
    if not isinstance(verdict, dict):
        return True
    return str(verdict.get("verdict", "") or "").strip().lower() in {"", "pass"}


def readonly_answer_can_complete(
    text: str,
    *,
    harness: Any,
    nudge_count: int,
) -> bool:
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    action_stalls = int(scratchpad.get("_action_stalls", 0)) if isinstance(scratchpad, dict) else 0
    if max(nudge_count, action_stalls) < 1:
        return False
    if isinstance(scratchpad, dict) and scratchpad.get("_unresolved_missing_input_file"):
        return False
    task = str(getattr(getattr(state, "run_brief", None), "original_task", "") or "").lower()
    current_goal = str(getattr(getattr(state, "working_memory", None), "current_goal", "") or "").lower()
    task_text = f"{task} {current_goal}"
    if any(marker in task_text for marker in ("patch", "edit", "modify", "fix ", "implement", "write file", "create file")):
        return False
    if not any(marker in task_text for marker in ("read", "list", "summarize", "inspect", "review", "improvement", "recommend")):
        return False
    lowered = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if any(marker in lowered for marker in ("let me ", "i'll ", "i will ", "i need to ", "next i ", "can inspect")):
        return False
    if len(lowered) < 80:
        return False
    return bool(
        re.search(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s+\S", text)
        or any(marker in lowered for marker in ("recommend", "improvement", "would make", "found", "the file"))
    )


def raw_terminal_json_completion_message(text: str) -> str:
    candidates = []
    for match in _FENCED_JSON_BLOCK_RE.finditer(text):
        candidate = match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            candidates.append(candidate)
    stripped = str(text or "").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        value = payload.get("task_complete")
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            message = str(value.get("message") or "").strip()
            if message:
                return message
    return ""


def terminal_prose_completion_message(
    assistant_text: str,
    *,
    harness: Any,
    nudge_count: int,
) -> str:
    text = str(assistant_text or "").strip()
    if not text:
        return ""
    raw_terminal_message = raw_terminal_json_completion_message(text)
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    has_strong_marker = any(marker in lowered for marker in _TERMINAL_PROSE_STRONG_MARKERS)
    readonly_completion = readonly_answer_can_complete(text, harness=harness, nudge_count=nudge_count)
    looks_like_completion = (
        bool(raw_terminal_message)
        or has_strong_marker
        or lowered.endswith("**task complete**")
        or lowered.endswith("task complete")
        or readonly_completion
    )
    if not looks_like_completion:
        return ""
    if not raw_terminal_message and not has_strong_marker and not readonly_completion and nudge_count < 1:
        return ""
    has_recent_tool_evidence = any(
        message.role == "tool"
        for message in getattr(harness.state, "recent_messages", [])[-12:]
    )
    has_working_memory_evidence = bool(getattr(getattr(harness.state, "working_memory", None), "known_facts", None))
    has_artifact_evidence = bool(getattr(harness.state, "artifacts", None))
    if not (has_recent_tool_evidence or has_working_memory_evidence or has_artifact_evidence):
        return ""
    if getattr(harness.state, "plan_execution_mode", False) and str(getattr(harness.state, "active_step_id", "") or ""):
        return ""
    session = getattr(harness.state, "write_session", None)
    if session is not None and str(getattr(session, "status", "") or "").strip().lower() != "complete":
        return ""
    acceptance_ready = getattr(harness.state, "acceptance_ready", None)
    if callable(acceptance_ready) and not acceptance_ready():
        return ""
    if not latest_verifier_allows_terminal_recovery(harness):
        return ""
    return (raw_terminal_message or text)[:4000]


def maybe_promote_terminal_prose_task_complete(
    graph_state: Any,
    harness: Any,
    *,
    nudge_count: int,
) -> bool:
    from .state import PendingToolCall
    message = terminal_prose_completion_message(
        graph_state.last_assistant_text,
        harness=harness,
        nudge_count=nudge_count,
    )
    if not message:
        return False
    raw_arguments = json.dumps({"message": message}, ensure_ascii=True)
    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="task_complete",
            args={"message": message},
            tool_call_id=f"synthetic-terminal-prose-{harness.state.step_count + 1}",
            raw_arguments=raw_arguments,
            source="system",
        )
    ]
    harness.state.scratchpad["_terminal_prose_task_complete_autopromoted"] = {
        "recovery_kind": "terminal_prose_task_complete",
        "message_preview": message[:500],
        "nudge_count": nudge_count,
    }
    harness._runlog(
        "terminal_prose_task_complete_autopromoted",
        "promoted terminal prose into task_complete tool call",
        recovery_kind="terminal_prose_task_complete",
        nudge_count=nudge_count,
        message_preview=message[:240],
    )
    return True


def maybe_promote_raw_terminal_json_task_complete(
    graph_state: Any,
    harness: Any,
) -> bool:
    from .state import PendingToolCall
    message = raw_terminal_json_completion_message(graph_state.last_assistant_text)
    if not message:
        return False
    raw_arguments = json.dumps({"message": message}, ensure_ascii=True)
    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="task_complete",
            args={"message": message},
            tool_call_id=f"synthetic-terminal-json-{harness.state.step_count + 1}",
            raw_arguments=raw_arguments,
            source="system",
        )
    ]
    harness.state.scratchpad["_terminal_json_task_complete_autopromoted"] = {
        "recovery_kind": "terminal_json_task_complete",
        "message_preview": message[:500],
    }
    harness._runlog(
        "terminal_json_task_complete_autopromoted",
        "promoted fenced terminal JSON into task_complete tool call",
        recovery_kind="terminal_json_task_complete",
        message_preview=message[:240],
    )
    return True
