from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..client import OpenAICompatClient
from .memory import assess_write_task_complexity
from ..guards import is_four_b_or_under_model_name
from ..interrupt_replies import (
    is_interrupt_affirmative_response,
    is_interrupt_response,
)
from ..models.events import UIEvent, UIEventType
from ..models.conversation import ConversationMessage
from .followup_signals import (
    assistant_message_proposes_concrete_implementation,
    is_affirmative_followup,
    recent_assistant_requested_action_confirmation,
)
from .task_classifier import (
    classify_task_mode,
    classify_runtime_intent,
    is_smalltalk,
    looks_like_complex_task,
    looks_like_readonly_chat_request,
    runtime_policy_for_intent,
)

if TYPE_CHECKING:
    from ..harness import Harness

logger = logging.getLogger("smallctl.harness.run_mode")
_ACTION_CONFIRMATION_PROMPTS = (
    "would you like me to",
    "do you want me to",
    "should i",
    "shall i",
    "want me to",
    "ready for me to",
)
_MODE_DECISION_ALIASES = {
    ("loop",): "loop",
    ("l",): "loop",
    ("chat",): "chat",
    ("c",): "chat",
    ("mode", "loop"): "loop",
    ("mode", "l"): "loop",
    ("loop", "mode"): "loop",
    ("mode", "chat"): "chat",
    ("mode", "c"): "chat",
    ("chat", "mode"): "chat",
}

def should_enable_complex_write_chat_draft(
    task: str,
    *,
    model_name: str | None = None,
    cwd: str | None = None,
) -> bool:
    """Return True when complex write tasks should take the chat drafting path."""
    text = str(task or "").strip()
    if not text:
        return False
    if not is_four_b_or_under_model_name(model_name):
        return False

    analysis = assess_write_task_complexity(text, cwd=cwd)
    return bool(analysis.get("force_chunk_mode") or analysis.get("force_chunk_mode_targets"))


def decide_run_mode_sync(
    task: str,
    *,
    model_name: str | None = None,
    cwd: str | None = None,
) -> str | None:
    """Sync heuristics for mode decision. Returns mode name or None to continue to model-based decision."""
    if should_enable_complex_write_chat_draft(task, model_name=model_name, cwd=cwd):
        return "chat"
    return None


def normalize_mode_decision(decision_text: str) -> str | None:
    normalized = str(decision_text or "").strip().lower()
    if not normalized:
        return None
    normalized = normalized.strip(" \t\r\n'\"`.,:;!?()[]{}<>")
    if not normalized:
        return None
    tokens = tuple(token for token in re.split(r"[^a-z]+", normalized) if token)
    if not tokens:
        return None
    return _MODE_DECISION_ALIASES.get(tokens)


def _is_affirmative_confirmation_reply(task: str) -> bool:
    return is_affirmative_followup(task)


def _recent_assistant_requested_action_confirmation(messages: list[ConversationMessage]) -> bool:
    return recent_assistant_requested_action_confirmation(messages, prompts=_ACTION_CONFIRMATION_PROMPTS)


def _has_single_confirmed_session_ssh_target(harness: Any) -> bool:
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    targets = scratchpad.get("_session_ssh_targets") if isinstance(scratchpad, dict) else None
    if not isinstance(targets, dict):
        return False
    confirmed = 0
    for value in targets.values():
        if isinstance(value, dict) and bool(value.get("confirmed")) and str(value.get("host") or "").strip():
            confirmed += 1
            if confirmed > 1:
                return False
    return confirmed == 1


def _recent_assistant_proposed_concrete_implementation(messages: list[ConversationMessage]) -> bool:
    for message in reversed(messages[-4:]):
        if getattr(message, "role", "") != "assistant":
            continue
        text = str(getattr(message, "content", "") or "").strip()
        if assistant_message_proposes_concrete_implementation(text):
            return True
    return False


def _has_plan_execution_approval_context(harness: Any) -> bool:
    state = getattr(harness, "state", None)
    pending_interrupt = getattr(state, "pending_interrupt", None)
    if isinstance(pending_interrupt, dict) and pending_interrupt.get("kind") == "plan_execute_approval":
        return True
    planner_interrupt = getattr(state, "planner_interrupt", None)
    if str(getattr(planner_interrupt, "kind", "") or "").strip() == "plan_execute_approval":
        return True
    plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
    status = str(getattr(plan, "status", "") or "").strip().lower()
    return status == "awaiting_approval"


def resolve_mode_task(harness: Any, task: str) -> tuple[str, str]:
    raw_task = str(task or "").strip()
    resolved_task = raw_task
    resolver = getattr(harness, "_resolve_followup_task", None)
    if callable(resolver):
        try:
            candidate = str(resolver(raw_task) or "").strip()
        except Exception:
            candidate = ""
        if candidate:
            resolved_task = candidate
    return raw_task, resolved_task


def is_contextual_affirmative_execution_continuation(
    harness: Any,
    *,
    raw_task: str,
    resolved_task: str,
) -> bool:
    pending_interrupt = getattr(getattr(harness, "state", None), "pending_interrupt", None)
    if is_interrupt_affirmative_response(pending_interrupt, raw_task):
        return True
    if not _is_affirmative_confirmation_reply(raw_task):
        return False

    recent_messages = list(getattr(getattr(harness, "state", None), "recent_messages", []) or [])
    has_recent_confirmation = _recent_assistant_requested_action_confirmation(recent_messages)
    if classify_task_mode(resolved_task) == "remote_execute":
        return True

    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    handoff = scratchpad.get("_last_task_handoff") if isinstance(scratchpad, dict) else None
    if not isinstance(handoff, dict):
        return False
    handoff_is_remote = False
    if str(handoff.get("task_mode") or "").strip() == "remote_execute":
        handoff_is_remote = True
    ssh_target = handoff.get("ssh_target")
    if isinstance(ssh_target, dict) and str(ssh_target.get("host") or "").strip():
        handoff_is_remote = True
    next_required_tool = handoff.get("next_required_tool")
    if isinstance(next_required_tool, dict) and str(next_required_tool.get("tool_name") or "").strip() == "ssh_exec":
        handoff_is_remote = True

    if handoff_is_remote and _has_plan_execution_approval_context(harness):
        return True
    if has_recent_confirmation:
        return handoff_is_remote
    if handoff_is_remote and _has_single_confirmed_session_ssh_target(harness):
        return _recent_assistant_proposed_concrete_implementation(recent_messages)
    return False


class ModeDecisionService:
    def __init__(self, harness: Harness):
        self.harness = harness

    async def _announce_mode_change(self, *, mode: str, reason: str) -> None:
        if mode != "planning":
            return
        await self.harness._emit(
            self.harness.event_handler,
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Planning mode enabled.",
                data={
                    "status_activity": "planning mode active",
                    "mode": mode,
                    "reason": reason,
                },
            ),
        )

    async def decide(self, task: str) -> str:
        raw_task, resolved_task = resolve_mode_task(self.harness, task)
        mode_task = resolved_task or raw_task
        scratchpad = getattr(getattr(self.harness, "state", None), "scratchpad", None)
        transaction = scratchpad.get("_task_transaction") if isinstance(scratchpad, dict) else None
        if isinstance(transaction, dict) and str(transaction.get("turn_type") or "").strip() == "CLARIFICATION":
            self.harness._runlog(
                "mode_decision",
                "selected run mode",
                mode="chat",
                raw="task_transaction_clarification",
                raw_task=raw_task,
                effective_task=mode_task,
            )
            return "chat"

        pending_interrupt = getattr(getattr(self.harness, "state", None), "pending_interrupt", None)
        if is_interrupt_response(pending_interrupt, raw_task):
            self.harness._runlog(
                "mode_decision",
                "selected run mode",
                mode="loop",
                raw="pending_interrupt_response",
                interrupt_kind=pending_interrupt.get("kind") if isinstance(pending_interrupt, dict) else "",
                raw_task=raw_task,
            )
            return "loop"

        plan_request = self._extract_planning_request(task)
        if plan_request is not None:
            output_path, output_format = plan_request
            self._set_planning_request(output_path=output_path, output_format=output_format)
            self.harness._runlog(
                "mode_decision",
                "selected run mode",
                mode="planning",
                raw="planning_intent",
                output_path=output_path,
                output_format=output_format,
            )
            await self._announce_mode_change(mode="planning", reason="planning_intent")
            return "planning"

        model_name = getattr(self.harness.client, "model", None)
        sync_mode = decide_run_mode_sync(
            mode_task,
            model_name=model_name,
            cwd=getattr(self.harness.state, "cwd", None),
        )
        if sync_mode is not None:
            self.harness._runlog(
                "mode_decision",
                "selected run mode",
                mode=sync_mode,
                raw="complex_write_sync_heuristic",
                raw_task=raw_task,
                effective_task=mode_task,
            )
            return sync_mode
            
        if self.harness.state.planning_mode_enabled and not (self.harness.state.active_plan and self.harness.state.active_plan.approved):
            self.harness._runlog(
                "mode_decision",
                "selected run mode",
                mode="planning",
                raw="planning_mode_enabled",
            )
            await self._announce_mode_change(mode="planning", reason="planning_mode_enabled")
            return "planning"

        if is_contextual_affirmative_execution_continuation(
            self.harness,
            raw_task=raw_task,
            resolved_task=mode_task,
        ):
            self.harness._runlog(
                "mode_decision",
                "selected run mode",
                mode="loop",
                raw="contextual_affirmative_execution_continuation",
                raw_task=raw_task,
                effective_task=mode_task,
            )
            return "loop"

        if is_smalltalk(mode_task):
            self.harness._runlog("mode_decision", "selected run mode", mode="chat", raw="smalltalk_heuristic")
            return "chat"

        pending_interrupt = getattr(self.harness.state, "pending_interrupt", None)
        runtime_intent = classify_runtime_intent(
            mode_task,
            recent_messages=self.harness.state.recent_messages,
            pending_interrupt=pending_interrupt,
        )
        runtime_policy = runtime_policy_for_intent(runtime_intent)
        if runtime_policy.route_mode is not None:
            # Auto-escalate complex execution tasks to planning mode
            if runtime_policy.route_mode == "loop" and looks_like_complex_task(mode_task):
                self.harness._runlog(
                    "mode_decision",
                    "selected run mode",
                    mode="planning",
                    raw="complex_task_auto_escalation",
                    intent=runtime_intent.label,
                    task_mode=runtime_intent.task_mode,
                    raw_task=raw_task,
                    effective_task=mode_task,
                )
                await self._announce_mode_change(mode="planning", reason="complex_task_auto_escalation")
                return "planning"
            self.harness._runlog(
                "mode_decision",
                "selected run mode",
                mode=runtime_policy.route_mode,
                raw="runtime_intent_policy",
                intent=runtime_intent.label,
                task_mode=runtime_intent.task_mode,
                raw_task=raw_task,
                effective_task=mode_task,
            )
            return runtime_policy.route_mode
            
        prompt = (
            "Decide whether the user request requires tool usage in a coding harness. "
            "Reply with exactly one word: chat or loop."
        )
        messages = [
            ConversationMessage(role="system", content=prompt).to_dict(),
            ConversationMessage(role="user", content=mode_task).to_dict(),
        ]
        chunks: list[dict[str, Any]] = []
        try:
            async for event in self.harness.client.stream_chat(messages=messages, tools=[]):
                chunks.append(event)
        except Exception:
            self.harness._runlog("mode_decision_fallback", "mode decision failed, using loop")
            return "loop"
            
        decision_result = OpenAICompatClient.collect_stream(
            chunks,
            reasoning_mode="off",
            thinking_start_tag=self.harness.thinking_start_tag,
            thinking_end_tag=self.harness.thinking_end_tag,
        )
        decision = decision_result.assistant_text.strip().lower()
        if not decision:
            decision = decision_result.thinking_text.strip().lower()
        normalized_decision = normalize_mode_decision(decision)
        mode = normalized_decision or "loop"
        self.harness._runlog(
            "mode_decision",
            "selected run mode",
            mode=mode,
            raw=decision,
            normalized=normalized_decision,
            raw_task=raw_task,
            effective_task=mode_task,
        )
        return mode

    def _set_planning_request(self, *, output_path: str | None = None, output_format: str | None = None) -> None:
        self.harness.state.planning_mode_enabled = True
        self.harness.state.planner_requested_output_path = str(output_path or "").strip()
        self.harness.state.planner_requested_output_format = str(output_format or "").strip().lower()
        self.harness.state.planner_resume_target_mode = "loop"
        self.harness.state.touch()
        self.harness.state.sync_plan_mirror()

    def _extract_planning_request(self, task: str) -> tuple[str | None, str | None] | None:
        lowered = (task or "").lower()
        if "plan" not in lowered:
            return None
        output_path: str | None = None
        output_format: str | None = None

        path_match = re.search(r"([^\s]+?\.(?:md|txt|text))\b", task, flags=re.IGNORECASE)
        if path_match:
            output_path = path_match.group(1)
            suffix = Path(output_path).suffix.lower()
            if suffix == ".md":
                output_format = "markdown"
            elif suffix in {".txt", ".text"}:
                output_format = "text"
        if "plan.md" in lowered and output_path is None:
            output_format = "markdown"
        if any(
            phrase in lowered
            for phrase in (
                "make a plan",
                "make a short plan",
                "create a plan",
                "create a short plan",
                "create a brief plan",
                "plan this",
                "plan this out",
                "make a plan first",
                "plan out",
                "before doing anything, create a short plan",
                "before doing anything, create a plan",
                "before doing anything, plan",
            )
        ):
            return output_path, output_format
        if output_path:
            return output_path, output_format
        return None

    def looks_like_readonly_chat_request(self, task: str) -> bool:
        return looks_like_readonly_chat_request(task)

    def chat_mode_requires_tools(self, task: str) -> bool:
        intent = classify_runtime_intent(
            task,
            recent_messages=self.harness.state.recent_messages,
        )
        session = getattr(self.harness.state, "write_session", None)
        if session is not None:
            status = str(getattr(session, "status", "") or "").strip().lower()
            if status != "complete":
                return True
        model_name = getattr(self.harness.client, "model", None)
        if should_enable_complex_write_chat_draft(
            task,
            model_name=model_name,
            cwd=getattr(self.harness.state, "cwd", None),
        ):
            return True
        return runtime_policy_for_intent(intent).chat_requires_tools
