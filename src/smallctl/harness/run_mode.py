from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..client import OpenAICompatClient
from ..guards import is_four_b_or_under_model_name
from ..prompt_model_classifiers import is_gemma_4_non_exact_small_model_name
from .memory import assess_write_task_complexity
from ..interrupt_replies import (
    is_interrupt_affirmative_response,
    is_interrupt_response,
    is_plan_approval_reply,
)
from ..models.events import UIEvent, UIEventType
from ..models.conversation import ConversationMessage
from .followup_signals import (
    assistant_message_proposes_concrete_implementation,
    is_affirmative_followup,
    normalize_followup_text,
    recent_assistant_requested_action_confirmation,
)
from .task_classifier import (
    classify_task_mode,
    classify_runtime_intent,
    is_smalltalk,
    looks_like_complex_task,
    looks_like_numbered_implementation_followup,
    looks_like_readonly_chat_request,
    looks_like_shell_request,
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


def _looks_like_implementation_followup_after_plan(raw_task: str, harness: Any) -> bool:
    """Detect follow-ups that implement a previously proposed improvement.

    When the assistant just produced a numbered list of improvements and the
    user replies with a phrase like "now implement the testing fixes",
    "apply fix 1", or "do the test suite improvement", we should treat it as
    an execution task rather than a chat request.
    """
    text = str(raw_task or "").strip().lower()
    if not text:
        return False
    if looks_like_numbered_implementation_followup(raw_task):
        return True
    implementation_verbs = (
        "implement", "apply", "do", "fix", "patch", "build", "write",
        "create", "add", "proceed with", "start on", "work on",
    )
    improvement_subjects = (
        "fix", "fixes", "improvement", "improvements", "change", "changes",
        "proposal", "proposals", "option", "options", "item", "items",
        "step", "steps", "issue", "issues", "test suite", "tests", "testing",
    )
    has_verb = any(verb in text for verb in implementation_verbs)
    has_subject = any(subject in text for subject in improvement_subjects)
    if not (has_verb and has_subject):
        return False
    # Require a recently completed task that proposed improvements/fixes.
    state = getattr(harness, "state", None)
    if state is None:
        return False
    for message in reversed(getattr(state, "recent_messages", []) or []):
        if message.role == "assistant" and message.content:
            content_lower = message.content.lower()
            if (
                "improvement" in content_lower
                or "improvemnts" in content_lower
                or "proposed" in content_lower
                or "fixes" in content_lower
            ) and any(marker in content_lower for marker in ("1)", "1.", "#1", "first", "test suite")):
                return True
    return False


def _resolve_numbered_proposal_reference(raw_task: str, harness: Any) -> str | None:
    """If the user refers to proposal/fix #N, return the matching proposal text.

    Expands terse or typo-prone follow-ups like "apply fi #3" into the actual
    proposal text from the assistant's previous message, so that downstream
    intent classification and tool exposure see a concrete implementation task.
    """
    text = str(raw_task or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if not looks_like_numbered_implementation_followup(text):
        return None
    match = re.search(r"#(\d+)", lowered)
    if not match:
        return None
    index = int(match.group(1))
    state = getattr(harness, "state", None)
    if state is None:
        return None
    for message in reversed(getattr(state, "recent_messages", []) or []):
        if message.role != "assistant" or not message.content:
            continue
        content = message.content
        # Find numbered list items in the assistant message.
        # Supports "1. **Title**", "1) Title", "**1. Title**", etc.
        items = re.findall(
            r"(?:^|\n)\s*(?:\*\*)?\s*(\d+)[.\)]\s*\*?\*?\s*(.+?)(?=\n\s*(?:\*\*)?\s*\d+[.\)]|\Z)",
            content,
            re.DOTALL,
        )
        if not items:
            continue
        by_number = {num: body.strip() for num, body in items}
        body = by_number.get(str(index))
        if body:
            # Collapse body to a single line for the effective task.
            body = " ".join(body.split())
            return f"Patch ./temp/vikunja-9b.py to implement proposal #{index}: {body}"
    return None


def has_active_remote_handoff(harness: Any) -> bool:
    state = getattr(harness, "state", None)
    if str(getattr(state, "task_mode", "") or "").strip() == "remote_execute":
        return True
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    handoff = scratchpad.get("_last_task_handoff")
    if isinstance(handoff, dict):
        if str(handoff.get("task_mode") or "").strip() == "remote_execute":
            return True
        ssh_target = handoff.get("ssh_target")
        if isinstance(ssh_target, dict) and str(ssh_target.get("host") or "").strip():
            return True
        ssh_targets = handoff.get("ssh_targets")
        if isinstance(ssh_targets, list) and any(
            isinstance(item, dict) and str(item.get("host") or "").strip()
            for item in ssh_targets
        ):
            return True
        next_required_tool = handoff.get("next_required_tool")
        if isinstance(next_required_tool, dict) and str(next_required_tool.get("tool_name") or "").strip() in {
            "ssh_exec",
            "ssh_file_read",
            "ssh_file_write",
            "ssh_file_patch",
            "ssh_file_replace_between",
        }:
            return True
    targets = scratchpad.get("_session_ssh_targets")
    return isinstance(targets, dict) and any(
        isinstance(value, dict) and str(value.get("host") or "").strip()
        for value in targets.values()
    )


def ensure_remote_tool_profile(harness: Any) -> None:
    state = getattr(harness, "state", None)
    raw_profiles = getattr(state, "active_tool_profiles", [])
    profiles = list(raw_profiles) if isinstance(raw_profiles, (list, tuple, set)) else []
    if "network" in profiles:
        return
    from ..harness.task_classifier import task_is_local_coding_target
    run_brief = getattr(state, "run_brief", None)
    task_text = str(getattr(run_brief, "original_task", "") or "") if run_brief else ""
    if task_is_local_coding_target(task_text):
        return
    profiles.append("network")
    try:
        state.active_tool_profiles = profiles
    except Exception:
        return


def _looks_like_execution_approval_reply(task: str) -> bool:
    if is_plan_approval_reply(task) or is_affirmative_followup(task):
        return True
    tokens = set(normalize_followup_text(task).split())
    if not tokens:
        return False
    approval_tokens = {"approve", "approved", "proceed", "continue", "execute", "run"}
    action_tokens = {"fix", "repair", "apply", "implement", "update", "change", "install", "deploy", "patch"}
    return bool(tokens & approval_tokens) and bool(tokens & action_tokens)


def _recent_assistant_proposed_concrete_implementation(messages: list[ConversationMessage]) -> bool:
    for message in reversed(messages[-4:]):
        if getattr(message, "role", "") != "assistant":
            continue
        text = str(getattr(message, "content", "") or "").strip()
        if assistant_message_proposes_concrete_implementation(text):
            return True
    return False


def _approved_plan_matches_plan_interrupt(state: Any, interrupt: Any) -> bool:
    if state is None:
        return False
    if isinstance(interrupt, dict):
        kind = str(interrupt.get("kind") or "").strip()
        plan_id = str(interrupt.get("plan_id") or "").strip()
    else:
        kind = str(getattr(interrupt, "kind", "") or "").strip()
        plan_id = str(getattr(interrupt, "plan_id", "") or "").strip()
    if kind != "plan_execute_approval":
        return False
    for plan in (getattr(state, "active_plan", None), getattr(state, "draft_plan", None)):
        if plan is None or not bool(getattr(plan, "approved", False)):
            continue
        if not plan_id or str(getattr(plan, "plan_id", "") or "").strip() == plan_id:
            return True
    return False


def _has_plan_execution_approval_context(harness: Any) -> bool:
    state = getattr(harness, "state", None)
    pending_interrupt = getattr(state, "pending_interrupt", None)
    if isinstance(pending_interrupt, dict) and pending_interrupt.get("kind") == "plan_execute_approval":
        return not _approved_plan_matches_plan_interrupt(state, pending_interrupt)
    planner_interrupt = getattr(state, "planner_interrupt", None)
    if str(getattr(planner_interrupt, "kind", "") or "").strip() == "plan_execute_approval":
        return not _approved_plan_matches_plan_interrupt(state, planner_interrupt)
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
    if not _looks_like_execution_approval_reply(raw_task):
        return False

    recent_messages = list(getattr(getattr(harness, "state", None), "recent_messages", []) or [])
    has_recent_confirmation = _recent_assistant_requested_action_confirmation(recent_messages)
    if classify_task_mode(resolved_task) == "remote_execute":
        ensure_remote_tool_profile(harness)
        return True

    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    handoff = scratchpad.get("_last_task_handoff") if isinstance(scratchpad, dict) else None
    if not isinstance(handoff, dict):
        if has_active_remote_handoff(harness):
            ensure_remote_tool_profile(harness)
            return True
        # Local task continuation: if the assistant recently proposed a
        # concrete implementation or asked for confirmation, and the user
        # is replying with an approval, keep the task in loop mode so the
        # model retains filesystem tool access.
        if has_recent_confirmation or _recent_assistant_proposed_concrete_implementation(recent_messages):
            return True
        return False
    handoff_is_remote = has_active_remote_handoff(harness)

    if handoff_is_remote and _has_plan_execution_approval_context(harness):
        ensure_remote_tool_profile(harness)
        return True
    if has_recent_confirmation:
        if handoff_is_remote:
            ensure_remote_tool_profile(harness)
        return handoff_is_remote
    if handoff_is_remote and _has_single_confirmed_session_ssh_target(harness):
        ensure_remote_tool_profile(harness)
        return _recent_assistant_proposed_concrete_implementation(recent_messages)
    if handoff_is_remote:
        ensure_remote_tool_profile(harness)
        return True
    return False


class ModeDecisionService:
    def __init__(self, harness: Harness):
        self.harness = harness

    def _log_mode_decision(self, *, mode: str, raw: str, model_decision_raw: str | None = None, confidence: str = "n/a", **extra: Any) -> None:
        task_preview = str(
            getattr(getattr(self.harness.state, "run_brief", None), "original_task", "")
            or getattr(self.harness, "_current_user_task", lambda: "")()
            or ""
        )[:200]
        self.harness._runlog(
            "mode_decision",
            "selected run mode",
            level="debug",
            subsystem="graph",
            selected_mode=mode,
            heuristic_matched=raw,
            task_preview=task_preview,
            model_decision_raw=model_decision_raw,
            confidence=confidence,
            **extra,
        )

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
        if isinstance(scratchpad, dict) and scratchpad.pop("_fama_force_tool_plan_next_turn", False):
            self._log_mode_decision(mode="tool_plan", raw="fama_hard_route", raw_task=raw_task, effective_task=mode_task)
            return "tool_plan"
        transaction = scratchpad.get("_task_transaction") if isinstance(scratchpad, dict) else None
        if isinstance(transaction, dict) and str(transaction.get("turn_type") or "").strip() == "CLARIFICATION":
            self._log_mode_decision(mode="chat", raw="task_transaction_clarification", raw_task=raw_task, effective_task=mode_task)
            return "chat"

        pending_interrupt = getattr(getattr(self.harness, "state", None), "pending_interrupt", None)
        if is_interrupt_response(pending_interrupt, raw_task):
            self._log_mode_decision(
                mode="loop",
                raw="pending_interrupt_response",
                interrupt_kind=pending_interrupt.get("kind") if isinstance(pending_interrupt, dict) else "",
                raw_task=raw_task,
            )
            return "loop"

        # Defensive fallback: if the pending interrupt was lost or not recognized,
        # but the user is clearly replying to a plan approval prompt, force loop mode
        # so the runtime can resume the planning graph.
        if is_plan_approval_reply(raw_task) and _has_plan_execution_approval_context(self.harness):
            self._log_mode_decision(mode="loop", raw="plan_approval_fallback", raw_task=raw_task)
            return "loop"

        plan_request = self._extract_planning_request(task)
        if plan_request is not None:
            output_path, output_format = plan_request
            self._set_planning_request(output_path=output_path, output_format=output_format)
            self._log_mode_decision(
                mode="planning",
                raw="planning_intent",
                output_path=output_path,
                output_format=output_format,
            )
            await self._announce_mode_change(mode="planning", reason="planning_intent")
            return "planning"

        model_name = getattr(self.harness.client, "model", None)

        # Continuation approvals (e.g. "apply the fixes", "proceed", "yes")
        # must be resolved before the complex-write heuristic. Otherwise a
        # follow-up that inherits prior task context can be misclassified as a
        # brand-new complex write and forced into chat mode, where mutating
        # tools are blocked and the run ends with chat_action_blocked.
        if is_contextual_affirmative_execution_continuation(
            self.harness,
            raw_task=raw_task,
            resolved_task=mode_task,
        ):
            self._log_mode_decision(
                mode="loop",
                raw="contextual_affirmative_execution_continuation",
                raw_task=raw_task,
                effective_task=mode_task,
            )
            return "loop"

        # Fix 1 (RCA 8b79ca76): Only apply complex_write_sync_heuristic on the
        # user's raw input, not on resolved/inherited effective_task which may
        # carry forward prior write context (e.g. "Continue current task: ...
        # User follow-up: ...") and falsely trigger chunk mode on follow-up
        # turns that are actually execution or verification requests.
        sync_mode = decide_run_mode_sync(
            raw_task,
            model_name=model_name,
            cwd=getattr(self.harness.state, "cwd", None),
        )
        if sync_mode is not None:
            self._log_mode_decision(
                mode=sync_mode,
                raw="complex_write_sync_heuristic",
                raw_task=raw_task,
                effective_task=mode_task,
            )
            return sync_mode

        if self.harness.state.planning_mode_enabled and not (self.harness.state.active_plan and self.harness.state.active_plan.approved):
            self._log_mode_decision(mode="planning", raw="planning_mode_enabled")
            await self._announce_mode_change(mode="planning", reason="planning_mode_enabled")
            return "planning"

        if is_smalltalk(mode_task):
            self._log_mode_decision(mode="chat", raw="smalltalk_heuristic")
            return "chat"

        # Fix for RCA 8ec35471: follow-ups that implement a previously proposed
        # improvement must not fall through to the chat-prone LLM classifier.
        expanded = _resolve_numbered_proposal_reference(raw_task, self.harness)
        if expanded and not mode_task:
            mode_task = expanded
        if _looks_like_implementation_followup_after_plan(raw_task, self.harness):
            self._log_mode_decision(
                mode="loop",
                raw="implementation_followup_after_plan",
                raw_task=raw_task,
                effective_task=mode_task,
            )
            return "loop"

        pending_interrupt = getattr(self.harness.state, "pending_interrupt", None)
        runtime_intent = classify_runtime_intent(
            mode_task,
            recent_messages=self.harness.state.recent_messages,
            pending_interrupt=pending_interrupt,
        )
        runtime_policy = runtime_policy_for_intent(runtime_intent)
        if runtime_policy.route_mode is not None:
            # Auto-escalate complex execution tasks to planning mode.
            # Guard A: Check complexity against the user's raw request, not
            #   the inherited effective_task (which concatenates prior task
            #   context and inflates verb counts, causing false positives).
            # Guard B: Never escalate direct execution/shell requests — even
            #   if they have enough verb diversity to trigger the complexity
            #   heuristic, users explicitly asking to "run/launch/debug"
            #   want loop mode, not planning.
            _raw_is_complex = looks_like_complex_task(raw_task)
            _raw_is_direct_execution = (
                looks_like_shell_request(raw_task)
                and runtime_intent.label == "execute"
            )
            _small_model = is_four_b_or_under_model_name(model_name)
            _gemma_4_planning_risk = is_gemma_4_non_exact_small_model_name(model_name)
            if runtime_policy.route_mode == "loop" and _raw_is_complex and not _raw_is_direct_execution:
                if _small_model or _gemma_4_planning_risk:
                    raw_reason = (
                        "complex_task_gemma_4_skip_planning"
                        if _gemma_4_planning_risk
                        else "complex_task_small_model_skip_planning"
                    )
                    self._log_mode_decision(
                        mode="loop",
                        raw=raw_reason,
                        intent=runtime_intent.label,
                        task_mode=runtime_intent.task_mode,
                        raw_task=raw_task,
                        effective_task=mode_task,
                    )
                    return "loop"
                self._log_mode_decision(
                    mode="planning",
                    raw="complex_task_auto_escalation",
                    intent=runtime_intent.label,
                    task_mode=runtime_intent.task_mode,
                    raw_task=raw_task,
                    effective_task=mode_task,
                )
                await self._announce_mode_change(mode="planning", reason="complex_task_auto_escalation")
                return "planning"
            self._log_mode_decision(
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
            self._log_mode_decision(mode="loop", raw="model_decision_fallback")
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
        self._log_mode_decision(
            mode=mode,
            raw="model_decision",
            model_decision_raw=decision,
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
