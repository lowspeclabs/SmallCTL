from __future__ import annotations

from typing import Any

from ..models.tool_result import ToolEnvelope
from .task_classifier import (
    is_smalltalk,
    looks_like_action_request,
    looks_like_execution_followup,
    looks_like_readonly_chat_request,
    looks_like_shell_request,
    needs_contextual_loop_escalation,
    needs_loop_for_content_lookup,
    needs_memory_persistence,
    recent_assistant_proposed_command,
    recent_assistant_referenced_tool_name,
)
from .task_intent import derive_task_contract as _derive_task_contract_helper
from .task_intent import extract_intent_state as _extract_intent_state_helper
from .task_intent import infer_entity_tags as _infer_entity_tags_helper
from .task_intent import infer_environment_tags as _infer_environment_tags_helper
from .task_intent import infer_requested_tool_name as _infer_requested_tool_name_helper
from .task_intent import next_action_for_task as _next_action_for_task_helper
from .tool_dispatch import (
    attempt_tool_sanitization as _attempt_tool_sanitization_helper,
    chat_mode_requires_tools as _chat_mode_requires_tools_helper,
    chat_mode_tools as _chat_mode_tools_helper,
    dispatch_tool_call as _dispatch_tool_call_helper,
    maybe_reuse_file_read as _maybe_reuse_file_read_helper,
)
from .tool_message_compaction import compact_oversized_tool_messages as _compact_oversized_tool_messages_helper


def _extract_intent_state(self: Any, task: str) -> tuple[str, list[str], list[str]]:
    return _extract_intent_state_helper(self, task)


def _infer_environment_tags(self: Any) -> list[str]:
    return _infer_environment_tags_helper(self)


def _infer_entity_tags(self: Any, task: str) -> list[str]:
    return _infer_entity_tags_helper(task)


def _infer_requested_tool_name(self: Any, task: str) -> str:
    return _infer_requested_tool_name_helper(self, task)


def _next_action_for_task(self: Any, task: str) -> str:
    return _next_action_for_task_helper(self, task)


def _derive_task_contract(self: Any, task: str) -> str:
    return _derive_task_contract_helper(task)


def _chat_mode_tools(self: Any) -> list[dict[str, Any]]:
    return _chat_mode_tools_helper(self)


async def _dispatch_tool_call(self: Any, tool_name: str, args: dict[str, Any]) -> ToolEnvelope:
    return await _dispatch_tool_call_helper(self, tool_name, args)


def _attempt_tool_sanitization(self: Any, tool_name: str) -> str | None:
    return _attempt_tool_sanitization_helper(self, tool_name)


def _maybe_reuse_file_read(self: Any, *, tool_name: str, args: dict[str, Any]) -> ToolEnvelope | None:
    return _maybe_reuse_file_read_helper(self, tool_name=tool_name, args=args)


def _is_smalltalk(self: Any, task: str) -> bool:
    return is_smalltalk(task)


def _needs_loop_for_content_lookup(self: Any, task: str) -> bool:
    return needs_loop_for_content_lookup(task)


def _needs_contextual_loop_escalation(self: Any, task: str) -> bool:
    return needs_contextual_loop_escalation(self.state.recent_messages, task)


def _compact_oversized_tool_messages(self: Any, *, soft_limit: int) -> bool:
    return _compact_oversized_tool_messages_helper(self, soft_limit=soft_limit)


def _chat_mode_requires_tools(self: Any, task: str) -> bool:
    return _chat_mode_requires_tools_helper(self, task)


def _looks_like_action_request(self: Any, task: str) -> bool:
    return looks_like_action_request(task)


def _needs_memory_persistence(self: Any, task: str) -> bool:
    return needs_memory_persistence(task)


def _looks_like_shell_request(self: Any, task: str) -> bool:
    return looks_like_shell_request(task)


def _looks_like_readonly_chat_request(self: Any, task: str) -> bool:
    return looks_like_readonly_chat_request(task)


def _looks_like_execution_followup(self: Any, text: str) -> bool:
    return looks_like_execution_followup(text)


def _recent_assistant_proposed_command(self: Any) -> bool:
    return recent_assistant_proposed_command(self.state.recent_messages)


def _recent_assistant_referenced_tool_name(self: Any, tool_name: str) -> bool:
    return recent_assistant_referenced_tool_name(self.state.recent_messages, tool_name)


def bind_intent_facade(cls: type[Any]) -> None:
    cls._extract_intent_state = _extract_intent_state
    cls._infer_environment_tags = _infer_environment_tags
    cls._infer_entity_tags = _infer_entity_tags
    cls._infer_requested_tool_name = _infer_requested_tool_name
    cls._next_action_for_task = _next_action_for_task
    cls._derive_task_contract = _derive_task_contract
    cls._chat_mode_tools = _chat_mode_tools
    cls._dispatch_tool_call = _dispatch_tool_call
    cls._attempt_tool_sanitization = _attempt_tool_sanitization
    cls._maybe_reuse_file_read = _maybe_reuse_file_read
    cls._is_smalltalk = _is_smalltalk
    cls._needs_loop_for_content_lookup = _needs_loop_for_content_lookup
    cls._needs_contextual_loop_escalation = _needs_contextual_loop_escalation
    cls._compact_oversized_tool_messages = _compact_oversized_tool_messages
    cls._chat_mode_requires_tools = _chat_mode_requires_tools
    cls._looks_like_action_request = _looks_like_action_request
    cls._needs_memory_persistence = _needs_memory_persistence
    cls._looks_like_shell_request = _looks_like_shell_request
    cls._looks_like_readonly_chat_request = _looks_like_readonly_chat_request
    cls._looks_like_execution_followup = _looks_like_execution_followup
    cls._recent_assistant_proposed_command = _recent_assistant_proposed_command
    cls._recent_assistant_referenced_tool_name = _recent_assistant_referenced_tool_name
