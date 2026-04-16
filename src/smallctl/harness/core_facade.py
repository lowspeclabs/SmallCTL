from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..guards import is_small_model_name
from ..logging_utils import log_kv
from ..models.events import UIEvent
from ..models.tool_result import ToolEnvelope
from ..state import (
    LOOP_STATE_SCHEMA_VERSION,
    ExperienceMemory,
    align_memory_entries,
    clip_string_list,
    clip_text_value,
    json_safe_value,
)
from ..normalization import dedupe_keep_tail
from ..tools import build_registry
from ..tools.profiles import classify_tool_profiles
from .task_intent import completion_next_action, extract_intent_state, next_action_for_task
from .tool_message_compaction import trim_recent_messages_window


async def _emit(
    self: Any,
    handler: Callable[[UIEvent], Awaitable[None] | None] | None,
    event: UIEvent,
) -> None:
    if handler is None:
        return
    maybe = handler(event)
    if maybe is not None and hasattr(maybe, "__await__"):
        await maybe


def _finalize(self: Any, result: dict[str, Any]) -> dict[str, Any]:
    status = str((result or {}).get("status") or "").strip().lower()
    task_summary = None
    if status not in {"needs_human", "plan_ready", "plan_approved"}:
        terminal_event = "task_interrupted" if status == "cancelled" else ""
        summary_status = "interrupted" if status == "cancelled" else status
        task_summary = self._finalize_task_scope(
            terminal_event=terminal_event,
            status=summary_status or "stopped",
            reason=str((result or {}).get("reason") or ""),
            result=result,
        )
        self._pending_task_shutdown_reason = ""
    summary_path = str((task_summary or {}).get("summary_path") or "").strip()
    task_id = str((task_summary or {}).get("task_id") or "").strip()
    self._runlog(
        "task_finalize",
        "task finished",
        result=result,
        task_id=task_id,
        task_summary_path=summary_path,
    )
    self._record_terminal_experience(result)
    self._rewrite_active_plan_export()
    if self.checkpoint_on_exit:
        self._persist_checkpoint(result)

    result["step_count"] = self.state.step_count
    result["inactive_steps"] = self.state.inactive_steps
    result["token_usage"] = self.state.token_usage

    if getattr(self, "run_logger", None) and hasattr(self.run_logger, "run_dir"):
        try:
            summary_payload = {
                "final_task_status": result.get("status", "unknown"),
                "total_tool_calls": self.state.step_count,
                "guard_trips": sum(1 for e in (getattr(self.state, "recent_errors", []) or []) if "Guard tripped" in str(e)),
                "postmortem_summary": result.get("reason") or "No reason provided",
            }
            summary_path = self.run_logger.run_dir / "task_summary.json"
            summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    self._cancel_requested = False
    self._active_dispatch_task = None
    return result


def _rewrite_active_plan_export(self: Any) -> None:
    plan = self.state.active_plan or self.state.draft_plan
    if plan is None or not plan.requested_output_path:
        return
    try:
        from ..plans import write_plan_file

        write_plan_file(plan, plan.requested_output_path, format=plan.requested_output_format)
    except Exception as exc:
        self.log.warning("failed to rewrite active plan export: %s", exc)


def _create_child_harness(
    self: Any,
    *,
    request: Any,
    harness_factory: Callable[..., Any] | None = None,
    artifact_start_index: int | None = None,
) -> Any:
    return self.subtasks.create_child_harness(
        request=request,
        harness_factory=harness_factory,
        artifact_start_index=artifact_start_index,
    )


def _build_subtask_result(
    self: Any,
    *,
    child: Any,
    request: Any,
    result: dict[str, Any],
) -> Any:
    return self.subtasks.build_subtask_result(child=child, request=request, result=result)


def _persist_checkpoint(self: Any, result: dict[str, Any]) -> None:
    path = (
        Path(self.checkpoint_path).resolve()
        if self.checkpoint_path
        else Path(self.state.cwd).resolve() / ".smallctl-checkpoint.json"
    )
    payload = {
        "checkpoint_schema_version": 1,
        "loop_state_schema_version": LOOP_STATE_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "result": json_safe_value(result),
        "state": self.state.to_dict(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log_kv(self.log, logging.INFO, "harness_checkpoint_saved", path=str(path))
    except Exception:
        self.log.exception("failed to persist checkpoint")


def _failure(
    message: str,
    *,
    error_type: str = "runtime",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "reason": message,
        "error": {
            "type": error_type,
            "message": message,
            "details": details or {},
        },
    }


def _runlog(self: Any, event: str, message: str, **data: Any) -> None:
    if self.run_logger:
        self.run_logger.log("harness", event, message, **data)
        if event.startswith("model_"):
            self.run_logger.log("model_output", event, message, **data)


def _stream_print(text: str) -> None:
    try:
        print(text, end="", flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe, end="", flush=True)


async def _rebuild_messages_after_context_overflow(
    self: Any,
    *,
    n_ctx: int,
    n_keep: int | None = None,
    error_message: str = "",
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> list[dict[str, Any]] | None:
    new_limit = self._apply_server_context_limit(
        n_ctx,
        source="stream_context_overflow",
        observed_n_keep=n_keep,
    )
    system_prompt = str(self.state.scratchpad.get("_last_system_prompt") or "")
    if not system_prompt:
        return None
    self._runlog(
        "context_limit_rebuild",
        "shrinking prompt budget after upstream context overflow",
        n_ctx=n_ctx,
        n_keep=n_keep,
        error=error_message,
        max_prompt_tokens=new_limit,
    )
    return await self._build_prompt_messages(system_prompt, event_handler=event_handler)


async def _build_prompt_messages(
    self: Any,
    system_prompt: str,
    *,
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> list[dict[str, Any]]:
    return await self.prompt_builder.build_messages(system_prompt, event_handler=event_handler)


async def _maybe_compact_context(
    self: Any,
    query: str,
    system_prompt: str,
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> None:
    await self.compaction.maybe_compact_context(
        query=query,
        system_prompt=system_prompt,
        event_handler=event_handler,
    )


def _update_working_memory(self: Any) -> None:
    self.memory.update_working_memory(self.context_policy.recent_message_limit)


def _refresh_active_intent(self: Any) -> None:
    self.memory._refresh_active_intent()


def _completion_next_action(self: Any) -> str:
    return "Decide whether the current evidence is sufficient; call task_complete when it is."


def _is_small_model_name(self: Any, model_name: str | None) -> bool:
    return is_small_model_name(model_name)


def _record_experience(
    self: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    evidence_refs: list[str] | None = None,
    notes: str = "",
    source: str = "observed",
) -> ExperienceMemory:
    return self.memory.record_experience(
        tool_name=tool_name,
        result=result,
        evidence_refs=evidence_refs,
        notes=notes,
        source=source,
    )


def _normalize_failure_mode(self: Any, error: Any, *, tool_name: str, success: bool) -> str:
    return self.memory._normalize_failure_mode(error, tool_name=tool_name, success=success)


def _reinforce_retrieved_experiences(self: Any, *, tool_name: str, success: bool) -> None:
    self.memory._reinforce_retrieved_experiences(tool_name=tool_name, success=success)


def _record_terminal_experience(self: Any, result: dict[str, Any]) -> None:
    self.memory.record_terminal_experience(result)


def _argument_fingerprint(self: Any, arguments: Any) -> str:
    return self.memory._argument_fingerprint(arguments)


def _activate_tool_profiles(self: Any, task: str) -> None:
    if self._configured_tool_profiles:
        profiles = set(self._configured_tool_profiles)
    else:
        profiles = classify_tool_profiles(task)
    self.state.active_tool_profiles = sorted(profiles)
    self.state.scratchpad["_last_task_text"] = task
    self._runlog(
        "tool_profiles",
        "selected tool profiles",
        task=task,
        profiles=self.state.active_tool_profiles,
        source="config" if self._configured_tool_profiles else "dynamic",
    )


def bind_core_facade(cls: type[Any]) -> None:
    cls._emit = _emit
    cls._finalize = _finalize
    cls._rewrite_active_plan_export = _rewrite_active_plan_export
    cls._create_child_harness = _create_child_harness
    cls._build_subtask_result = _build_subtask_result
    cls._persist_checkpoint = _persist_checkpoint
    cls._failure = staticmethod(_failure)
    cls._runlog = _runlog
    cls._stream_print = staticmethod(_stream_print)
    cls._rebuild_messages_after_context_overflow = _rebuild_messages_after_context_overflow
    cls._build_prompt_messages = _build_prompt_messages
    cls._maybe_compact_context = _maybe_compact_context
    cls._update_working_memory = _update_working_memory
    cls._refresh_active_intent = _refresh_active_intent
    cls._completion_next_action = _completion_next_action
    cls._is_small_model_name = _is_small_model_name
    cls._record_experience = _record_experience
    cls._normalize_failure_mode = _normalize_failure_mode
    cls._reinforce_retrieved_experiences = _reinforce_retrieved_experiences
    cls._record_terminal_experience = _record_terminal_experience
    cls._argument_fingerprint = _argument_fingerprint
    cls._activate_tool_profiles = _activate_tool_profiles
