from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..guards import is_small_model_name
from ..logging_utils import log_kv, synthetic_trace_id
from ..models.events import (
    UIEvent,
    UIEventType,
    UIStatusSnapshot,
    compute_activity_for_event,
)
from ..challenge_progress import challenge_progress_report
from ..remote_scope import (
    has_any_session_ssh_target,
    handoff_supports_remote_continuation,
    recent_remote_target_paths,
    task_matches_remote_continuation,
)
from ..models.tool_result import ToolEnvelope
from ..state import (
    LOOP_STATE_SCHEMA_VERSION,
    ExperienceMemory,
    json_safe_value,
)
from ..state_support import safe_scratchpad
from ..redaction import redact_sensitive_text
from ..redaction import redact_sensitive_data
from ..tools.profiles import (
    MUTATE_PROFILE,
    NETWORK_PROFILE,
    NETWORK_RAW_PROFILE,
    NETWORK_READ_PROFILE,
    classify_tool_profiles,
)
from ..tools.control_phase_gates import task_involves_interactive_program
from .task_classifier_support import (
    _SSH_COMMAND_TARGET_RE,
    task_is_local_ssh_file_target,
)


def _write_json_file(
    path: Path, payload: dict[str, Any], *, trailing_newline: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(redact_sensitive_data(payload), indent=2)
    if trailing_newline:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _ssh_host_key_failure_causal_chain(state: Any, result: dict[str, Any]) -> str:
    texts: list[str] = []
    texts.extend(
        str(item or "") for item in (getattr(state, "recent_errors", []) or [])
    )
    scratchpad = safe_scratchpad(state)
    if scratchpad is not None:
        latest_blocker = scratchpad.get("_latest_execution_blocker")
        if isinstance(latest_blocker, dict):
            texts.extend(
                str(latest_blocker.get(key) or "")
                for key in ("command", "salient_error", "reason")
            )
        recovery_state = scratchpad.get("_ssh_auth_recovery_state")
        if isinstance(recovery_state, dict):
            for record in recovery_state.values():
                if isinstance(record, dict):
                    texts.extend(
                        str(record.get(key) or "")
                        for key in ("last_error", "last_error_class", "last_command")
                    )
    texts.extend(
        str(result.get(key) or "") for key in ("reason", "message", "last_recent_error")
    )
    combined = "\n".join(texts).lower()
    if not any(
        marker in combined
        for marker in (
            "host key verification",
            "remote host identification has changed",
            "known_hosts",
        )
    ):
        return ""
    blocked_recovery = any(
        marker in combined
        for marker in ("raw_ssh_shell_blocked", "ssh-keygen", "raw `ssh`")
    )
    if blocked_recovery:
        return (
            "ssh_exec failed due to host-key mismatch -> correct local ssh-keygen known_hosts recovery was blocked or required approval -> "
            "subsequent recovery risked wrong tool families -> guard trip/failure"
        )
    return (
        "ssh_exec failed due to host-key mismatch -> local known_hosts trust must be fixed with approved ssh-keygen recovery -> "
        "retry SSH only after trust is fixed"
    )


def _write_checkpoint_file(
    path: Path, result: dict[str, Any], state_snapshot: dict[str, Any]
) -> None:
    payload = redact_sensitive_data(
        {
            "checkpoint_schema_version": 1,
            "loop_state_schema_version": LOOP_STATE_SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "result": json_safe_value(result),
            "state": state_snapshot,
        }
    )
    _write_json_file(path, payload)


async def _emit(
    self: Any,
    handler: Callable[[UIEvent], Awaitable[None] | None] | None,
    event: UIEvent,
    *,
    emit_status: bool = True,
) -> None:
    if handler is None:
        return
    scratchpad = getattr(self.state, "scratchpad", None)
    thread_id = str(getattr(self.state, "thread_id", "") or "").strip()
    task_id = (
        str(self.state.scratchpad.get("_active_task_id") or "").strip()
        if isinstance(scratchpad, dict)
        else ""
    )
    if isinstance(scratchpad, dict):
        ledger = scratchpad.setdefault("_ui_event_ledger", [])
        if isinstance(ledger, list):
            ledger.append(
                {
                    "event_type": str(
                        getattr(event.event_type, "value", event.event_type)
                    ),
                    "content": redact_sensitive_text(str(event.content or "")),
                    "trace_id": thread_id or "",
                    "task_id": task_id,
                }
            )
            if len(ledger) > 80:
                del ledger[:-80]
    if self.run_logger and hasattr(self.run_logger, "log"):
        self.run_logger.log(
            "harness",
            "ui_event",
            "ui event emitted",
            level="debug",
            subsystem="ui",
            event_type=str(getattr(event.event_type, "value", event.event_type)),
            content=redact_sensitive_text(str(event.content or "")),
            trace_id=thread_id or (self.run_logger.extra_fields.get("trace_id") or ""),
            task_id=task_id,
        )
    if event.data.get("is_api_error"):
        if isinstance(scratchpad, dict):
            scratchpad["_ui_api_error_count"] = (
                int(scratchpad.get("_ui_api_error_count", 0) or 0) + 1
            )
    maybe = handler(event)
    if maybe is not None and hasattr(maybe, "__await__"):
        await maybe
    if not emit_status or event.event_type == UIEventType.STATUS:
        return
    snapshot_event = UIEvent(
        event_type=UIEventType.STATUS,
        data={
            "snapshot": self.build_status_snapshot(
                activity=compute_activity_for_event(event, active_task_done=False) or ""
            )
        },
    )
    maybe = handler(snapshot_event)
    if maybe is not None and hasattr(maybe, "__await__"):
        await maybe


def build_status_snapshot(
    self: Any,
    *,
    activity: str = "",
    api_errors: int | None = None,
) -> dict[str, Any]:
    return UIStatusSnapshot.from_harness(
        self,
        self.config,
        activity=activity,
        api_errors=api_errors,
    ).to_dict()


def _inject_recovery_metrics(result: dict[str, Any], state: Any) -> None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    metrics = scratchpad.get("_recovery_metrics")
    if isinstance(metrics, dict) and metrics:
        result["recovery_metrics"] = dict(metrics)


def _task_summary_status(item: dict[str, Any]) -> str:
    return str(item.get("status") or item.get("result_status") or "").strip().lower()


def _task_summary_failed(item: dict[str, Any]) -> bool:
    status = _task_summary_status(item)
    result_status = str(item.get("result_status") or "").strip().lower()
    text = " ".join(
        str(item.get(key) or "")
        for key in ("reason", "message", "last_recent_error", "postmortem_summary")
    )
    if status in {"failed", "aborted", "interrupted", "cancelled", "error", "stopped"}:
        return True
    if status == "completed" and result_status in {
        "failed",
        "aborted",
        "interrupted",
        "cancelled",
        "error",
        "stopped",
    }:
        return True
    if "Guard tripped" in text:
        return True
    if (
        item.get("deliverable_verified") is False
        and item.get("diagnostic_only") is False
    ):
        return True
    return False


def _session_objective_status(
    task_summaries: list[dict[str, Any]], latest_status: str
) -> dict[str, Any]:
    incomplete_ids: list[str] = []
    has_incomplete_prior = False
    latest_index = len(task_summaries)
    for index, item in enumerate(task_summaries, start=1):
        if _task_summary_failed(item):
            incomplete_ids.append(str(item.get("task_id") or f"task-{index:04d}"))
            if index < latest_index:
                has_incomplete_prior = True
    if incomplete_ids:
        overall_status = "incomplete"
    elif latest_status in {"completed", "success", "succeeded"}:
        overall_status = "completed"
    elif latest_status:
        overall_status = latest_status
    else:
        overall_status = "unknown"
    return {
        "local_task_status": latest_status or "unknown",
        "overall_objective_status": overall_status,
        "incomplete_task_ids": incomplete_ids,
        "has_incomplete_prior_tasks": has_incomplete_prior,
    }


def _run_metric_flags(
    state: Any, challenge_progress: dict[str, Any], *, status: str = ""
) -> dict[str, Any]:
    task_text = (
        str(getattr(getattr(state, "run_brief", None), "original_task", "") or "")
        .strip()
        .lower()
    )
    no_op = (
        task_text in {"hi", "hello", "hey", "thanks", "thank you"}
        and int(getattr(state, "step_count", 0) or 0) <= 1
    )
    code_changes = (
        int(challenge_progress.get("code_change_count", 0) or 0)
        if challenge_progress
        else 0
    )
    deliverable_verified = (
        bool(challenge_progress.get("verified_after_last_change"))
        if challenge_progress
        else False
    )
    task_category = str(challenge_progress.get("task_category") or "").strip().lower()
    result_success = status in {
        "completed",
        "complete",
        "success",
        "succeeded",
        "chat_completed",
        "chat_success",
    }
    if (
        not deliverable_verified
        and result_success
        and code_changes <= 0
        and challenge_progress
    ):
        last_verifier_verdict = (
            str(challenge_progress.get("last_verifier_verdict") or "").strip().lower()
        )
        last_verifier_kind = (
            str(challenge_progress.get("last_verifier_kind") or "").strip().lower()
        )
        if (
            task_category == "coding"
            and last_verifier_verdict == "pass"
            and last_verifier_kind == "test_suite"
        ):
            deliverable_verified = True
    # If the run was cancelled, do not claim deliverable_verified=true unless the objective verifier
    # passed after the last failure.
    if status == "cancelled" and deliverable_verified:
        last_verifier = getattr(state, "last_verifier_verdict", None)
        last_verifier_pass = False
        if isinstance(last_verifier, dict):
            last_verifier_pass = (
                str(last_verifier.get("verdict") or "").strip().lower() == "pass"
            )
        if not last_verifier_pass:
            deliverable_verified = False
    diagnostic_only = code_changes <= 0 and not deliverable_verified and not no_op
    if (
        task_category in {"sysadmin", "install", "setup", "deploy", "configure"}
        and deliverable_verified
    ):
        diagnostic_only = False
    return {
        "no_op": no_op,
        "deliverable_verified": deliverable_verified,
        "diagnostic_only": diagnostic_only,
    }


def _finalize(self: Any, result: dict[str, Any]) -> dict[str, Any]:
    status = str((result or {}).get("status") or "").strip().lower()
    if status in {"completed", "success", "succeeded"}:
        scratchpad = getattr(self, "state", None)
        if scratchpad is not None:
            scratchpad = getattr(scratchpad, "scratchpad", None)
        if isinstance(scratchpad, dict):
            scratchpad.pop("_latest_execution_blocker", None)
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
    task_summary_path = str((task_summary or {}).get("summary_path") or "").strip()
    task_id = str((task_summary or {}).get("task_id") or "").strip()
    self._runlog(
        "task_finalize",
        "task finished",
        result=result,
        task_id=task_id,
        task_summary_path=task_summary_path,
    )
    self._record_terminal_experience(result)
    self._rewrite_active_plan_export()
    if self.checkpoint_on_exit:
        self._persist_checkpoint(result)

    result["step_count"] = self.state.step_count
    result["inactive_steps"] = self.state.inactive_steps
    result["token_usage"] = self.state.token_usage
    challenge_progress = challenge_progress_report(self.state)
    if challenge_progress:
        result["challenge_progress"] = challenge_progress
    result.update(_run_metric_flags(self.state, challenge_progress, status=status))
    unverified_change_warning = ""
    terminal_not_success = status not in {
        "completed",
        "complete",
        "success",
        "succeeded",
        "chat_completed",
        "chat_success",
    }
    if terminal_not_success and challenge_progress:
        code_changes = int(challenge_progress.get("code_change_count", 0) or 0)
        verified_after_last_change = bool(
            challenge_progress.get("verified_after_last_change")
        )
        if code_changes > 0 and not verified_after_last_change:
            changed_paths = challenge_progress.get("last_code_change_paths")
            if not isinstance(changed_paths, list):
                changed_paths = []
            path_text = ", ".join(
                str(path) for path in changed_paths[:3] if str(path).strip()
            )
            target_text = f" to {path_text}" if path_text else ""
            unverified_change_warning = f"Task ended with status {status or 'unknown'} after modifying files{target_text}. Changes were not verified after the latest edit."
            result["unverified_change_warning"] = unverified_change_warning
    _inject_recovery_metrics(result, self.state)

    if getattr(self, "run_logger", None) and hasattr(self.run_logger, "run_dir"):
        try:
            postmortem_summary = ""
            if isinstance(result, dict):
                postmortem_summary = str(result.get("reason") or "").strip()
                if not postmortem_summary:
                    interrupt = result.get("interrupt")
                    if isinstance(interrupt, dict):
                        postmortem_summary = str(
                            interrupt.get("question") or ""
                        ).strip()
                if not postmortem_summary:
                    message = result.get("message")
                    if isinstance(message, dict):
                        postmortem_summary = str(
                            message.get("question") or message.get("message") or ""
                        ).strip()
                    elif isinstance(message, str):
                        postmortem_summary = message.strip()
                if not postmortem_summary:
                    postmortem_summary = str(result.get("assistant") or "").strip()
            postmortem_summary = postmortem_summary or "No reason provided"
            scratchpad = getattr(self.state, "scratchpad", {}) or {}
            status = result.get("status", "unknown")
            stall_classification = None
            if status == "timeout":
                if scratchpad.get("_first_tool_dispatch_complete_time") is None:
                    if scratchpad.get("_first_assistant_text_time") is not None:
                        stall_classification = "timeout_pre_tool_after_text"
                    else:
                        stall_classification = "timeout_no_first_tool"
                else:
                    stall_classification = "timeout_after_progress"
            error_type = result.get("error_type", "")
            if error_type == "backend_stream_failure":
                stall_classification = "backend_failed"
            primary_blocker = ""
            latest_blocker = scratchpad.get("_latest_execution_blocker")
            if isinstance(latest_blocker, dict):
                salient = str(latest_blocker.get("salient_error") or "").strip()
                command = str(latest_blocker.get("command") or "").strip()
                if salient:
                    primary_blocker = salient
                    if command:
                        primary_blocker = f"{primary_blocker} [command: {command}]"
            summary_payload = {
                "final_task_status": status,
                "total_tool_calls": self.state.step_count,
                "guard_trips": sum(
                    1
                    for e in (getattr(self.state, "recent_errors", []) or [])
                    if any(
                        marker in str(e)
                        for marker in (
                            "Guard tripped",
                            "file_read_hard_block",
                            "human_resteer",
                        )
                    )
                ),
                "postmortem_summary": postmortem_summary,
                "primary_blocker": primary_blocker,
                "unverified_change_warning": unverified_change_warning,
                "latest_task_id": task_id,
                "latest_task_summary_path": task_summary_path,
                "stall_classification": stall_classification,
                "error_type": error_type,
                **_run_metric_flags(self.state, challenge_progress, status=status),
            }
            causal_chain = _ssh_host_key_failure_causal_chain(self.state, result)
            if causal_chain:
                summary_payload["causal_chain"] = causal_chain
                if not postmortem_summary or "Guard tripped" in postmortem_summary:
                    summary_payload["postmortem_summary"] = causal_chain
            if challenge_progress:
                summary_payload["challenge_progress"] = challenge_progress
            summary_path = self.run_logger.run_dir / "task_summary.json"
            schedule = getattr(self, "_schedule_background_persistence", None)
            if callable(schedule):
                schedule(
                    _write_json_file,
                    summary_path,
                    summary_payload,
                    trailing_newline=True,
                )
            else:
                _write_json_file(summary_path, summary_payload, trailing_newline=True)
            session_summary_path = self.run_logger.run_dir / "session_summary.json"
            tasks_dir = self.run_logger.run_dir / "tasks"
            task_summary_paths = (
                [
                    str(path)
                    for path in sorted(tasks_dir.glob("task-*/task_summary.json"))
                ]
                if tasks_dir.exists()
                else []
            )
            task_summaries: list[dict[str, Any]] = []
            for path_text in task_summary_paths:
                try:
                    payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(payload, dict):
                    task_summaries.append(payload)
            if isinstance(task_summary, dict) and task_summary:
                current_summary = dict(task_summary)
                current_summary.setdefault(
                    "step_count", summary_payload.get("total_tool_calls", 0)
                )
                current_task_id = str(current_summary.get("task_id") or "").strip()
                current_summary_path = str(
                    current_summary.get("summary_path") or ""
                ).strip()
                matched_current_summary = False
                for index, item in enumerate(task_summaries):
                    item_task_id = str(item.get("task_id") or "").strip()
                    item_summary_path = str(item.get("summary_path") or "").strip()
                    if (current_task_id and item_task_id == current_task_id) or (
                        current_summary_path
                        and item_summary_path == current_summary_path
                    ):
                        task_summaries[index] = {**item, **current_summary}
                        matched_current_summary = True
                        break
                if not matched_current_summary:
                    task_summaries.append(current_summary)
                current_summary_path = str(
                    task_summary.get("summary_path") or ""
                ).strip()
                if (
                    current_summary_path
                    and current_summary_path not in task_summary_paths
                ):
                    task_summary_paths.append(current_summary_path)
                # Ensure the latest task identity is always reflected, even when
                # the current task was interrupted/replaced and not yet on disk.
                summary_payload["latest_task_id"] = (
                    current_task_id or summary_payload.get("latest_task_id", "")
                )
                summary_payload["latest_task_summary_path"] = (
                    current_summary_path
                    or summary_payload.get("latest_task_summary_path", "")
                )
            session_total_tool_calls = 0
            for item in task_summaries:
                try:
                    session_total_tool_calls += max(
                        0,
                        int(
                            item.get("step_count") or item.get("total_tool_calls") or 0
                        ),
                    )
                except (TypeError, ValueError):
                    continue
            session_guard_trips = 0
            for item in task_summaries:
                text = " ".join(
                    str(item.get(key) or "")
                    for key in ("reason", "message", "last_recent_error")
                )
                if "Guard tripped" in text:
                    session_guard_trips += 1
            prior_task_completed = any(
                str(item.get("status") or item.get("result_status") or "")
                .strip()
                .lower()
                in {"completed", "success", "succeeded"}
                for item in task_summaries[:-1]
            )
            latest_task_cancelled = status == "cancelled" or any(
                str(item.get("terminal_event") or "").strip() == "task_interrupted"
                for item in task_summaries[-1:]
            )
            session_objective = _session_objective_status(task_summaries, status)
            session_summary_payload = {
                **summary_payload,
                **session_objective,
                "final_task_status": status or summary_payload.get("status", ""),
                "total_tool_calls": session_total_tool_calls
                or summary_payload.get("total_tool_calls", 0),
                "guard_trips": session_guard_trips
                or summary_payload.get("guard_trips", 0),
                "task_count": len(task_summary_paths),
                "task_summary_paths": task_summary_paths,
                "prior_task_completed": prior_task_completed,
                "latest_task_cancelled": latest_task_cancelled,
                "files_changed_after_latest_task_start": bool(
                    challenge_progress
                    and int(challenge_progress.get("code_change_count", 0) or 0) > 0
                ),
                "verification_after_latest_change": bool(
                    challenge_progress
                    and challenge_progress.get("verified_after_last_change")
                ),
            }
            run_summary_path = self.run_logger.run_dir / "run_summary.json"
            run_summary_payload = {
                **session_summary_payload,
                "summary_kind": "run",
                "run_dir": str(self.run_logger.run_dir),
                "session_summary_path": str(session_summary_path),
                "latest_task_summary_path": summary_payload.get(
                    "latest_task_summary_path", ""
                ),
            }
            if callable(schedule):
                schedule(
                    _write_json_file,
                    session_summary_path,
                    session_summary_payload,
                    trailing_newline=True,
                )
                schedule(
                    _write_json_file,
                    run_summary_path,
                    run_summary_payload,
                    trailing_newline=True,
                )
            else:
                _write_json_file(
                    session_summary_path, session_summary_payload, trailing_newline=True
                )
                _write_json_file(
                    run_summary_path, run_summary_payload, trailing_newline=True
                )
        except Exception:
            self.log.exception("failed to write finalization summaries")

    self._cancel_requested = False
    self._active_dispatch_task = None
    return result


def _reset_cancel_requested(self: Any) -> None:
    self._cancel_requested = False
    self._cancel_source = ""


def _rewrite_active_plan_export(self: Any) -> None:
    plan = self.state.active_plan or self.state.draft_plan
    if plan is None or not plan.requested_output_path:
        return
    try:
        from ..plans import write_plan_file

        write_plan_file(
            plan,
            plan.requested_output_path,
            format=plan.requested_output_format,
            cwd=getattr(self.state, "cwd", None),
        )
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
    return self.subtasks.build_subtask_result(
        child=child, request=request, result=result
    )


def _persist_checkpoint(self: Any, result: dict[str, Any]) -> None:
    path = (
        Path(self.checkpoint_path).resolve()
        if self.checkpoint_path
        else Path(self.state.cwd).resolve() / ".smallctl-checkpoint.json"
    )
    result_snapshot = dict(result)
    try:
        # Snapshot state on the event loop so the background writer does not
        # serialize the live object while it is being mutated.
        state_snapshot = self.state.to_dict(
            artifact_store=getattr(self, "artifact_store", None)
        )
        schedule = getattr(self, "_schedule_background_persistence", None)
        if callable(schedule):
            schedule(_write_checkpoint_file, path, result_snapshot, state_snapshot)
        else:
            _write_checkpoint_file(path, result_snapshot, state_snapshot)
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


def _runlog(
    self: Any,
    event: str,
    message: str,
    *,
    level: str = "info",
    subsystem: str | None = None,
    **data: Any,
) -> None:
    if self.run_logger:
        if not self.run_logger.extra_fields.get("trace_id"):
            self.run_logger.set_trace_id(
                synthetic_trace_id(self.state, suffix="harness")
            )
        self.run_logger.log(
            "harness", event, message, level=level, subsystem=subsystem, **data
        )
        if event.startswith("model_"):
            self.run_logger.log(
                "model_output", event, message, level=level, subsystem=subsystem, **data
            )


def _stream_print(text: str) -> None:
    try:
        print(text, end="", flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(
            encoding, errors="replace"
        )
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


async def _shrink_messages_for_prompt_processing_timeout(
    self: Any,
    *,
    messages: list[dict[str, Any]],
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> list[dict[str, Any]] | None:
    """Shrink messages after the backend spends too long processing the prompt.

    llama.cpp with small Gemma-4 can invalidate its prompt cache on long contexts,
    causing each turn to re-process the entire prompt. Drop old messages to leave
    only the system prompt and the most recent exchange, then rebuild so the
    context policy can apply the reduced budget.
    """
    system_prompt = str(self.state.scratchpad.get("_last_system_prompt") or "")
    if not system_prompt:
        return None
    # Reduce the effective prompt budget to create headroom and reduce cache
    # pressure on the next turn. Use the SWA-aware cap when available.
    policy = getattr(self, "context_policy", None)
    if policy is not None:
        current_limit = getattr(policy, "max_prompt_tokens", None)
        target_cap = getattr(policy, "swa_prompt_cap", 16384)
        if isinstance(current_limit, int) and current_limit > target_cap:
            policy.max_prompt_tokens = target_cap
            self._runlog(
                "prompt_processing_timeout_budget_cap",
                "capped prompt budget to reduce cache pressure",
                new_max_prompt_tokens=target_cap,
                previous_max_prompt_tokens=current_limit,
            )
    return await self._build_prompt_messages(system_prompt, event_handler=event_handler)


async def _build_prompt_messages(
    self: Any,
    system_prompt: str,
    *,
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> list[dict[str, Any]]:
    return await self.prompt_builder.build_messages(
        system_prompt, event_handler=event_handler
    )


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
    if not self._configured_tool_profiles:
        task = self.state.run_brief.original_task or self._current_user_task()
        prior_profiles = list(self.state.active_tool_profiles or [])
        self._activate_tool_profiles(task)
        new_profiles = list(self.state.active_tool_profiles or [])
        if new_profiles != prior_profiles:
            self._runlog(
                "tool_profiles_refresh",
                "refreshed tool profiles after intent change",
                task=task,
                profiles=new_profiles,
                prior_profiles=prior_profiles,
                active_intent=self.state.active_intent,
            )


def _completion_next_action(self: Any) -> str:
    return "Decide whether the current evidence is sufficient; call task_complete when it is."


def _is_small_model_name(self: Any, model_name: str | None) -> bool:
    return is_small_model_name(model_name)


def switch_model(self: Any, model: str) -> None:
    model_name = str(model or "").strip()
    if not model_name:
        raise ValueError("model name cannot be empty")

    from .bootstrap_support import build_client, resolve_provider_profile

    config = self.config
    endpoint = str(config.endpoint or getattr(self.client, "base_url", "")).rstrip("/")
    api_key = config.api_key or getattr(self.client, "api_key", None)
    provider_profile = str(
        config.provider_profile or getattr(self, "provider_profile", "generic")
    )
    resolved_provider_profile = resolve_provider_profile(
        endpoint, model_name, provider_profile
    )

    self.client = build_client(
        endpoint=endpoint,
        model=model_name,
        api_key=api_key,
        chat_endpoint=str(
            config.chat_endpoint
            or getattr(self.client, "chat_endpoint", "/chat/completions")
        ),
        provider_profile=resolved_provider_profile,
        first_token_timeout_sec=config.first_token_timeout_sec,
        runtime_context_probe=bool(config.runtime_context_probe),
        run_logger=getattr(self, "run_logger", None),
        backend_recovery_handler=self.recover_backend_wedge,
        max_completion_tokens=getattr(config, "max_completion_tokens", None),
    )
    self.provider_profile = self.client.provider_profile
    config.model = model_name
    config.provider_profile = self.provider_profile
    config.context_limit = None
    self.state.scratchpad["_model_name"] = model_name
    self.state.scratchpad["_model_is_small"] = self._is_small_model_name(model_name)
    self.discovered_server_context_limit = None
    self.server_context_limit = None
    self._runtime_context_probe_attempted = False


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


def _normalize_failure_mode(
    self: Any, error: Any, *, tool_name: str, success: bool
) -> str:
    return self.memory._normalize_failure_mode(
        error, tool_name=tool_name, success=success
    )


def _reinforce_retrieved_experiences(
    self: Any, *, tool_name: str, success: bool
) -> None:
    self.memory._reinforce_retrieved_experiences(tool_name=tool_name, success=success)


def _record_terminal_experience(self: Any, result: dict[str, Any]) -> None:
    self.memory.record_terminal_experience(result)


def _argument_fingerprint(self: Any, arguments: Any) -> str:
    return self.memory._argument_fingerprint(arguments)


def _task_mentions_remote_web_continuation(state: Any, task: str) -> bool:
    text = " ".join(str(task or "").strip().lower().split())
    if not text:
        return False
    if any(
        marker in text
        for marker in ("/home/", " local repo", " in this repo", " locally")
    ):
        return False
    if any(marker in text for marker in ("/var/www/", "/etc/nginx", "/srv/", "/opt/")):
        return True

    remote_paths = recent_remote_target_paths(state)
    has_remote_web_path = any(
        str(path).startswith("/var/www/")
        or str(path).endswith(".html")
        or str(path).endswith(".htm")
        or str(path).endswith(".css")
        for path in remote_paths
    )
    if not has_remote_web_path:
        return False
    web_hints = (
        "background",
        "button",
        "buttons",
        "color",
        "colors",
        "css",
        "design",
        "font",
        "fonts",
        "html",
        "layout",
        "page",
        "pages",
        "site",
        "style",
        "styling",
        "theme",
        "website",
    )
    return any(hint in text for hint in web_hints)


def _looks_like_soft_resteer(current_task: str, last_task: str) -> bool:
    """Detect if the current task is a soft resteer (continuation) of the previous task."""
    current = str(current_task or "").strip().lower()
    last = str(last_task or "").strip().lower()
    if not current or not last:
        return False
    if current == last:
        return True
    current_words = set(current.split())
    last_words = set(last.split())
    if not current_words or not last_words:
        return False
    overlap = len(current_words & last_words)
    similarity = overlap / max(len(current_words), len(last_words))
    return similarity >= 0.6


_REMOTE_INSTALL_MARKERS: tuple[str, ...] = (
    "install",
    "setup",
    "deploy",
    "installer",
    "installfog",
    "bootstrap",
)

_INTERACTIVE_REMOTE_INSTALLER_MARKERS: tuple[str, ...] = (
    "installfog",
    "installfog.sh",
    "fog",
    "pxe",
    "installer",
    "setup",
    "package config",
    "follow prompts",
    "interactive install",
    "run the installer",
)


def _looks_like_remote_install_task(task: str) -> bool:
    text = str(task or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in _REMOTE_INSTALL_MARKERS)


def _looks_like_interactive_remote_installer(task: str) -> bool:
    text = str(task or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in _INTERACTIVE_REMOTE_INSTALLER_MARKERS)


def _state_signals_interactive_program(state: Any) -> bool:
    try:
        return bool(task_involves_interactive_program(state))
    except Exception:
        return False


def _numeric_scratchpad_signal(
    scratchpad: dict[str, Any], keys: tuple[str, ...]
) -> bool:
    for key in keys:
        try:
            if int(scratchpad.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _state_signals_remote_install_recovery(state: Any) -> bool:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    if scratchpad.get("_install_source_invalid_blocker"):
        return True
    diagnosis = scratchpad.get("_install_source_diagnosis")
    if isinstance(diagnosis, dict):
        if diagnosis.get("public_dns_nxdomain") and diagnosis.get("network_ok"):
            if _numeric_scratchpad_signal(
                diagnosis, ("invalid_fetch_count", "resolve_fail_count")
            ):
                return True
        if diagnosis.get("install_context_resolve_failed"):
            return True
    if _numeric_scratchpad_signal(
        scratchpad,
        (
            "_install_source_invalid_fetch_count",
            "_install_source_resolve_fail_count",
            "_remote_install_fetch_fail_count",
            "_remote_install_resolve_fail_count",
        ),
    ):
        return True
    preflight = scratchpad.get("_remote_installer_preflight")
    if isinstance(preflight, dict):
        for entry in preflight.values():
            if isinstance(entry, dict) and str(entry.get("status") or "").strip() in {
                "required",
                "missing_critical_files",
                "corrupt",
            }:
                return True
    return False


def _activate_tool_profiles(self: Any, task: str) -> None:
    # Fix 6: Freeze tool profiles on soft resteers to prevent tool whiplash
    existing_profiles = list(getattr(self.state, "active_tool_profiles", []) or [])
    last_task = str(self.state.scratchpad.get("_last_task_text") or "").strip()
    if existing_profiles and _looks_like_soft_resteer(task, last_task):
        self.state.scratchpad["_last_task_text"] = task
        self._runlog(
            "tool_profiles",
            "preserved tool profiles on soft resteer",
            task=task,
            profiles=existing_profiles,
            source="soft_resteer",
        )
        return

    if self._configured_tool_profiles:
        profiles = set(self._configured_tool_profiles)
    else:
        profiles = classify_tool_profiles(task)
        handoff = self.state.scratchpad.get("_last_task_handoff")
        prior_profiles = (
            handoff.get("active_tool_profiles") if isinstance(handoff, dict) else None
        )
        task_mode = str(getattr(self.state, "task_mode", "") or "").strip().lower()
        if task_mode == "remote_execute":
            profiles.add(NETWORK_PROFILE)
        elif task_mode == "local_execute":
            # Coding/local tasks do not need SSH tools; removing them prevents
            # small models from hallucinating remote operations during local work.
            # However, if the user explicitly wrote an "ssh user@host" command,
            # do not strip the network profile; the task is a remote shell request.
            if _SSH_COMMAND_TARGET_RE.search(task):
                profiles.add(NETWORK_PROFILE)
            else:
                profiles.discard(NETWORK_PROFILE)
                profiles.discard(NETWORK_RAW_PROFILE)
                if task_is_local_ssh_file_target(task):
                    profiles.discard(NETWORK_READ_PROFILE)
        resolved_remote = self.state.scratchpad.get("_resolved_remote_followup")
        if isinstance(resolved_remote, dict) and resolved_remote:
            profiles.add(NETWORK_PROFILE)
        elif handoff_supports_remote_continuation(
            self.state
        ) and task_matches_remote_continuation(self.state, task):
            profiles.add(NETWORK_PROFILE)
            if isinstance(prior_profiles, list) and NETWORK_PROFILE in prior_profiles:
                profiles.add(NETWORK_PROFILE)
        elif has_any_session_ssh_target(
            self.state
        ) and _task_mentions_remote_web_continuation(self.state, task):
            profiles.add(NETWORK_PROFILE)

        from ..remote_scope import remote_scope_is_active

        state_mode = str(getattr(self.state, "task_mode", "") or "").strip().lower()
        active_intent = (
            str(getattr(self.state, "active_intent", "") or "").strip().lower()
        )
        if remote_scope_is_active(self.state) and (
            state_mode == "remote_execute"
            or active_intent == "requested_ssh_exec"
            or isinstance(resolved_remote, dict)
            and resolved_remote
        ):
            profiles.add(NETWORK_PROFILE)
        if active_intent in {
            "author_write",
            "write_file",
            "requested_write_file",
            "requested_file_write",
            "requested_file_append",
            "requested_file_patch",
            "requested_ast_patch",
            "requested_file_delete",
        }:
            profiles.add(MUTATE_PROFILE)
        if isinstance(prior_profiles, list) and NETWORK_READ_PROFILE in prior_profiles:
            if (
                self.state.scratchpad.get("_task_boundary_previous_task")
                or isinstance(resolved_remote, dict)
                and resolved_remote
                or task_matches_remote_continuation(self.state, task)
            ):
                profiles.add(NETWORK_READ_PROFILE)

    task_mode = str(getattr(self.state, "task_mode", "") or "").strip().lower()
    if task_mode == "remote_execute":
        if _looks_like_remote_install_task(
            task
        ) or _state_signals_remote_install_recovery(self.state):
            if NETWORK_READ_PROFILE not in profiles:
                profiles.add(NETWORK_READ_PROFILE)
                self._runlog(
                    "tool_profiles",
                    "added network_read for remote install recovery",
                    task=task,
                    source="remote_install_recovery_network_read",
                )
        if _looks_like_remote_install_task(task) and (
            _looks_like_interactive_remote_installer(task)
            or _state_signals_interactive_program(self.state)
        ):
            scratchpad = (
                self.state.scratchpad
                if isinstance(getattr(self.state, "scratchpad", None), dict)
                else {}
            )
            if not scratchpad.get("_expose_interactive_session_tools"):
                scratchpad["_expose_interactive_session_tools"] = True
                self._runlog(
                    "tool_profiles",
                    "exposed interactive session tools for remote installer",
                    task=task,
                    source="remote_installer_interactive_session",
                )

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
    cls.build_status_snapshot = build_status_snapshot
    cls._finalize = _finalize
    cls._reset_cancel_requested = _reset_cancel_requested
    cls._rewrite_active_plan_export = _rewrite_active_plan_export
    cls._create_child_harness = _create_child_harness
    cls._build_subtask_result = _build_subtask_result
    cls._persist_checkpoint = _persist_checkpoint
    cls._failure = staticmethod(_failure)
    cls._runlog = _runlog
    cls._stream_print = staticmethod(_stream_print)
    cls._rebuild_messages_after_context_overflow = (
        _rebuild_messages_after_context_overflow
    )
    cls._shrink_messages_for_prompt_processing_timeout = (
        _shrink_messages_for_prompt_processing_timeout
    )
    cls._build_prompt_messages = _build_prompt_messages
    cls._maybe_compact_context = _maybe_compact_context
    cls._update_working_memory = _update_working_memory
    cls._refresh_active_intent = _refresh_active_intent
    cls._completion_next_action = _completion_next_action
    cls._is_small_model_name = _is_small_model_name
    cls.switch_model = switch_model
    cls._record_experience = _record_experience
    cls._normalize_failure_mode = _normalize_failure_mode
    cls._reinforce_retrieved_experiences = _reinforce_retrieved_experiences
    cls._record_terminal_experience = _record_terminal_experience
    cls._argument_fingerprint = _argument_fingerprint
    cls._activate_tool_profiles = _activate_tool_profiles
