from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from ..interrupt_replies import is_interrupt_response
from ..logging_utils import log_kv
from ..models.events import UIEvent


async def _run_sync_persistence_task(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    return await asyncio.to_thread(func, *args, **kwargs)


class HarnessRunAlreadyActiveError(RuntimeError):
    """Raised when a harness run starts while another run is in flight."""


def _reset_cancel_flag_at_run_start(self: Any) -> None:
    reset = getattr(self, "_reset_cancel_requested", None)
    if callable(reset):
        reset()
        return
    self._cancel_requested = False


def _begin_run_guard(self: Any) -> None:
    if getattr(self, "_run_guard_in_flight", False) is True:
        raise HarnessRunAlreadyActiveError(
            "A harness run is already active on this harness; wait for it to "
            "finish or cancel it before starting another."
        )
    self._run_guard_in_flight = True
    _reset_cancel_flag_at_run_start(self)


def _end_run_guard(self: Any) -> None:
    self._run_guard_in_flight = False


def _track_background_persistence_task(self: Any, task: asyncio.Task[Any]) -> asyncio.Task[Any]:
    tasks = getattr(self, "_background_persistence_tasks", None)
    if tasks is None:
        tasks = set()
        self._background_persistence_tasks = tasks
    tasks.add(task)
    task.add_done_callback(lambda done: tasks.discard(done))
    return task


def _schedule_background_persistence(
    self: Any,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> asyncio.Task[Any] | None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        func(*args, **kwargs)
        return None
    task = loop.create_task(_run_sync_persistence_task(func, *args, **kwargs))
    return _track_background_persistence_task(self, task)


async def _drain_background_persistence_tasks(self: Any) -> None:
    tasks = list(getattr(self, "_background_persistence_tasks", set()) or [])
    if not tasks:
        return
    await asyncio.gather(*tasks, return_exceptions=True)


def _persist_chat_session_state_sync(
    cwd: str,
    thread_id: str,
    state_payload: dict[str, Any],
    model: str,
) -> None:
    from ..chat_sessions import persist_chat_session_state

    persist_chat_session_state(
        cwd=cwd,
        thread_id=thread_id,
        state_payload=state_payload,
        model=model,
    )


def _persist_chat_session_state_from_runtime_state_sync(
    cwd: str,
    thread_id: str,
    state_payload: dict[str, Any],
    model: str,
) -> None:
    _persist_chat_session_state_sync(
        cwd=cwd,
        thread_id=thread_id,
        state_payload=state_payload,
        model=model,
    )


def _close_process_pipe(pipe: Any) -> None:
    if pipe is None:
        return

    close = getattr(pipe, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass

    for attr_name in ("transport", "_transport"):
        transport = getattr(pipe, attr_name, None)
        if transport is None:
            continue
        try:
            transport.close()
        except Exception:
            pass


def _close_process_transports(proc: Any) -> None:
    for attr_name in ("stdin", "stdout", "stderr"):
        _close_process_pipe(getattr(proc, attr_name, None))

    transport = getattr(proc, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:
            pass


def _approved_plan_matches_interrupt(self: Any, interrupt: dict[str, Any]) -> bool:
    if str(interrupt.get("kind") or "") != "plan_execute_approval":
        return False
    plan_id = str(interrupt.get("plan_id") or "").strip()
    state = getattr(self, "state", None)
    if state is None:
        return False
    for plan in (getattr(state, "active_plan", None), getattr(state, "draft_plan", None)):
        if plan is None or not bool(getattr(plan, "approved", False)):
            continue
        if not plan_id or str(getattr(plan, "plan_id", "") or "").strip() == plan_id:
            return True
    return False


def _planner_interrupt_payload(self: Any) -> dict[str, Any] | None:
    planner_interrupt = getattr(getattr(self, "state", None), "planner_interrupt", None)
    if planner_interrupt is None:
        return None
    payload = {
        "kind": getattr(planner_interrupt, "kind", "plan_execute_approval"),
        "question": getattr(planner_interrupt, "question", ""),
        "plan_id": getattr(planner_interrupt, "plan_id", ""),
        "approved": getattr(planner_interrupt, "approved", False),
        "response_mode": getattr(planner_interrupt, "response_mode", "yes/no/revise"),
    }
    if _approved_plan_matches_interrupt(self, payload):
        return None
    return payload


async def _run_teardown(self: Any) -> None:
    from ..logging_utils import log_kv

    shutdown_reason = str(getattr(self, "_pending_task_shutdown_reason", "") or "").strip()
    if shutdown_reason:
        self._finalize_task_scope(
            terminal_event="task_interrupted",
            status="interrupted",
            reason=shutdown_reason,
        )
        self._pending_task_shutdown_reason = ""

    try:
        from ..graph.lifecycle_nodes_support import _flush_open_write_sessions

        await _flush_open_write_sessions(
            self,
            reason="teardown_abandoned",
            event_suffix="teardown",
        )
    except Exception:
        pass

    # Session closure audit log (Fix 7)
    state = getattr(self, "state", None)
    session_id = str(getattr(state, "thread_id", "") or getattr(self, "conversation_id", "") or "").strip()
    step_count = int(getattr(state, "step_count", 0) or 0) if state else 0
    task_received_at = str(getattr(state, "task_received_at", "") or "").strip() if state else ""
    created_at = str(getattr(state, "created_at", "") or "").strip() if state else ""
    duration_sec = 0.0
    if created_at:
        try:
            from datetime import datetime, timezone
            created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            duration_sec = (datetime.now(timezone.utc) - created_dt).total_seconds()
        except Exception:
            pass

    reason = "idle"
    if shutdown_reason:
        reason = "interrupted"
    elif step_count > 0:
        reason = "task_complete" if getattr(state, "last_verifier_verdict", None) else "closed"

    # Idle session detection warning (Fix 2)
    if step_count == 0 and not task_received_at:
        log_kv(
            self.log if hasattr(self, "log") else logging.getLogger("smallctl.harness"),
            logging.WARNING,
            "session_idle",
            msg="Session closed without executing tasks. If this was unintentional, check that the TUI rendered correctly and the task was submitted.",
            session_id=session_id,
            duration_sec=round(duration_sec, 2),
        )

    # Include context pipeline metrics if available (Fix 6)
    context_metrics = {}
    if state and hasattr(state, "scratchpad") and isinstance(state.scratchpad, dict):
        context_metrics = dict(state.scratchpad.get("_context_metrics", {}))

    assemble_calls = context_metrics.get("assemble_calls", 0)
    if assemble_calls == 0:
        log_kv(
            self.log if hasattr(self, "log") else logging.getLogger("smallctl.harness"),
            logging.WARNING,
            "context_pipeline_idle",
            msg="Context pipeline was never exercised.",
            session_id=session_id,
        )

    log_kv(
        self.log if hasattr(self, "log") else logging.getLogger("smallctl.harness"),
        logging.INFO,
        "session_closing",
        session_id=session_id,
        duration_sec=round(duration_sec, 2),
        step_count=step_count,
        task_submitted=bool(task_received_at),
        reason=reason,
        context_metrics=context_metrics if context_metrics else None,
    )

    # Extract model calls, tools dispatched, and fama configuration (Fix 7)
    model_calls = 0
    tools_dispatched = 0
    fama_active = True
    if state and hasattr(state, "scratchpad") and isinstance(state.scratchpad, dict):
        model_calls = int(state.scratchpad.get("_model_calls", 0))
        tools_dispatched = int(state.scratchpad.get("_tools_dispatched", 0))
        fama_config = state.scratchpad.get("_fama_config", {})
        if isinstance(fama_config, dict):
            fama_active = fama_config.get("enabled", True)

    log_kv(
        self.log if hasattr(self, "log") else logging.getLogger("smallctl.harness"),
        logging.INFO,
        "session_audit",
        session_id=session_id,
        duration_sec=round(duration_sec, 2),
        step_count=step_count,
        task_submitted=bool(task_received_at),
        model_calls=model_calls,
        tools_dispatched=tools_dispatched,
        fama_active=fama_active,
        reason=reason,
    )

    self.event_handler = None
    autosave = getattr(self, "_autosave_chat_session_state", None)
    if callable(autosave):
        try:
            autosave()
        except Exception:
            pass
    await _drain_background_persistence_tasks(self)

    try:
        from ..client import OpenAICompatClient
        await OpenAICompatClient.aclose_shared_clients()
    except Exception as exc:
        log = self.log if hasattr(self, "log") else logging.getLogger("smallctl.harness")
        log.warning("Failed to close shared API clients during teardown: %s", exc)

    self.approvals.reject_pending_shell_approvals()
    self.approvals.reject_pending_sudo_password_prompts()

    procs = list(getattr(self, "_active_processes", set()) or [])
    if hasattr(self, "_active_processes"):
        self._active_processes.clear()

    for proc in procs:
        try:
            if proc.returncode is None:
                proc.terminate()
        except (ProcessLookupError, RuntimeError):
            pass
        except Exception:
            pass

    wait_tasks = [asyncio.create_task(proc.wait()) for proc in procs]
    if wait_tasks:
        done, pending = await asyncio.wait(wait_tasks, timeout=0.3)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if done:
            await asyncio.gather(*done, return_exceptions=True)

    for proc in procs:
        try:
            if proc.returncode is None:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=1.0)
        except Exception:
            pass
        finally:
            _close_process_transports(proc)

async def run_chat_with_events(
    self: Any,
    task: str,
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    from ..graph.runtime import ChatGraphRuntime
    from datetime import datetime, timezone

    _begin_run_guard(self)
    try:
        redirected = await _maybe_resume_pending_interrupt(self, task, event_handler=event_handler)
        if redirected is not None:
            return redirected

        # Track task receipt time (Fix 2)
        state = getattr(self, "state", None)
        if state is not None:
            state.task_received_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            touch = getattr(state, "touch", None)
            if callable(touch):
                touch()

        log_kv(
            self.log if hasattr(self, "log") else logging.getLogger("smallctl.harness"),
            logging.INFO,
            "task_received",
            mode="chat",
            task_length=len(task),
            session_id=str(getattr(state, "thread_id", "") or "").strip() if state else "",
        )

        self.event_handler = event_handler
        runtime = ChatGraphRuntime.from_harness(
            self,
            event_handler=event_handler,
        )
        return await runtime.run(task)
    finally:
        _end_run_guard(self)


async def run_auto_with_events(
    self: Any,
    task: str,
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    from ..graph.runtime import AutoGraphRuntime
    from datetime import datetime, timezone

    _begin_run_guard(self)
    try:
        redirected = await _maybe_resume_pending_interrupt(self, task, event_handler=event_handler)
        if redirected is not None:
            return redirected

        # Track task receipt time (Fix 2)
        state = getattr(self, "state", None)
        if state is not None:
            state.task_received_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            touch = getattr(state, "touch", None)
            if callable(touch):
                touch()

        # Lifecycle telemetry (Fix 1)
        log_kv(
            self.log if hasattr(self, "log") else logging.getLogger("smallctl.harness"),
            logging.INFO,
            "task_received",
            task_length=len(task),
            session_id=str(getattr(state, "thread_id", "") or "").strip() if state else "",
        )

        self.event_handler = event_handler
        runtime = AutoGraphRuntime.from_harness(
            self,
            event_handler=event_handler,
        )
        return await runtime.run(task)
    finally:
        _end_run_guard(self)


def set_interactive_shell_approval(self: Any, enabled: bool) -> None:
    self.allow_interactive_shell_approval = bool(enabled)
    self.config.allow_interactive_shell_approval = self.allow_interactive_shell_approval


def set_shell_approval_session_default(self: Any, enabled: bool) -> None:
    self.shell_approval_session_default = bool(enabled)
    self.config.shell_approval_session_default = self.shell_approval_session_default


def set_planning_mode(self: Any, enabled: bool) -> dict[str, Any]:
    enabled_flag = bool(enabled)
    self.state.planning_mode_enabled = enabled_flag
    if enabled_flag:
        self.state.planner_resume_target_mode = "loop"
    else:
        self.state.planner_requested_output_path = ""
        self.state.planner_requested_output_format = ""
    self.state.touch()
    return self.build_status_snapshot()


async def run_task(self: Any, task: str) -> dict[str, Any]:
    return await self.run_task_with_events(task)


async def run_subtask(
    self: Any,
    brief: str,
    phase: str = "plan",
    depth: int = 1,
    max_prompt_tokens: int | None = None,
    recent_message_limit: int = 4,
    metadata: dict[str, Any] | None = None,
    harness_factory: Callable[..., Any] | None = None,
    artifact_start_index: int | None = None,
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> Any:
    return await self.subtasks.run_subtask(
        brief=brief,
        phase=phase,
        depth=depth,
        max_prompt_tokens=max_prompt_tokens,
        recent_message_limit=recent_message_limit,
        metadata=metadata,
        harness_factory=harness_factory,
        artifact_start_index=artifact_start_index,
        event_handler=event_handler,
    )


async def run_auto(
    self: Any,
    task: str,
    *,
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    from ..graph.runtime import IndexerGraphRuntime

    if self._indexer:
        runtime = IndexerGraphRuntime.from_harness(self, event_handler=event_handler)
        return await runtime.run(task)

    return await self.run_auto_with_events(task, event_handler=event_handler)


async def resume_task(self: Any, human_input: str) -> dict[str, Any]:
    return await self.resume_task_with_events(human_input)


def restore_graph_state(self: Any, thread_id: str | None = None) -> bool:
    from ..graph.runtime import LoopGraphRuntime
    from ..state_memory import _trim_recent_messages

    runtime = LoopGraphRuntime.from_harness(self)
    restored = runtime.restore(thread_id=thread_id)
    if restored:
        recent_limit = getattr(getattr(self, "context_policy", None), "recent_message_limit", None)
        if recent_limit is not None:
            self.state.recent_message_limit = max(1, int(recent_limit))
            self.state.recent_messages = _trim_recent_messages(
                list(getattr(self.state, "recent_messages", []) or []),
                limit=self.state.recent_message_limit,
            )
        if isinstance(getattr(self.state, "scratchpad", None), dict):
            self.state.scratchpad["_session_restored"] = True
            self.state.scratchpad["_resume_contract"] = {
                "kind": "chat_session_resume",
                "thread_id": str(getattr(self.state, "thread_id", "") or thread_id or "").strip(),
            }
        self._sync_run_logger_session_id()
    return restored


def _sync_run_logger_session_id(self: Any) -> None:
    run_logger = getattr(self, "run_logger", None)
    if run_logger is None or not hasattr(run_logger, "set_session_id"):
        return
    session_id = str(getattr(self.state, "thread_id", "") or self.conversation_id or "").strip()
    if not session_id:
        return
    try:
        run_logger.set_session_id(session_id)
    except Exception:
        self.log.debug("Unable to sync run logger session id", exc_info=True)


def _autosave_chat_session_state(self: Any) -> None:
    state = getattr(self, "state", None)
    if state is None:
        return
    thread_id = str(getattr(state, "thread_id", "") or getattr(self, "conversation_id", "") or "").strip()
    cwd = str(getattr(state, "cwd", "") or "").strip()
    if not thread_id or not cwd:
        return
    # Skip autosaving if session is idle (Fix 2)
    step_count = int(getattr(state, "step_count", 0) or 0)
    task_received_at = str(getattr(state, "task_received_at", "") or "").strip()
    recent_messages = getattr(state, "recent_messages", None)
    if step_count == 0 and not task_received_at and isinstance(recent_messages, list) and not recent_messages:
        return
    try:
        client = getattr(self, "client", None)
        # Snapshot state on the event loop so the background writer does not
        # serialize the live object while it is being mutated.
        state_payload = state.to_dict(artifact_store=getattr(self, "artifact_store", None))
        _schedule_background_persistence(
            self,
            _persist_chat_session_state_from_runtime_state_sync,
            cwd,
            thread_id,
            state_payload,
            str(getattr(client, "model", "") or ""),
        )
    except Exception:
        self.log.debug("Unable to autosave chat session state", exc_info=True)


async def decide_run_mode(self: Any, task: str) -> str:
    return await self.mode_decision.decide(task)


def _set_planning_request(self: Any, *, output_path: str | None = None, output_format: str | None = None) -> None:
    self.mode_decision._set_planning_request(output_path=output_path, output_format=output_format)


def _extract_planning_request(self: Any, task: str) -> tuple[str | None, str | None] | None:
    return self.mode_decision._extract_planning_request(task)


def cancel(self: Any, source: str = "manual") -> None:
    source_text = str(source or "manual").strip() or "manual"
    self._cancel_requested = True
    self.note_task_shutdown("cancel_requested")
    self._cancel_source = source_text
    self.approvals.reject_pending_shell_approvals()
    self.approvals.reject_pending_sudo_password_prompts()
    if self._active_dispatch_task and not self._active_dispatch_task.done():
        self._active_dispatch_task.cancel()
    from ..logging_utils import log_kv

    log_kv(self.log, logging.INFO, "harness_cancel_requested", source=source_text)


def note_task_shutdown(self: Any, reason: str) -> None:
    self._pending_task_shutdown_reason = str(reason or "").strip()


async def teardown(self: Any) -> None:
    existing_task = getattr(self, "_teardown_task", None)
    current_task = asyncio.current_task()

    if existing_task is not None and not existing_task.done():
        if existing_task is current_task:
            return
        await asyncio.shield(existing_task)
        return

    task = asyncio.create_task(_run_teardown(self))
    self._teardown_task = task
    try:
        await asyncio.shield(task)
    finally:
        if getattr(self, "_teardown_task", None) is task:
            self._teardown_task = None


async def request_shell_approval(
    self: Any,
    *,
    command: str,
    cwd: str,
    timeout_sec: int,
    proof_bundle: dict[str, Any] | None = None,
) -> bool:
    return await self.approvals.request_shell_approval(
        command=command,
        cwd=cwd,
        timeout_sec=timeout_sec,
        proof_bundle=proof_bundle,
    )


async def request_sudo_password(
    self: Any,
    *,
    command: str,
    prompt_text: str,
    timeout_sec: int = 300,
) -> str | None:
    return await self.approvals.request_sudo_password(command=command, prompt_text=prompt_text, timeout_sec=timeout_sec)


def get_sudo_password(self: Any, *, command: str = "") -> str | None:
    """Return a configured sudo password if available, otherwise None.

    The harness uses this as the non-TUI password source when a running
    shell command prompts for sudo.  It intentionally does not prompt the
    user inline; callers that need interactive prompting should use
    request_sudo_password instead.
    """
    store = getattr(self, "credential_store", None)
    if store is not None:
        return store.get_sudo_password()
    password = getattr(self.config, "sudo_password", None) or getattr(self, "sudo_password", None)
    if isinstance(password, str) and password.strip():
        return password.strip()
    return None


def resolve_shell_approval(self: Any, approval_id: str, approved: bool) -> None:
    self.approvals.resolve_shell_approval(approval_id, approved)


def resolve_sudo_password(self: Any, prompt_id: str, password: str | None) -> None:
    self.approvals.resolve_sudo_password(prompt_id, password)


def _reject_pending_shell_approvals(self: Any) -> None:
    self.approvals.reject_pending_shell_approvals()


def _reject_pending_sudo_password_prompts(self: Any) -> None:
    self.approvals.reject_pending_sudo_password_prompts()


def _reject_shell_approval(self: Any, approval_id: str) -> None:
    self.approvals.resolve_shell_approval(approval_id, False)


async def run_task_with_events(
    self: Any,
    task: str,
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    from ..graph.runtime import LoopGraphRuntime
    from ..graph.runtime_staged import StagedExecutionRuntime
    from datetime import datetime, timezone

    _begin_run_guard(self)
    try:
        redirected = await _maybe_resume_pending_interrupt(self, task, event_handler=event_handler)
        if redirected is not None:
            return redirected

        # Track task receipt time (Fix 2)
        state = getattr(self, "state", None)
        if state is not None:
            state.task_received_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            touch = getattr(state, "touch", None)
            if callable(touch):
                touch()

        log_kv(
            self.log if hasattr(self, "log") else logging.getLogger("smallctl.harness"),
            logging.INFO,
            "task_received",
            mode="loop",
            task_length=len(task),
            session_id=str(getattr(state, "thread_id", "") or "").strip() if state else "",
        )

        self.event_handler = event_handler
        if _should_use_staged_execution_runtime(self):
            runtime = StagedExecutionRuntime.from_harness(
                self,
                event_handler=event_handler,
            )
            return await runtime.run(task)
        runtime = LoopGraphRuntime.from_harness(
            self,
            event_handler=event_handler,
        )
        return await runtime.run(task)
    finally:
        _end_run_guard(self)


async def _resume_task_with_events_impl(
    self: Any,
    human_input: str,
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    from ..graph.runtime import LoopGraphRuntime, PlanningGraphRuntime
    from ..graph.runtime_staged import StagedExecutionRuntime

    self.event_handler = event_handler
    interrupt = self.get_pending_interrupt() or {}
    if str(interrupt.get("kind") or "") == "plan_execute_approval":
        runtime = PlanningGraphRuntime.from_harness(
            self,
            event_handler=event_handler,
        )
        return await runtime.resume(human_input)
    if _should_resume_staged_runtime(self, interrupt):
        runtime = StagedExecutionRuntime.from_harness(
            self,
            event_handler=event_handler,
        )
        return await runtime.resume(human_input)
    runtime = LoopGraphRuntime.from_harness(
        self,
        event_handler=event_handler,
    )
    return await runtime.resume(human_input)


async def resume_task_with_events(
    self: Any,
    human_input: str,
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    _begin_run_guard(self)
    try:
        return await _resume_task_with_events_impl(self, human_input, event_handler)
    finally:
        _end_run_guard(self)


def has_pending_interrupt(self: Any) -> bool:
    pending = getattr(self.state, "pending_interrupt", None)
    if isinstance(pending, dict) and bool(pending):
        if _approved_plan_matches_interrupt(self, pending):
            return False
        return True
    return _planner_interrupt_payload(self) is not None


def get_pending_interrupt(self: Any) -> dict[str, Any] | None:
    pending = getattr(self.state, "pending_interrupt", None)
    if isinstance(pending, dict) and pending:
        if _approved_plan_matches_interrupt(self, pending):
            return None
        return dict(pending)
    return _planner_interrupt_payload(self)


async def _maybe_resume_pending_interrupt(
    self: Any,
    task: str,
    *,
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> dict[str, Any] | None:
    get_pending_interrupt = getattr(self, "get_pending_interrupt", None)
    if not callable(get_pending_interrupt):
        return None
    interrupt = get_pending_interrupt() or {}
    if not is_interrupt_response(interrupt, task):
        return None
    # The caller already holds the run guard, so the redirect must not
    # re-enter it. Harnesses (or test doubles) that override the facade
    # resume entrypoint still get their override honored.
    resume = getattr(self, "resume_task_with_events", None)
    if callable(resume) and getattr(resume, "__func__", resume) is not resume_task_with_events:
        return await resume(task, event_handler=event_handler)
    return await _resume_task_with_events_impl(self, task, event_handler)


def _staged_execution_enabled(self: Any) -> bool:
    return bool(getattr(getattr(self, "config", None), "staged_execution_enabled", False))


def _approved_plan_available(self: Any) -> bool:
    plan = getattr(self.state, "active_plan", None) or getattr(self.state, "draft_plan", None)
    return bool(plan is not None and getattr(plan, "approved", False))


def _should_use_staged_execution_runtime(self: Any) -> bool:
    return bool(
        _staged_execution_enabled(self)
        and _approved_plan_available(self)
        and not getattr(self.state, "planning_mode_enabled", False)
    )


def _should_resume_staged_runtime(self: Any, interrupt: dict[str, Any]) -> bool:
    interrupt_kind = str(interrupt.get("kind") or "").strip()
    if interrupt_kind == "staged_step_blocked":
        return True
    return bool(
        _staged_execution_enabled(self)
        and getattr(self.state, "plan_execution_mode", False)
        and _approved_plan_available(self)
    )


def bind_runtime_facade(cls: type[Any]) -> None:
    cls.run_chat_with_events = run_chat_with_events
    cls.run_auto_with_events = run_auto_with_events
    cls.set_interactive_shell_approval = set_interactive_shell_approval
    cls.set_shell_approval_session_default = set_shell_approval_session_default
    cls.set_planning_mode = set_planning_mode
    cls.run_task = run_task
    cls.run_subtask = run_subtask
    cls.run_auto = run_auto
    cls.resume_task = resume_task
    cls.restore_graph_state = restore_graph_state
    cls._sync_run_logger_session_id = _sync_run_logger_session_id
    cls._schedule_background_persistence = _schedule_background_persistence
    cls._drain_background_persistence_tasks = _drain_background_persistence_tasks
    cls._autosave_chat_session_state = _autosave_chat_session_state
    cls._approved_plan_matches_interrupt = _approved_plan_matches_interrupt
    cls.decide_run_mode = decide_run_mode
    cls._set_planning_request = _set_planning_request
    cls._extract_planning_request = _extract_planning_request
    cls.cancel = cancel
    cls.note_task_shutdown = note_task_shutdown
    cls.teardown = teardown
    cls.request_shell_approval = request_shell_approval
    cls.request_sudo_password = request_sudo_password
    cls.get_sudo_password = get_sudo_password
    cls.resolve_shell_approval = resolve_shell_approval
    cls.resolve_sudo_password = resolve_sudo_password
    cls._reject_pending_shell_approvals = _reject_pending_shell_approvals
    cls._reject_pending_sudo_password_prompts = _reject_pending_sudo_password_prompts
    cls._reject_shell_approval = _reject_shell_approval
    cls.run_task_with_events = run_task_with_events
    cls.resume_task_with_events = resume_task_with_events
    cls.has_pending_interrupt = has_pending_interrupt
    cls.get_pending_interrupt = get_pending_interrupt
