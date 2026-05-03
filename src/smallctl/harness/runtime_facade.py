from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from ..interrupt_replies import is_interrupt_response
from ..models.events import UIEvent


async def _run_sync_persistence_task(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    return await asyncio.to_thread(func, *args, **kwargs)


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
    state: Any,
    model: str,
) -> None:
    _persist_chat_session_state_sync(
        cwd=cwd,
        thread_id=thread_id,
        state_payload=state.to_dict(),
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


async def _run_teardown(self: Any) -> None:
    shutdown_reason = str(getattr(self, "_pending_task_shutdown_reason", "") or "").strip()
    if shutdown_reason:
        self._finalize_task_scope(
            terminal_event="task_interrupted",
            status="interrupted",
            reason=shutdown_reason,
        )
        self._pending_task_shutdown_reason = ""

    self.event_handler = None
    autosave = getattr(self, "_autosave_chat_session_state", None)
    if callable(autosave):
        try:
            autosave()
        except Exception:
            pass
    await _drain_background_persistence_tasks(self)
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

    redirected = await _maybe_resume_pending_interrupt(self, task, event_handler=event_handler)
    if redirected is not None:
        return redirected

    self.event_handler = event_handler
    runtime = ChatGraphRuntime.from_harness(
        self,
        event_handler=event_handler,
    )
    return await runtime.run(task)


async def run_auto_with_events(
    self: Any,
    task: str,
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    from ..graph.runtime import AutoGraphRuntime

    redirected = await _maybe_resume_pending_interrupt(self, task, event_handler=event_handler)
    if redirected is not None:
        return redirected

    self.event_handler = event_handler
    runtime = AutoGraphRuntime.from_harness(
        self,
        event_handler=event_handler,
    )
    return await runtime.run(task)


def set_interactive_shell_approval(self: Any, enabled: bool) -> None:
    self.allow_interactive_shell_approval = bool(enabled)
    self._harness_kwargs["allow_interactive_shell_approval"] = self.allow_interactive_shell_approval


def set_shell_approval_session_default(self: Any, enabled: bool) -> None:
    self.shell_approval_session_default = bool(enabled)
    self._harness_kwargs["shell_approval_session_default"] = self.shell_approval_session_default


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
    try:
        client = getattr(self, "client", None)
        _schedule_background_persistence(
            self,
            _persist_chat_session_state_from_runtime_state_sync,
            cwd,
            thread_id,
            state,
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


def cancel(self: Any) -> None:
    self._cancel_requested = True
    self.note_task_shutdown("cancel_requested")
    self.approvals.reject_pending_shell_approvals()
    self.approvals.reject_pending_sudo_password_prompts()
    if self._active_dispatch_task and not self._active_dispatch_task.done():
        self._active_dispatch_task.cancel()
    from ..logging_utils import log_kv

    log_kv(self.log, logging.INFO, "harness_cancel_requested")
    asyncio.create_task(self.teardown())


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
) -> str | None:
    return await self.approvals.request_sudo_password(command=command, prompt_text=prompt_text)


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

    redirected = await _maybe_resume_pending_interrupt(self, task, event_handler=event_handler)
    if redirected is not None:
        return redirected

    self.event_handler = event_handler
    runtime = LoopGraphRuntime.from_harness(
        self,
        event_handler=event_handler,
    )
    return await runtime.run(task)


async def resume_task_with_events(
    self: Any,
    human_input: str,
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    from ..graph.runtime import LoopGraphRuntime, PlanningGraphRuntime

    self.event_handler = event_handler
    interrupt = self.get_pending_interrupt() or {}
    if str(interrupt.get("kind") or "") == "plan_execute_approval":
        runtime = PlanningGraphRuntime.from_harness(
            self,
            event_handler=event_handler,
        )
        return await runtime.resume(human_input)
    runtime = LoopGraphRuntime.from_harness(
        self,
        event_handler=event_handler,
    )
    return await runtime.resume(human_input)


def has_pending_interrupt(self: Any) -> bool:
    return isinstance(self.state.pending_interrupt, dict) and bool(self.state.pending_interrupt)


def get_pending_interrupt(self: Any) -> dict[str, Any] | None:
    if not isinstance(self.state.pending_interrupt, dict):
        return None
    return dict(self.state.pending_interrupt)


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
    return await self.resume_task_with_events(task, event_handler=event_handler)


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
    cls.decide_run_mode = decide_run_mode
    cls._set_planning_request = _set_planning_request
    cls._extract_planning_request = _extract_planning_request
    cls.cancel = cancel
    cls.note_task_shutdown = note_task_shutdown
    cls.teardown = teardown
    cls.request_shell_approval = request_shell_approval
    cls.request_sudo_password = request_sudo_password
    cls.resolve_shell_approval = resolve_shell_approval
    cls.resolve_sudo_password = resolve_sudo_password
    cls._reject_pending_shell_approvals = _reject_pending_shell_approvals
    cls._reject_pending_sudo_password_prompts = _reject_pending_sudo_password_prompts
    cls._reject_shell_approval = _reject_shell_approval
    cls.run_task_with_events = run_task_with_events
    cls.resume_task_with_events = resume_task_with_events
    cls.has_pending_interrupt = has_pending_interrupt
    cls.get_pending_interrupt = get_pending_interrupt
