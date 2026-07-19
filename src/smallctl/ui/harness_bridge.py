from __future__ import annotations

import asyncio
from concurrent.futures import Future as ConcurrentFuture
import inspect
import logging
import threading
import time
from typing import Any, Callable

from ..models.events import UIEvent


_logger = logging.getLogger("smallctl.ui.harness_bridge")


def _close_process_transports(proc: Any) -> None:
    """Best-effort close of an asyncio subprocess transport and its pipes."""
    if proc is None:
        return
    for attr_name in ("stdin", "stdout", "stderr"):
        pipe = getattr(proc, attr_name, None)
        if pipe is None:
            continue
        close = getattr(pipe, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        for transport_attr in ("transport", "_transport"):
            transport = getattr(pipe, transport_attr, None)
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:
            pass


class HarnessRunBusyError(RuntimeError):
    """Raised when a run is submitted while another run is still active."""


class HarnessBridge:
    def __init__(
        self,
        *,
        harness: Any,
        post_ui_event: Callable[[UIEvent], None],
        thread_name: str = "smallctl-harness",
        shutdown_timeout_sec: float = 10.0,
    ) -> None:
        self.harness = harness
        self._post_ui_event = post_ui_event
        self._thread_name = thread_name
        self._shutdown_timeout_sec = max(0.1, float(shutdown_timeout_sec))
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._closed = False
        self._close_lock = threading.Lock()
        self._inflight_future: ConcurrentFuture[Any] | None = None
        self._run_future: ConcurrentFuture[Any] | None = None
        self._run_state_lock = threading.Lock()
        self._wedged = False
        # Fix 5: Bridge heartbeat for deadlock detection
        self._heartbeat_counter: int = 0
        self._heartbeat_task: asyncio.Task[Any] | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._closed = False
        self._ready.clear()
        thread = threading.Thread(target=self._thread_main, name=self._thread_name, daemon=True)
        self._thread = thread
        thread.start()
        if not self._ready.wait(timeout=30.0):
            raise RuntimeError(
                "HarnessBridge background thread failed to start within 30 seconds. "
                "Check that Harness initialization does not deadlock or throw unhandled exceptions."
            )

    async def run_auto(self, task: str) -> dict[str, Any]:
        return await self._submit_run(
            self.harness.run_auto_with_events(task, self._forward_event)
        )

    async def resume(self, human_input: str) -> dict[str, Any]:
        return await self._submit_run(
            self.harness.resume_task_with_events(human_input, self._forward_event)
        )

    async def switch_model(
        self,
        model: str,
        *,
        activity: str = "",
        api_errors: int | None = None,
    ) -> dict[str, Any]:
        def _switch() -> dict[str, Any]:
            self.harness.switch_model(model)
            return {
                "provider_profile": getattr(self.harness, "provider_profile", "generic"),
                "snapshot": self.harness.build_status_snapshot(
                    activity=activity,
                    api_errors=api_errors,
                ),
            }

        return await self._submit_callable(_switch)

    async def set_planning_mode(self, enabled: bool) -> dict[str, Any]:
        return await self._submit_callable(self.harness.set_planning_mode, enabled)

    async def restore_graph_state(self, thread_id: str | None = None) -> dict[str, Any]:
        def _restore() -> dict[str, Any]:
            restored = bool(self.harness.restore_graph_state(thread_id=thread_id))
            payload: dict[str, Any] = {
                "restored": restored,
                "thread_id": str(getattr(self.harness.state, "thread_id", "") or thread_id or ""),
            }
            if not restored:
                return payload
            payload["has_pending_interrupt"] = bool(self.harness.has_pending_interrupt())
            interrupt = self.harness.get_pending_interrupt()
            if interrupt is not None:
                payload["interrupt"] = interrupt
            payload["snapshot"] = self.harness.build_status_snapshot()
            payload["recent_messages"] = _serialize_recent_messages(self.harness.state)
            return payload

        return await self._submit_callable(_restore)

    async def replace_state_from_payload(self, state_payload: dict[str, Any]) -> dict[str, Any]:
        def _replace() -> dict[str, Any]:
            from ..state import LoopState

            self.harness.state = LoopState.from_dict(state_payload)
            if isinstance(getattr(self.harness.state, "scratchpad", None), dict):
                self.harness.state.scratchpad["_session_restored"] = True
                self.harness.state.scratchpad["_resume_contract"] = {
                    "kind": "chat_session_resume",
                    "thread_id": str(getattr(self.harness.state, "thread_id", "") or ""),
                }
            self.harness._sync_run_logger_session_id()
            return {
                "thread_id": str(getattr(self.harness.state, "thread_id", "") or ""),
                "snapshot": self.harness.build_status_snapshot(),
                "recent_messages": _serialize_recent_messages(self.harness.state),
            }

        return await self._submit_callable(_replace)

    def set_shell_approval_session_default(self, enabled: bool) -> None:
        self._call_soon(self.harness.set_shell_approval_session_default, enabled)

    def cancel(self, source: str = "ui_stop_button") -> None:
        self._call_soon(self._cancel_harness, source)

    def resolve_shell_approval(self, approval_id: str, approved: bool) -> None:
        self._call_soon(self.harness.resolve_shell_approval, approval_id, approved)

    def resolve_sudo_password(self, prompt_id: str, password: str | None) -> None:
        self._call_soon(self.harness.resolve_sudo_password, prompt_id, password)

    async def shutdown(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            return
        if not thread.is_alive():
            self._loop = None
            self._thread = None
            return

        # Abort any in-flight run first so teardown does not overlap it, then
        # give the run a bounded window to finish unwinding on the bridge loop.
        self.abort()
        await self.wait_for_idle(timeout=self._shutdown_timeout_sec)

        future = asyncio.run_coroutine_threadsafe(self.harness.teardown(), loop)
        wrapped = asyncio.wrap_future(future)
        try:
            # asyncio.wait does not cancel the wrapped future on timeout; the
            # concurrent future is cancelled explicitly below instead. This
            # avoids wait_for/wrap_future cancellation recursion wedging the
            # bridge loop.
            done, _pending = await asyncio.wait(
                {wrapped}, timeout=self._shutdown_timeout_sec
            )
            if not done:
                # Teardown wedged: cancel it and force-stop the loop rather
                # than hanging shutdown forever. The thread's background-loop
                # cleanup cancels whatever remains once run_forever returns.
                future.cancel()
                _logger.warning(
                    "Harness teardown exceeded %.1fs; forcing bridge loop stop.",
                    self._shutdown_timeout_sec,
                )
            else:
                wrapped.result()
        finally:
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        if thread.is_alive():
            # The background thread survived loop stop: a task suppressed
            # cancellation. Python cannot safely kill a thread, so shutdown
            # returns bounded here, but the bridge stays wedged and refuses
            # new runs until the thread actually exits.
            self._wedged = True
            _logger.warning(
                "Harness bridge thread %r is STILL ALIVE after shutdown (join "
                "timed out); a run suppressed cancellation. Shutdown is "
                "returning bounded, but the bridge will refuse new runs until "
                "the wedged thread exits.",
                thread.name,
            )
            return
        self._wedged = False
        self._loop = None
        self._thread = None

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        # Fix 5: Start heartbeat task
        self._heartbeat_task = loop.create_task(self._heartbeat_loop())
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(self._shutdown_background_loop())
            loop.close()

    async def _shutdown_background_loop(self) -> None:
        """Cancel remaining tasks, close subprocess transports, and drain callbacks."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        heartbeat = getattr(self, "_heartbeat_task", None)
        if heartbeat is not None:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
        pending = [
            task
            for task in asyncio.all_tasks(loop)
            if task is not asyncio.current_task(loop) and not task.done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        # Close any subprocess transports that are still referenced so their
        # __del__ methods do not try to schedule work after the loop closes.
        harness = getattr(self, "harness", None)
        if harness is not None:
            active_procs = list(getattr(harness, "_active_processes", set()) or [])
            getattr(harness, "_active_processes", set()).clear()
            for proc in active_procs:
                _close_process_transports(proc)
        # Allow transport close callbacks and async generators to drain.
        try:
            await loop.shutdown_asyncgens()
        except Exception:
            pass
        await asyncio.sleep(0)

    async def _heartbeat_loop(self) -> None:
        """Increment a counter every 10 seconds while the bridge is alive."""
        try:
            while True:
                await asyncio.sleep(10.0)
                self._heartbeat_counter += 1
        except asyncio.CancelledError:
            pass

    def get_heartbeat(self) -> int:
        """Return the current heartbeat counter for deadlock detection."""
        return self._heartbeat_counter

    def _forward_event(self, event: UIEvent) -> None:
        self._post_ui_event(event)

    async def _submit_coroutine(self, coro: Any) -> Any:
        loop = self._require_loop()
        future: ConcurrentFuture[Any] = asyncio.run_coroutine_threadsafe(coro, loop)
        self._inflight_future = future
        try:
            return await asyncio.wrap_future(future)
        finally:
            self._inflight_future = None

    async def _submit_run(self, coro: Any) -> Any:
        if self._wedged:
            thread = self._thread
            if thread is not None and thread.is_alive():
                coro.close()
                raise HarnessRunBusyError(
                    "A previous harness run suppressed cancellation and its "
                    "bridge thread is still alive; no new run is admitted "
                    "until that thread exits."
                )
            self._wedged = False
        loop = self._require_loop()
        with self._run_state_lock:
            if self._run_future is not None:
                coro.close()
                raise HarnessRunBusyError(
                    "A harness run is already active; wait for it to finish "
                    "or cancel it before submitting another."
                )
            future: ConcurrentFuture[Any] = asyncio.run_coroutine_threadsafe(
                self._run_and_release(coro), loop
            )
            self._run_future = future
        self._inflight_future = future
        try:
            return await asyncio.wrap_future(future)
        finally:
            self._inflight_future = None

    async def _run_and_release(self, coro: Any) -> Any:
        try:
            return await coro
        finally:
            # A cancelled concurrent future reports done() immediately, while
            # the coroutine may still be unwinding on the bridge loop. Busy
            # state is cleared only here so a new run is admitted solely after
            # the previous run has truly terminated.
            with self._run_state_lock:
                self._run_future = None

    def is_run_active(self) -> bool:
        """Return True while a submitted run has not terminated."""
        with self._run_state_lock:
            return self._run_future is not None

    async def wait_for_idle(self, timeout: float = 5.0) -> bool:
        """Wait for an in-flight run to terminate, bounded by timeout."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if not self.is_run_active():
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.02)

    async def _submit_callable(self, callback: Callable[..., Any], *args: Any) -> Any:
        async def _invoke() -> Any:
            return callback(*args)

        return await self._submit_coroutine(_invoke())

    def _call_soon(self, callback: Callable[..., Any], *args: Any) -> None:
        loop = self._require_loop()
        loop.call_soon_threadsafe(callback, *args)

    def _cancel_harness(self, source: str) -> None:
        cancel = self.harness.cancel
        try:
            signature = inspect.signature(cancel)
        except (TypeError, ValueError):
            try:
                cancel(source)
            except TypeError:
                cancel()
            return

        parameters = list(signature.parameters.values())
        accepts_source = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters) or any(
            param.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            for param in parameters
        )
        if accepts_source:
            cancel(source)
        else:
            cancel()

    def abort(self) -> None:
        futures = [self._inflight_future]
        with self._run_state_lock:
            futures.append(self._run_future)
        for future in futures:
            if future is not None and not future.done():
                future.cancel()

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        self.start()
        loop = self._loop
        if loop is None:
            raise RuntimeError("Harness bridge loop is not available.")
        return loop


def _serialize_recent_messages(state: Any) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    source_messages = list(getattr(state, "transcript_messages", []) or [])
    if not source_messages:
        source_messages = list(getattr(state, "recent_messages", []) or [])
    for message in source_messages:
        role = str(getattr(message, "role", "") or "").strip().lower()
        if not role:
            continue
        content_value = getattr(message, "content", None)
        content = "" if content_value is None else str(content_value)
        metadata = getattr(message, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("tui_visibility") == "tool_scoped_only":
            continue
        if (
            role == "tool"
            and isinstance(metadata, dict)
            and metadata.get("artifact_id")
            and len(content) > 500
        ):
            content = f"[tool result stored in artifact {metadata['artifact_id']}]"
        tool_calls = getattr(message, "tool_calls", None)
        has_tool_calls = isinstance(tool_calls, list) and bool(tool_calls)
        if not content.strip() and not has_tool_calls:
            continue
        payload: dict[str, Any] = {"role": role, "content": content}
        name = getattr(message, "name", None)
        if name:
            payload["name"] = str(name)
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            payload["tool_call_id"] = str(tool_call_id)
        if has_tool_calls:
            payload["tool_calls"] = tool_calls
        if isinstance(metadata, dict) and metadata:
            payload["metadata"] = metadata
        serialized.append(payload)
    return serialized
