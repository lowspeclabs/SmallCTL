from __future__ import annotations

import asyncio
from concurrent.futures import Future as ConcurrentFuture
import inspect
import threading
from typing import Any, Callable

from ..models.events import UIEvent


class HarnessBridge:
    def __init__(
        self,
        *,
        harness: Any,
        post_ui_event: Callable[[UIEvent], None],
        thread_name: str = "smallctl-harness",
    ) -> None:
        self.harness = harness
        self._post_ui_event = post_ui_event
        self._thread_name = thread_name
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._closed = False
        self._close_lock = threading.Lock()
        self._inflight_future: ConcurrentFuture[Any] | None = None
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
        return await self._submit_coroutine(
            self.harness.run_auto_with_events(task, self._forward_event)
        )

    async def resume(self, human_input: str) -> dict[str, Any]:
        return await self._submit_coroutine(
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

        future = asyncio.run_coroutine_threadsafe(self.harness.teardown(), loop)
        await asyncio.wrap_future(future)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
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
            if self._heartbeat_task is not None:
                self._heartbeat_task.cancel()
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(asyncio.sleep(0))
            loop.close()

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
        future = self._inflight_future
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
