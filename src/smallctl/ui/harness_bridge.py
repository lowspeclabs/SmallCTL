from __future__ import annotations

import asyncio
from concurrent.futures import Future as ConcurrentFuture
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

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._closed = False
        self._ready.clear()
        thread = threading.Thread(target=self._thread_main, name=self._thread_name, daemon=True)
        self._thread = thread
        thread.start()
        self._ready.wait()

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
            self.harness._sync_run_logger_session_id()
            return {
                "thread_id": str(getattr(self.harness.state, "thread_id", "") or ""),
                "snapshot": self.harness.build_status_snapshot(),
                "recent_messages": _serialize_recent_messages(self.harness.state),
            }

        return await self._submit_callable(_replace)

    def set_shell_approval_session_default(self, enabled: bool) -> None:
        self._call_soon(self.harness.set_shell_approval_session_default, enabled)

    def cancel(self) -> None:
        self._call_soon(self.harness.cancel)

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
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    def _forward_event(self, event: UIEvent) -> None:
        self._post_ui_event(event)

    async def _submit_coroutine(self, coro: Any) -> Any:
        loop = self._require_loop()
        future: ConcurrentFuture[Any] = asyncio.run_coroutine_threadsafe(coro, loop)
        return await asyncio.wrap_future(future)

    async def _submit_callable(self, callback: Callable[..., Any], *args: Any) -> Any:
        async def _invoke() -> Any:
            return callback(*args)

        return await self._submit_coroutine(_invoke())

    def _call_soon(self, callback: Callable[..., Any], *args: Any) -> None:
        loop = self._require_loop()
        loop.call_soon_threadsafe(callback, *args)

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        self.start()
        loop = self._loop
        if loop is None:
            raise RuntimeError("Harness bridge loop is not available.")
        return loop


def _serialize_recent_messages(state: Any) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    for message in list(getattr(state, "recent_messages", []) or []):
        role = str(getattr(message, "role", "") or "").strip().lower()
        content = str(getattr(message, "content", "") or "").strip()
        if not role or not content:
            continue
        serialized.append({"role": role, "content": content})
    return serialized
