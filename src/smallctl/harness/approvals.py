from __future__ import annotations

import asyncio
import uuid
import logging
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from ..models.events import UIEvent, UIEventType

if TYPE_CHECKING:
    from ..harness import Harness

logger = logging.getLogger("smallctl.harness.approvals")


def _set_future_result_if_pending(future: asyncio.Future[Any], value: Any) -> None:
    if future.done():
        return
    future.set_result(value)


class ApprovalService:
    def __init__(self, harness: Harness):
        self.harness = harness
        self._shell_approval_waiters: dict[str, asyncio.Future[bool]] = {}
        self._sudo_password_waiters: dict[str, asyncio.Future[str | None]] = {}

    async def request_shell_approval(
        self,
        *,
        command: str,
        cwd: str,
        timeout_sec: int,
        proof_bundle: dict[str, Any] | None = None,
    ) -> bool:
        if not self.harness.allow_interactive_shell_approval or getattr(self.harness, "event_handler", None) is None:
            return True
        if self.harness.shell_approval_session_default:
            return True

        approval_id = f"shell-{uuid.uuid4().hex[:10]}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._shell_approval_waiters[approval_id] = future
        event = UIEvent(
            event_type=UIEventType.ALERT,
            content="Approve shell command?",
            data={
                "ui_kind": "approve_prompt",
                "approval_id": approval_id,
                "command": command,
                "cwd": cwd,
                "timeout_sec": timeout_sec,
                "proof_bundle": proof_bundle or {},
                "status_activity": "awaiting approval...",
            },
        )
        try:
            await self.harness._emit(self.harness.event_handler, event)
            return await future
        except Exception:
            self._reject_shell_approval(approval_id)
            raise
        finally:
            self._shell_approval_waiters.pop(approval_id, None)

    def resolve_shell_approval(self, approval_id: str, approved: bool) -> None:
        future = self._shell_approval_waiters.get(approval_id)
        self._resolve_future(future, bool(approved))

    def reject_pending_shell_approvals(self) -> None:
        for approval_id in list(self._shell_approval_waiters.keys()):
            self._reject_shell_approval(approval_id)

    def _reject_shell_approval(self, approval_id: str) -> None:
        future = self._shell_approval_waiters.get(approval_id)
        self._resolve_future(future, False)

    async def request_sudo_password(
        self,
        *,
        command: str,
        prompt_text: str,
    ) -> str | None:
        if not self.harness.allow_interactive_shell_approval or getattr(self.harness, "event_handler", None) is None:
            return None

        prompt_id = f"sudo-{uuid.uuid4().hex[:10]}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str | None] = loop.create_future()
        self._sudo_password_waiters[prompt_id] = future
        event = UIEvent(
            event_type=UIEventType.ALERT,
            content="Sudo password required.",
            data={
                "ui_kind": "sudo_password_prompt",
                "prompt_id": prompt_id,
                "command": command,
                "prompt_text": prompt_text,
                "status_activity": "awaiting sudo password...",
            },
        )
        try:
            await self.harness._emit(self.harness.event_handler, event)
            return await future
        except Exception:
            self._reject_sudo_password_prompt(prompt_id)
            raise
        finally:
            self._sudo_password_waiters.pop(prompt_id, None)

    def resolve_sudo_password(self, prompt_id: str, password: str | None) -> None:
        future = self._sudo_password_waiters.get(prompt_id)
        self._resolve_future(future, password)

    def reject_pending_sudo_password_prompts(self) -> None:
        for prompt_id in list(self._sudo_password_waiters.keys()):
            self._reject_sudo_password_prompt(prompt_id)

    def _reject_sudo_password_prompt(self, prompt_id: str) -> None:
        future = self._sudo_password_waiters.get(prompt_id)
        self._resolve_future(future, None)

    @staticmethod
    def _resolve_future(future: asyncio.Future[Any] | None, value: Any) -> None:
        if future is None or future.done():
            return
        loop = future.get_loop()
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            _set_future_result_if_pending(future, value)
            return
        loop.call_soon_threadsafe(_set_future_result_if_pending, future, value)
