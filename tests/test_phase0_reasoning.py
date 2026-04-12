from __future__ import annotations

import asyncio

from smallctl.context import ContextPolicy, PromptAssembler
from smallctl.harness.approvals import ApprovalService
from smallctl.models.conversation import ConversationMessage
from smallctl.state import LoopState


def test_prompt_assembler_build_messages_preserves_core_state() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Read README.md and summarize it"
    state.working_memory.current_goal = state.run_brief.original_task
    state.recent_messages = [
        ConversationMessage(role="assistant", content="I will inspect README.md first."),
    ]

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    assert assembly.messages[0]["role"] == "system"
    assert "SYSTEM PROMPT" in assembly.messages[0]["content"]
    assert "Read README.md and summarize it" in assembly.messages[0]["content"]
    assert any(message["role"] == "user" and message["content"] == state.run_brief.original_task for message in assembly.messages)


def test_shell_approval_request_emits_payload_and_resolves() -> None:
    events: list[object] = []

    class _FakeHarness:
        allow_interactive_shell_approval = True
        shell_approval_session_default = False
        event_handler = object()

        async def _emit(self, handler: object, event: object) -> None:
            del handler
            events.append(event)

    async def _run() -> bool:
        harness = _FakeHarness()
        service = ApprovalService(harness)
        approval_task = asyncio.create_task(
            service.request_shell_approval(command="pwd", cwd="/tmp", timeout_sec=12)
        )
        await asyncio.sleep(0)
        assert events
        approval_event = events[0]
        approval_id = str(getattr(approval_event, "data", {}).get("approval_id") or "")
        assert approval_id.startswith("shell-")
        assert getattr(approval_event, "data", {}).get("command") == "pwd"
        assert getattr(approval_event, "data", {}).get("cwd") == "/tmp"
        service.resolve_shell_approval(approval_id, True)
        return await approval_task

    assert asyncio.run(_run()) is True
