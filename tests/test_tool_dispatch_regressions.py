from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.harness.tool_dispatch import dispatch_tool_call
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


def _make_fake_harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        config=SimpleNamespace(graph_dispatch_tools_timeout_sec=300),
        registry=SimpleNamespace(
            names=lambda: {"file_write", "file_patch", "ssh_file_write", "ssh_file_patch", "shell_exec", "ssh_exec", "ssh_file_read"},
            get=lambda name: None,
        ),
        dispatcher=SimpleNamespace(
            dispatch=lambda name, args: ToolEnvelope(success=True, output={"tool": name, "args": args})
        ),
        _current_user_task=lambda: "test task",
        _runlog=lambda *args, **kwargs: None,
        artifact_store=SimpleNamespace(
            compact_tool_message=lambda artifact, result, **kwargs: str(result.output or result.error or "")
        ),
        context_policy=SimpleNamespace(tool_result_inline_token_limit=200),
    )


async def _async_timeout_override_caps_at_harness_limit() -> None:
    state = LoopState()
    harness = _make_fake_harness(state)

    async def _dispatch(name, args):
        return ToolEnvelope(success=True, output={"tool": name, "args": args})

    harness.dispatcher.dispatch = _dispatch
    result = await dispatch_tool_call(harness, "shell_exec", {"command": "sleep 1", "timeout_sec": 600})
    assert result.metadata["effective_timeout_sec"] == 300
    assert result.metadata["timeout_override_reason"] == "capped by harness graph_dispatch_tools_timeout_sec (300s)"


def test_timeout_override_caps_at_harness_limit() -> None:
    asyncio.run(_async_timeout_override_caps_at_harness_limit())


async def _async_timeout_override_no_override_when_within_limit() -> None:
    state = LoopState()
    harness = _make_fake_harness(state)

    async def _dispatch(name, args):
        return ToolEnvelope(success=True, output={"tool": name, "args": args})

    harness.dispatcher.dispatch = _dispatch
    result = await dispatch_tool_call(harness, "shell_exec", {"command": "sleep 1", "timeout_sec": 60})
    assert "effective_timeout_sec" not in result.metadata


def test_timeout_override_no_override_when_within_limit() -> None:
    asyncio.run(_async_timeout_override_no_override_when_within_limit())


def test_phase_reset_on_continue_after_verifier_failure() -> None:
    state = LoopState()
    state.current_phase = "repair"
    state.last_failure_class = "verifier_failed"
    state.scratchpad["_last_task_status"] = "cancelled_after_verifier_failure"
    state.scratchpad["_task_transaction"] = {"turn_type": "CONTINUE"}

    last_status = str(state.scratchpad.get("_last_task_status") or "").strip()
    if last_status in {"cancelled_after_verifier_failure", "tool_dispatch_cancelled"}:
        if str(state.current_phase or "").strip().lower() == "repair":
            state.current_phase = "execute"
            state.last_failure_class = ""

    assert state.current_phase == "execute"
    assert state.last_failure_class == ""


def test_phase_not_reset_on_manual_task_fail() -> None:
    state = LoopState()
    state.current_phase = "repair"
    state.last_failure_class = "verifier_failed"
    state.scratchpad["_last_task_status"] = "task_fail"

    last_status = str(state.scratchpad.get("_last_task_status") or "").strip()
    if last_status in {"cancelled_after_verifier_failure", "tool_dispatch_cancelled"}:
        if str(state.current_phase or "").strip().lower() == "repair":
            state.current_phase = "execute"
            state.last_failure_class = ""

    assert state.current_phase == "repair"
    assert state.last_failure_class == "verifier_failed"
