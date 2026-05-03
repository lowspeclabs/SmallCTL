from __future__ import annotations

import asyncio

from smallctl.fama.signals import ActiveMitigation
from smallctl.fama.state import activate_mitigations
from smallctl.fama.tool_policy import apply_fama_tool_exposure, enforce_fama_tool_call
from smallctl.state import LoopState
from smallctl.tools.base import ToolSpec
from smallctl.tools.dispatcher import ToolDispatcher
from smallctl.tools.registry import ToolRegistry


class _Config:
    fama_enabled = True
    fama_mode = "lite"
    fama_done_gate_on_failure = True


class _DisabledConfig:
    fama_enabled = False
    fama_mode = "lite"
    fama_done_gate_on_failure = True


def _schema(name: str) -> dict[str, object]:
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def _activate_done_gate(state: LoopState) -> None:
    activate_mitigations(
        state,
        [
            ActiveMitigation(
                name="done_gate",
                reason="verifier failed",
                source_signal="early_stop:0",
                activated_step=0,
                expires_after_step=2,
            )
        ],
        max_active=2,
    )


def test_fama_done_gate_hides_task_complete_from_loop_exposure() -> None:
    state = LoopState()
    _activate_done_gate(state)

    schemas = apply_fama_tool_exposure(
        [_schema("task_complete"), _schema("task_fail"), _schema("file_read")],
        state=state,
        mode="loop",
        config=_Config(),
    )

    names = [entry["function"]["name"] for entry in schemas]
    assert names == ["task_fail", "file_read"]


def test_fama_done_gate_dispatch_blocks_hidden_task_complete() -> None:
    state = LoopState()
    state.last_verifier_verdict = {"verdict": "fail", "command": "pytest"}
    _activate_done_gate(state)

    blocked = enforce_fama_tool_call(
        "task_complete",
        {"message": "done"},
        state=state,
        mode="loop",
        config=_Config(),
    )

    assert blocked is not None
    assert blocked.success is False
    assert blocked.metadata["reason"] == "fama_done_gate"
    assert blocked.metadata["active_mitigation"] == "done_gate"


def test_fama_done_gate_does_not_block_task_fail() -> None:
    state = LoopState()
    _activate_done_gate(state)

    assert enforce_fama_tool_call("task_fail", {}, state=state, mode="loop", config=_Config()) is None


def test_fama_disabled_is_noop() -> None:
    state = LoopState()
    _activate_done_gate(state)
    schemas = [_schema("task_complete"), _schema("task_fail")]

    assert apply_fama_tool_exposure(schemas, state=state, mode="loop", config=_DisabledConfig()) == schemas
    assert (
        enforce_fama_tool_call(
            "task_complete",
            {"message": "done"},
            state=state,
            mode="loop",
            config=_DisabledConfig(),
        )
        is None
    )


def test_fama_done_gate_blocks_direct_tool_dispatcher_task_complete() -> None:
    async def handler() -> dict:
        raise AssertionError("task_complete handler should not be invoked while done_gate is active")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="task_complete",
            description="complete",
            schema={"type": "object", "properties": {"message": {"type": "string"}}, "required": []},
            handler=handler,
        )
    )
    state = LoopState()
    state.last_verifier_verdict = {"verdict": "fail", "command": "pytest"}
    _activate_done_gate(state)

    blocked = asyncio.run(
        ToolDispatcher(registry, state=state, phase="loop").dispatch(
            "task_complete",
            {"message": "done"},
        )
    )

    assert blocked.success is False
    assert blocked.metadata["reason"] == "fama_done_gate"


def test_fama_disabled_config_allows_direct_tool_dispatcher() -> None:
    async def handler(message: str = "") -> dict:
        return {"success": True, "output": message, "error": None, "metadata": {}}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="task_complete",
            description="complete",
            schema={"type": "object", "properties": {"message": {"type": "string"}}, "required": []},
            handler=handler,
        )
    )
    state = LoopState()
    state.scratchpad["_fama_config"] = {"enabled": False, "mode": "lite"}
    state.last_verifier_verdict = {"verdict": "fail", "command": "pytest"}
    _activate_done_gate(state)

    result = asyncio.run(
        ToolDispatcher(registry, state=state, phase="loop").dispatch(
            "task_complete",
            {"message": "done"},
        )
    )

    assert result.success is True
    assert result.output == "done"
