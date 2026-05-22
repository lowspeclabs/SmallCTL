from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.fama.runtime import observe_tool_result
from smallctl.fama.signals import ActiveMitigation
from smallctl.fama.state import activate_mitigations, active_mitigation_names
from smallctl.fama.tool_policy import apply_fama_tool_exposure, enforce_fama_tool_call
from smallctl.models.tool_result import ToolEnvelope
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
        [_schema("task_complete"), _schema("task_fail")],
        state=state,
        mode="loop",
        config=_Config(),
    )

    names = [entry["function"]["name"] for entry in schemas]
    assert names == ["task_fail"]


def test_fama_done_gate_hides_task_fail_when_repair_tools_are_available() -> None:
    state = LoopState()
    _activate_done_gate(state)

    schemas = apply_fama_tool_exposure(
        [_schema("task_complete"), _schema("task_fail"), _schema("file_read"), _schema("file_patch")],
        state=state,
        mode="loop",
        config=_Config(),
    )

    names = [entry["function"]["name"] for entry in schemas]
    assert names == ["file_read", "file_patch"]


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


def test_fama_done_gate_block_metadata_includes_fingerprints() -> None:
    state = LoopState()
    state.last_verifier_verdict = {"verdict": "fail", "command": "pytest tests/test_other.py"}
    activate_mitigations(
        state,
        [
            ActiveMitigation(
                name="done_gate",
                reason="verifier verdict fail: pytest tests/test_target.py",
                source_signal="early_stop:0",
                activated_step=0,
                expires_after_step=2,
            )
        ],
        max_active=2,
    )

    blocked = enforce_fama_tool_call(
        "task_complete",
        {"message": "done"},
        state=state,
        mode="loop",
        config=_Config(),
    )

    assert blocked is not None
    assert blocked.metadata["required_fingerprints"] == ["pytest tests/test_target.py"]
    assert blocked.metadata["actual_fingerprint"] == "pytest tests/test_other.py"
    assert blocked.metadata["fingerprint_match"] is False


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


def test_ssh_auth_failure_releases_done_gate() -> None:
    state = LoopState()
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
    state.task_mode = "remote_execute"
    state.active_intent = "requested_ssh_exec"
    state.last_verifier_verdict = {"verdict": "fail", "command": "pytest"}

    result = ToolEnvelope(
        success=False,
        error="Permission denied (publickey,password).",
        metadata={
            "tool_name": "ssh_exec",
            "output": {
                "stdout": "",
                "stderr": "Permission denied (publickey,password).",
                "exit_code": 255,
            },
        },
    )

    def _runlog(*args, **kwargs):
        pass

    harness = SimpleNamespace(
        state=state,
        config=_Config(),
        _runlog=_runlog,
    )

    asyncio.run(
        observe_tool_result(
            SimpleNamespace(harness=harness),
            tool_name="ssh_exec",
            result=result,
            operation_id="op-ssh-auth-fail",
        )
    )

    assert "done_gate" not in active_mitigation_names(state)
    assert state.task_mode == "local_execute"
    assert state.active_intent == "general_task"

    schemas = apply_fama_tool_exposure(
        [_schema("task_complete"), _schema("task_fail"), _schema("shell_exec")],
        state=state,
        mode="loop",
        config=_Config(),
    )
    names = [entry["function"]["name"] for entry in schemas]
    assert "task_complete" in names
    assert "task_fail" in names
