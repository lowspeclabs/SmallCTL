from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.harness.tool_dispatch import dispatch_tool_call
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools.base import ToolSpec, build_tool_schema
from smallctl.tools.dispatcher import ToolDispatcher
from smallctl.tools.registry import ToolRegistry


def _make_fake_harness(state: LoopState) -> SimpleNamespace:
    async def _dispatch(name, args):
        return ToolEnvelope(success=True, output={"tool": name, "args": args})

    return SimpleNamespace(
        state=state,
        config=SimpleNamespace(graph_dispatch_tools_timeout_sec=300),
        registry=SimpleNamespace(
            names=lambda: {
                "ast_patch",
                "file_read",
                "file_write",
                "file_patch",
                "ssh_file_write",
                "ssh_file_patch",
                "shell_exec",
                "ssh_exec",
                "ssh_file_read",
            },
            get=lambda name: None,
        ),
        dispatcher=SimpleNamespace(dispatch=_dispatch),
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


async def _async_file_patch_fresh_read_gate_blocks_only_file_patch() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_file_patch_fresh_read_required"] = {
        "target_path": "./foo.py",
        "error_kind": "patch_target_not_found",
        "recovery_count": 2,
    }
    harness = _make_fake_harness(state)

    blocked = await dispatch_tool_call(
        harness,
        "file_patch",
        {"path": "./foo.py", "target_text": "old", "replacement_text": "new"},
    )

    assert blocked.success is False
    assert blocked.status == "recoverable"
    assert blocked.metadata["reason"] == "fresh_file_read_required_before_patch"
    assert blocked.metadata["next_required_tool"]["tool_name"] == "file_read"

    ast_result = await dispatch_tool_call(harness, "ast_patch", {"path": "./foo.py"})
    assert ast_result.success is True
    assert state.scratchpad.get("_file_patch_fresh_read_required") is not None


def test_file_patch_fresh_read_gate_blocks_only_file_patch() -> None:
    asyncio.run(_async_file_patch_fresh_read_gate_blocks_only_file_patch())


async def _async_successful_uncached_file_read_clears_fresh_read_gate() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_file_patch_fresh_read_required"] = {
        "target_path": "./foo.py",
        "error_kind": "patch_target_not_found",
        "recovery_count": 2,
    }
    harness = _make_fake_harness(state)

    result = await dispatch_tool_call(harness, "file_read", {"path": "./foo.py"})

    assert result.success is True
    assert "_file_patch_fresh_read_required" not in state.scratchpad


def test_successful_uncached_file_read_clears_fresh_read_gate() -> None:
    asyncio.run(_async_successful_uncached_file_read_clears_fresh_read_gate())


async def _async_cached_file_read_does_not_clear_fresh_read_gate() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_file_patch_fresh_read_required"] = {
        "target_path": "./foo.py",
        "error_kind": "patch_target_not_found",
        "recovery_count": 2,
    }
    harness = _make_fake_harness(state)

    async def _dispatch(name, args):
        return ToolEnvelope(success=True, metadata={"cache_hit": True})

    harness.dispatcher.dispatch = _dispatch

    result = await dispatch_tool_call(harness, "file_read", {"path": "./foo.py"})

    assert result.success is True
    assert state.scratchpad.get("_file_patch_fresh_read_required") is not None


def test_cached_file_read_does_not_clear_fresh_read_gate() -> None:
    asyncio.run(_async_cached_file_read_does_not_clear_fresh_read_gate())


async def _async_dispatcher_structured_validation_metadata() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="needs_path",
            description="test",
            schema=build_tool_schema(properties={"path": {"type": "string"}}, required=["path"]),
            handler=lambda **kwargs: kwargs,
        )
    )
    dispatcher = ToolDispatcher(registry, phase="execute")

    result = await dispatcher.dispatch("needs_path", {})

    assert result.success is False
    assert result.metadata["validation_error"] == "schema_validation"
    assert result.metadata["validation_issues"] == [
        {
            "path": ["path"],
            "kind": "required",
            "expected": None,
            "actual": None,
            "message": "missing required field path",
        }
    ]


def test_dispatcher_structured_validation_metadata() -> None:
    asyncio.run(_async_dispatcher_structured_validation_metadata())


async def _async_dispatcher_marks_legacy_coercion_on_success() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="read_path",
            description="test",
            schema=build_tool_schema(properties={"path": {"type": "string"}}, required=["path"]),
            handler=lambda **kwargs: {"success": True, "output": kwargs, "error": None, "metadata": {}},
        )
    )
    dispatcher = ToolDispatcher(registry, phase="execute")

    result = await dispatcher.dispatch("read_path", {"path": "a.py", "extra": "ignored"})

    assert result.success is True
    assert result.output == {"path": "a.py"}
    assert result.metadata["legacy_dispatch_coercion"] is True
    assert result.metadata["ignored_arguments"] == ["extra"]


def test_dispatcher_marks_legacy_coercion_on_success() -> None:
    asyncio.run(_async_dispatcher_marks_legacy_coercion_on_success())


async def _async_dispatcher_non_object_shell_args_return_validation_error() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="shell_exec",
            description="run shell command",
            schema=build_tool_schema(properties={"command": {"type": "string"}}, required=["command"]),
            handler=lambda **kwargs: kwargs,
        )
    )
    dispatcher = ToolDispatcher(registry, phase="execute")

    result = await dispatcher.dispatch("shell_exec", "pytest")

    assert result.success is False
    assert result.error == "Tool arguments must be an object."
    assert result.metadata["validation_error"] == "schema_validation"
    assert result.metadata["validation_issues"][0]["kind"] == "type"


def test_dispatcher_non_object_shell_args_return_validation_error() -> None:
    asyncio.run(_async_dispatcher_non_object_shell_args_return_validation_error())


async def _async_dispatcher_non_object_ssh_args_return_validation_error() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="ssh_exec",
            description="run ssh command",
            schema=build_tool_schema(
                properties={"command": {"type": "string"}, "host": {"type": "string"}},
                required=["command", "host"],
            ),
            handler=lambda **kwargs: kwargs,
        )
    )
    dispatcher = ToolDispatcher(registry, phase="execute")

    result = await dispatcher.dispatch("ssh_exec", "whoami")

    assert result.success is False
    assert result.error == "Tool arguments must be an object."
    assert result.metadata["validation_error"] == "schema_validation"
    assert result.metadata["validation_issues"][0]["actual"] == "str"


def test_dispatcher_non_object_ssh_args_return_validation_error() -> None:
    asyncio.run(_async_dispatcher_non_object_ssh_args_return_validation_error())


async def _async_dispatcher_empty_ssh_command_names_ssh_exec() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="ssh_exec",
            description="run ssh command",
            schema=build_tool_schema(
                properties={"command": {"type": "string"}, "host": {"type": "string"}},
                required=["command", "host"],
            ),
            handler=lambda **kwargs: kwargs,
        )
    )
    dispatcher = ToolDispatcher(registry, phase="execute")

    result = await dispatcher.dispatch("ssh_exec", {"host": "192.0.2.10", "command": ""})

    assert result.success is False
    assert result.error == "ssh_exec requires a non-empty command string"
    assert result.metadata["validation_error"] == "empty_command"


def test_dispatcher_empty_ssh_command_names_ssh_exec() -> None:
    asyncio.run(_async_dispatcher_empty_ssh_command_names_ssh_exec())


async def _async_harness_dispatch_non_object_args_return_validation_error() -> None:
    state = LoopState()
    harness = _make_fake_harness(state)

    result = await dispatch_tool_call(harness, "shell_exec", "pytest")

    assert result.success is False
    assert result.error == "Tool arguments must be an object."
    assert result.metadata["tool_name"] == "shell_exec"
    assert result.metadata["validation_error"] == "schema_validation"
    assert result.metadata["validation_issues"][0]["actual"] == "str"


def test_harness_dispatch_non_object_args_return_validation_error() -> None:
    asyncio.run(_async_harness_dispatch_non_object_args_return_validation_error())


def test_verifier_loop_hard_stop_blocks_verifiers_and_task_complete() -> None:
    from smallctl.tools.dispatcher_policy_guards import _verifier_loop_dispatch_block

    state = LoopState()
    state.scratchpad["_verifier_loop_required_action_classes"] = [
        "research",
        "mutation",
        "ask_user",
        "stop_blocked",
    ]
    state.scratchpad["_verifier_loop_rejection_count"] = 3

    assert _verifier_loop_dispatch_block(state, "shell_exec") is not None
    assert _verifier_loop_dispatch_block(state, "ssh_exec") is not None
    assert _verifier_loop_dispatch_block(state, "task_complete") is not None
    assert _verifier_loop_dispatch_block(state, "file_read") is None
    assert _verifier_loop_dispatch_block(state, "file_patch") is None
    assert _verifier_loop_dispatch_block(state, "ask_human") is None
    assert _verifier_loop_dispatch_block(state, "task_fail") is None


def test_verifier_loop_dispatch_block_ignored_when_rejection_count_low() -> None:
    from smallctl.tools.dispatcher_policy_guards import _verifier_loop_dispatch_block

    state = LoopState()
    state.scratchpad["_verifier_loop_required_action_classes"] = [
        "research",
        "mutation",
        "ask_user",
        "stop_blocked",
    ]
    state.scratchpad["_verifier_loop_rejection_count"] = 2

    assert _verifier_loop_dispatch_block(state, "shell_exec") is None
