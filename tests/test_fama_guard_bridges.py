from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.fama.capsules import render_fama_capsules
from smallctl.fama.detectors import (
    detect_remote_local_confusion,
    detect_repeated_tool_loop,
    detect_write_session_stall,
)
from smallctl.fama.runtime import expire_for_turn, observe_tool_result
from smallctl.fama.state import active_mitigation_names
from smallctl.fama.tool_policy import apply_fama_tool_exposure
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState, WriteSession


class _Config:
    fama_enabled = True
    fama_mode = "lite"
    fama_default_ttl_steps = 2
    fama_max_active_mitigations = 4
    fama_signal_window = 8
    fama_done_gate_on_failure = True
    loop_guard_stagnation_threshold = 3


def _schema(name: str) -> dict[str, object]:
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def _names(schemas: list[dict[str, object]]) -> list[str]:
    return [str(item["function"]["name"]) for item in schemas]


def _harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        config=_Config(),
        _runlog=lambda *args, **kwargs: None,
    )


def test_fama_loop_guard_counters_emit_looping_once_per_signature() -> None:
    state = LoopState(step_count=7)
    state.stagnation_counters = {"no_actionable_progress": 3}
    state.tool_history = [
        'artifact_read|{"artifact_id": "A1"}|success',
        'artifact_read|{"artifact_id": "A1"}|success',
        'artifact_read|{"artifact_id": "A1"}|success',
    ]

    signal = detect_repeated_tool_loop(state, threshold=3)
    assert signal is not None
    assert signal.kind.value == "looping"
    assert signal.tool_name == "artifact_read"

    harness = _harness(state)
    expire_for_turn(harness, mode="loop")
    expire_for_turn(harness, mode="loop")

    payload = state.scratchpad["_fama"]
    assert [item["kind"] for item in payload["signals"]] == ["looping"]
    assert "tool_exposure_narrowing" in active_mitigation_names(state)

    schemas = apply_fama_tool_exposure(
        [_schema("artifact_read"), _schema("file_read"), _schema("task_fail")],
        state=state,
        mode="loop",
        config=_Config(),
    )
    assert _names(schemas) == ["file_read", "task_fail"]


def test_fama_remote_local_confusion_hides_local_mutation_and_renders_capsule() -> None:
    state = LoopState(step_count=2)
    result = ToolEnvelope(
        success=False,
        error="Use ssh_file_write, not local file_write.",
        metadata={
            "reason": "remote_path_requires_typed_ssh_file_tool",
            "path": "/var/www/html/index.html",
            "suggested_tool": "ssh_file_write",
        },
    )

    signal = detect_remote_local_confusion(
        state,
        tool_name="file_write",
        result=result,
        operation_id="op-remote",
    )
    assert signal is not None
    assert signal.kind.value == "remote_local_confusion"

    asyncio.run(
        observe_tool_result(
            SimpleNamespace(harness=_harness(state)),
            tool_name="file_write",
            result=result,
            operation_id="op-remote",
        )
    )

    assert {"remote_scope_capsule", "remote_tool_exposure_guard"} <= active_mitigation_names(state)
    schemas = apply_fama_tool_exposure(
        [_schema("file_write"), _schema("shell_exec"), _schema("ssh_file_write"), _schema("task_fail")],
        state=state,
        mode="loop",
        config=_Config(),
    )
    assert _names(schemas) == ["ssh_file_write", "task_fail"]
    assert render_fama_capsules(state, token_budget=180) == [
        "Remote scope is active; use SSH tools for remote paths and verify remotely before finishing."
    ]


def test_fama_write_session_stall_capsule_without_blocking_finalize() -> None:
    state = LoopState(step_count=4)
    state.write_session = WriteSession(
        write_session_id="ws-1",
        write_target_path="src/app.py",
        write_session_mode="chunked_author",
        write_sections_completed=["imports"],
        status="open",
    )
    state.scratchpad["_last_write_session_schema_failure"] = {
        "tool_name": "file_write",
        "write_session_id": "ws-1",
    }

    signal = detect_write_session_stall(state, threshold=3)
    assert signal is not None
    assert signal.kind.value == "write_session_stall"

    expire_for_turn(_harness(state), mode="loop")

    assert {"write_session_recovery_capsule", "outline_only_recovery"} <= active_mitigation_names(state)
    schemas = apply_fama_tool_exposure(
        [_schema("finalize_write_session"), _schema("file_write"), _schema("task_fail")],
        state=state,
        mode="loop",
        config=_Config(),
    )
    assert _names(schemas) == ["finalize_write_session", "file_write", "task_fail"]
    capsule_text = " ".join(render_fama_capsules(state, token_budget=180))
    assert "Resume the active write session" in capsule_text
