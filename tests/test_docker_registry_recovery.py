from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from smallctl.docker_retry_normalization import (
    docker_retry_family,
    docker_retry_key,
    extract_docker_command_target,
)
from smallctl.graph.state import PendingToolCall, ToolExecutionRecord
from smallctl.graph.tool_execution_recovery_helpers import _maybe_emit_repair_recovery_nudge
from smallctl.graph.tool_loop_guards import _detect_repeated_tool_loop
from smallctl.harness.tool_result_verification import _classify_execution_failure, _store_verifier_verdict
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


def _make_state() -> LoopState:
    state = LoopState(cwd=str(Path.cwd()))
    state.active_tool_profiles = ["core"]
    return state


def test_classify_execution_failure_recognizes_docker_registry_failures() -> None:
    assert _classify_execution_failure(
        "Error response from daemon: manifest for openproject/openproject:latest not found: manifest unknown: manifest unknown"
    ) == "docker_image_not_found"
    assert _classify_execution_failure(
        "Error response from daemon: pull access denied for planeio/plane, repository does not exist or may require 'docker login': denied: requested access to the resource is denied"
    ) == "docker_registry_denied"
    assert _classify_execution_failure(
        "denied: requested access to the resource is denied"
    ) == "docker_registry_denied"


def test_docker_retry_normalization_stabilizes_equivalent_pull_variants() -> None:
    assert extract_docker_command_target(
        "docker pull openproject/openproject:latest"
    ) == ("docker_pull", "openproject/openproject:latest")
    assert extract_docker_command_target(
        "docker pull --platform linux/amd64 openproject/openproject:latest"
    ) == ("docker_pull", "openproject/openproject:latest")
    assert extract_docker_command_target(
        "docker run -d --name openproject -p 8080:8080 admin/openproject:latest"
    ) == ("docker_run_pull_resolution", "admin/openproject:latest")

    assert docker_retry_family(
        "docker pull openproject/openproject:latest"
    ) == docker_retry_family(
        "docker pull --platform linux/amd64 openproject/openproject:latest"
    )
    assert docker_retry_key(
        "docker pull openproject/openproject:latest",
        "docker_image_not_found",
    ) == docker_retry_key(
        "docker pull --platform linux/amd64 openproject/openproject:latest",
        "docker_image_not_found",
    )


def test_docker_specific_repair_nudge_fires_on_second_equivalent_failure_only() -> None:
    state = _make_state()
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    deps = SimpleNamespace(harness=harness, event_handler=None)

    first_command = "docker pull openproject/openproject:latest"
    first_error = (
        "Error response from daemon: manifest for openproject/openproject:latest "
        "not found: manifest unknown: manifest unknown"
    )
    first_result = ToolEnvelope(
        success=False,
        error=first_error,
        metadata={"command": first_command},
    )
    _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=first_result,
        arguments={"host": "192.168.1.63", "command": first_command},
    )
    first_record = ToolExecutionRecord(
        operation_id="op-1",
        tool_name="ssh_exec",
        args={"host": "192.168.1.63", "command": first_command},
        tool_call_id="tool-1",
        result=first_result,
    )

    assert _maybe_emit_repair_recovery_nudge(harness, first_record, deps) is False

    second_command = "docker pull --platform linux/amd64 openproject/openproject:latest"
    second_result = ToolEnvelope(
        success=False,
        error=first_error,
        metadata={"command": second_command},
    )
    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=second_result,
        arguments={"host": "192.168.1.63", "command": second_command},
    )
    second_record = ToolExecutionRecord(
        operation_id="op-2",
        tool_name="ssh_exec",
        args={"host": "192.168.1.63", "command": second_command},
        tool_call_id="tool-2",
        result=second_result,
    )

    assert verdict is not None
    assert verdict["docker_retry_count"] == 2
    assert _maybe_emit_repair_recovery_nudge(harness, second_record, deps) is True
    assert state.recent_messages[-1].metadata["recovery_kind"] == "docker_registry_repair"
    assert "Retrying the same image ref will not help" in state.recent_messages[-1].content

    third_result = ToolEnvelope(
        success=False,
        error=first_error,
        metadata={"command": second_command},
    )
    _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=third_result,
        arguments={"host": "192.168.1.63", "command": second_command},
    )
    third_record = ToolExecutionRecord(
        operation_id="op-3",
        tool_name="ssh_exec",
        args={"host": "192.168.1.63", "command": second_command},
        tool_call_id="tool-3",
        result=third_result,
    )

    assert _maybe_emit_repair_recovery_nudge(harness, third_record, deps) is False


def test_repeated_tool_loop_blocks_exhausted_docker_registry_family() -> None:
    state = _make_state()
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    error_text = (
        "Error response from daemon: manifest for openproject/openproject:latest "
        "not found: manifest unknown: manifest unknown"
    )
    commands = [
        "docker pull openproject/openproject:latest",
        "docker pull --platform linux/amd64 openproject/openproject:latest",
        "docker pull openproject/openproject:latest",
        "docker pull --platform linux/amd64 openproject/openproject:latest",
    ]
    for command in commands:
        _store_verifier_verdict(
            state,
            tool_name="ssh_exec",
            result=ToolEnvelope(success=False, error=error_text, metadata={"command": command}),
            arguments={"host": "192.168.1.63", "command": command},
        )

    pending = PendingToolCall(
        tool_name="ssh_exec",
        args={
            "host": "192.168.1.63",
            "command": "docker pull --platform linux/amd64 openproject/openproject:latest",
        },
        tool_call_id="tool-repeat",
    )

    guard_error = _detect_repeated_tool_loop(harness, pending)

    assert guard_error is not None
    assert "Docker registry/image resolution loop" in guard_error
    assert "openproject/openproject:latest" in guard_error
