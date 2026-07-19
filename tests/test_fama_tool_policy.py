from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.fama.fingerprints import install_verifier_passes_objective
from smallctl.fama.runtime import _is_ssh_transport_impossibility, observe_tool_result
from smallctl.fama.signals import ActiveMitigation, FamaFailureKind, FamaSignal, push_fama_signal
from smallctl.fama.state import activate_mitigations, active_mitigation_names
from smallctl.fama.tool_policy import apply_fama_tool_exposure, enforce_fama_tool_call
from smallctl.harness.tool_visibility import schedule_retry_tool_exposure
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools.base import ToolSpec
from smallctl.tools.dispatcher import ToolDispatcher
from smallctl.tools.registry import ToolRegistry


def test_install_verifier_passes_objective_clears_done_gate_match() -> None:
    state = LoopState()
    state.run_brief.original_task = "ssh_exec to 192.168.1.161 and install webmin"
    state.last_verifier_verdict = {
        "verdict": "pass",
        "command": "dnf list installed webmin",
    }

    assert install_verifier_passes_objective(state) is True


def test_install_verifier_passes_objective_rejects_non_install_task() -> None:
    state = LoopState()
    state.run_brief.original_task = "Read the webmin config file"
    state.last_verifier_verdict = {
        "verdict": "pass",
        "command": "dnf list installed webmin",
    }

    assert install_verifier_passes_objective(state) is False


def test_install_verifier_passes_objective_rejects_weak_file_verifier() -> None:
    state = LoopState()
    state.run_brief.original_task = "ssh_exec to 192.168.1.161 and install webmin"
    state.last_verifier_verdict = {
        "verdict": "pass",
        "command": "cat /tmp/webmin-install.log",
    }

    assert install_verifier_passes_objective(state) is False


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


def test_fama_keeps_task_fail_visible_for_pending_ssh_auth_recovery() -> None:
    state = LoopState()
    _activate_done_gate(state)
    state.scratchpad["_ssh_auth_recovery_state"] = {
        "192.168.1.161::root": {
            "host": "192.168.1.161",
            "user": "root",
            "last_error": "Permission denied (publickey,password).",
        }
    }

    schemas = apply_fama_tool_exposure(
        [_schema("task_complete"), _schema("task_fail"), _schema("ask_human"), _schema("ssh_exec")],
        state=state,
        mode="loop",
        config=_Config(),
    )

    names = [entry["function"]["name"] for entry in schemas]
    assert "task_complete" not in names
    assert "task_fail" in names
    assert "ask_human" in names


def test_recovery_control_tools_can_be_scheduled_after_unavailable_call() -> None:
    state = LoopState()

    assert schedule_retry_tool_exposure(state, mode="loop", tool_name="ask_human") is True
    assert schedule_retry_tool_exposure(state, mode="loop", tool_name="task_fail") is True

    scheduled = state.scratchpad["_retry_tool_exposures"]
    assert [item["tool_name"] for item in scheduled] == ["ask_human", "task_fail"]


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


def test_done_gate_exposes_task_complete_for_read_only_status_inquiry() -> None:
    state = LoopState()
    state.run_brief.original_task = "is docker up and running on the remote host?"
    state.last_verifier_verdict = {
        "verdict": "fail",
        "command": "systemctl status docker",
    }
    _activate_done_gate(state)

    schemas = apply_fama_tool_exposure(
        [_schema("task_complete"), _schema("task_fail")],
        state=state,
        mode="loop",
        config=_Config(),
    )

    names = [entry["function"]["name"] for entry in schemas]
    assert "task_complete" in names


def test_done_gate_hides_task_complete_when_status_inquiry_has_action_intent() -> None:
    state = LoopState()
    state.run_brief.original_task = "start docker and verify it is running on the remote host"
    state.last_verifier_verdict = {
        "verdict": "fail",
        "command": "systemctl status docker",
    }
    _activate_done_gate(state)

    schemas = apply_fama_tool_exposure(
        [_schema("task_complete"), _schema("task_fail")],
        state=state,
        mode="loop",
        config=_Config(),
    )

    names = [entry["function"]["name"] for entry in schemas]
    assert "task_complete" not in names


def test_done_gate_allows_dispatch_for_diagnostic_status_failure() -> None:
    state = LoopState()
    state.run_brief.original_task = "is docker up and running on the remote host?"
    state.last_verifier_verdict = {
        "verdict": "fail",
        "command": "systemctl status docker",
    }
    _activate_done_gate(state)

    result = enforce_fama_tool_call(
        "task_complete",
        {"message": "Docker is not up and running on the remote host."},
        state=state,
        mode="loop",
        config=_Config(),
    )

    assert result is None


def test_done_gate_blocks_dispatch_when_diagnostic_message_omits_failure() -> None:
    state = LoopState()
    state.run_brief.original_task = "is docker up and running on the remote host?"
    state.last_verifier_verdict = {
        "verdict": "fail",
        "command": "systemctl status docker",
    }
    _activate_done_gate(state)

    result = enforce_fama_tool_call(
        "task_complete",
        {"message": "done"},
        state=state,
        mode="loop",
        config=_Config(),
    )

    assert result is not None
    assert result.success is False
    assert result.metadata["reason"] == "fama_done_gate"


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


def test_ssh_auth_failure_preserves_remote_blocker() -> None:
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
    assert state.task_mode == "remote_execute"
    assert state.active_intent == "requested_ssh_exec"
    assert state.scratchpad["_pinned_recovery"]["kind"] == "ssh_auth_blocker"

    schemas = apply_fama_tool_exposure(
        [_schema("task_complete"), _schema("task_fail"), _schema("shell_exec")],
        state=state,
        mode="loop",
        config=_Config(),
    )
    names = [entry["function"]["name"] for entry in schemas]
    assert "task_complete" in names
    assert "task_fail" in names


def test_remote_transport_verifier_failure_does_not_hide_completion_tools() -> None:
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
    state.last_verifier_verdict = {
        "tool": "ssh_exec",
        "verdict": "fail",
        "exit_code": 255,
        "failure_mode": "environment",
        "key_stderr": "ssh: connect to host 192.168.1.16 port 22: No route to host",
    }

    schemas = apply_fama_tool_exposure(
        [_schema("task_complete"), _schema("task_fail"), _schema("shell_exec")],
        state=state,
        mode="loop",
        config=_Config(),
    )
    names = [entry["function"]["name"] for entry in schemas]
    assert "task_complete" in names
    assert "task_fail" in names


def test_interactive_terminal_verifier_failure_exposes_ssh_session_tools() -> None:
    state = LoopState()
    state.last_verifier_verdict = {
        "verdict": "fail",
        "tool": "ssh_exec",
        "failure_mode": "interactive_installer_blocked",
        "key_stderr": "Error opening terminal: unknown.",
        "command": "DEBIAN_FRONTEND=noninteractive bash -c 'curl -sSL https://install.pi-hole.net | bash'",
    }

    schemas = apply_fama_tool_exposure(
        [
            _schema("ssh_exec"),
            _schema("ssh_session_start"),
            _schema("ssh_session_read"),
            _schema("ssh_session_send"),
            _schema("ssh_session_close"),
        ],
        state=state,
        mode="loop",
        config=_Config(),
    )

    names = [entry["function"]["name"] for entry in schemas]
    assert "ssh_exec" in names
    assert "ssh_session_start" in names
    assert "ssh_session_read" in names
    assert "ssh_session_send" in names
    assert "ssh_session_close" in names


def test_ssh_host_key_failure_activates_recovery_capsule_and_runlog() -> None:
    state = LoopState()
    state.scratchpad["_ssh_auth_recovery_state"] = {
        "root@192.168.1.161": {
            "host": "192.168.1.161",
            "user": "root",
            "last_error_class": "host_key_verification",
            "consecutive_count": 2,
            "last_command": "hostname",
            "last_error": "Host key verification failed.",
        }
    }

    runlog: list[tuple[str, str, dict[str, object]]] = []

    def _runlog(event: str, message: str, **data: object) -> None:
        runlog.append((event, message, data))

    harness = SimpleNamespace(
        state=state,
        config=_Config(),
        _runlog=_runlog,
    )
    result = ToolEnvelope(
        success=False,
        error="Host key verification failed.",
        metadata={
            "tool_name": "ssh_exec",
            "failure_mode": "host_key_verification",
            "output": {
                "stdout": "",
                "stderr": "Host key verification failed.",
                "exit_code": 255,
            },
        },
    )

    asyncio.run(
        observe_tool_result(
            SimpleNamespace(harness=harness),
            tool_name="ssh_exec",
            result=result,
            operation_id="op-host-key-1",
        )
    )

    assert "ssh_host_key_recovery_capsule" in active_mitigation_names(state)
    assert any(event == "ssh_host_key_recovery_required" for event, _msg, _data in runlog)
    runlog_entry = next((entry for entry in runlog if entry[0] == "ssh_host_key_recovery_required"), None)
    assert runlog_entry is not None
    assert runlog_entry[2].get("host") == "192.168.1.161"
    assert "ssh-keygen -R 192.168.1.161" in str(runlog_entry[2].get("suggested_command", ""))


def test_permission_denied_failure_does_not_activate_host_key_capsule() -> None:
    state = LoopState()
    state.scratchpad["_ssh_auth_recovery_state"] = {
        "root@192.168.1.161": {
            "host": "192.168.1.161",
            "user": "root",
            "last_error_class": "auth_permission_denied",
            "consecutive_count": 2,
            "last_command": "hostname",
            "last_error": "Permission denied (publickey,password).",
        }
    }

    harness = SimpleNamespace(
        state=state,
        config=_Config(),
        _runlog=lambda *args, **kwargs: None,
    )
    result = ToolEnvelope(
        success=False,
        error="Permission denied (publickey,password).",
        metadata={
            "tool_name": "ssh_exec",
            "failure_mode": "auth_permission_denied",
            "output": {
                "stdout": "",
                "stderr": "Permission denied (publickey,password).",
                "exit_code": 255,
            },
        },
    )

    asyncio.run(
        observe_tool_result(
            SimpleNamespace(harness=harness),
            tool_name="ssh_exec",
            result=result,
            operation_id="op-auth-fail-1",
        )
    )

    assert "ssh_host_key_recovery_capsule" not in active_mitigation_names(state)


def test_interactive_installer_stall_capsule_narrows_tool_exposure() -> None:
    """When the interactive installer stall capsule is active, only ssh_session_send/read/close are exposed."""
    state = LoopState()
    state.tool_history = [
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
    ]
    activate_mitigations(
        state,
        [
            ActiveMitigation(
                name="interactive_installer_stall_capsule",
                reason="stall_detected",
                source_signal="interactive_session_stall:10",
                activated_step=10,
                expires_after_step=15,
            )
        ],
        max_active=5,
    )

    schemas = apply_fama_tool_exposure(
        [
            _schema("ssh_exec"),
            _schema("interactive_run"),
            _schema("ssh_session_start"),
            _schema("ssh_session_read"),
            _schema("ssh_session_send"),
            _schema("ssh_session_close"),
            _schema("shell_exec"),
            _schema("task_complete"),
        ],
        state=state,
        mode="loop",
        config=_Config(),
    )

    names = {entry["function"]["name"] for entry in schemas}
    assert names == {"ssh_session_read", "ssh_session_send", "ssh_session_close"}


def test_interactive_installer_stall_capsule_reasons() -> None:
    """Hidden tools during an interactive installer stall carry the correct reason."""
    from smallctl.fama.tool_policy import fama_hidden_tool_reasons_for_exposure

    state = LoopState()
    activate_mitigations(
        state,
        [
            ActiveMitigation(
                name="interactive_installer_stall_capsule",
                reason="stall_detected",
                source_signal="interactive_session_stall:10",
                activated_step=10,
                expires_after_step=15,
            )
        ],
        max_active=5,
    )

    schemas = [
        _schema("ssh_exec"),
        _schema("interactive_run"),
        _schema("ssh_session_send"),
    ]
    reasons = fama_hidden_tool_reasons_for_exposure(schemas, state=state, mode="loop", config=_Config())
    assert reasons.get("ssh_exec") == ["interactive_installer_stall_narrows_to_send"]
    assert reasons.get("interactive_run") == ["interactive_installer_stall_narrows_to_send"]
    assert "ssh_session_send" not in reasons


def _set_repeated_verifier_rejection(state: LoopState, command: str, failure_mode: str) -> None:
    state.last_verifier_verdict = {
        "verdict": "fail",
        "command": command,
        "failure_mode": failure_mode,
    }
    state.scratchpad["_verifier_rejection_count"] = 5
    state.scratchpad["_last_verifier_rejection"] = {"command": command}
    state.scratchpad["_fama_same_target_streak"] = {"command": command, "streak": 4}


def test_fama_done_gate_hides_task_fail_on_insufficient_verifier() -> None:
    """task_fail must not be the only completion tool when the harness itself is
    uncertain about verifier strength (insufficient_verifier). The model should
    run a corrected/stronger verifier instead of escaping via task_fail."""
    state = LoopState()
    _activate_done_gate(state)
    _set_repeated_verifier_rejection(
        state,
        command="cd /repo && python3 ./temp/text_chunker.py",
        failure_mode="insufficient_verifier",
    )

    schemas = apply_fama_tool_exposure(
        [
            _schema("task_complete"),
            _schema("task_fail"),
            _schema("file_read"),
            _schema("shell_exec"),
        ],
        state=state,
        mode="loop",
        config=_Config(),
    )

    names = {entry["function"]["name"] for entry in schemas}
    assert "task_complete" not in names
    assert "task_fail" not in names
    assert "file_read" in names
    assert "shell_exec" in names


def test_fama_done_gate_exposes_task_fail_after_repeated_genuine_failures() -> None:
    """The dead-end escape hatch should still work for genuine (non-insufficient)
    verifier failures so the model can report an unfixable blocker."""
    state = LoopState()
    _activate_done_gate(state)
    _set_repeated_verifier_rejection(
        state,
        command="cd /repo && python3 ./temp/text_chunker.py",
        failure_mode="test",
    )

    schemas = apply_fama_tool_exposure(
        [
            _schema("task_complete"),
            _schema("task_fail"),
            _schema("file_read"),
            _schema("shell_exec"),
        ],
        state=state,
        mode="loop",
        config=_Config(),
    )

    names = {entry["function"]["name"] for entry in schemas}
    assert "task_complete" not in names
    assert "task_fail" in names
    assert "file_read" in names
    assert "shell_exec" in names


def test_remote_command_exit_255_is_not_ssh_transport_impossibility() -> None:
    result = ToolEnvelope(
        success=False,
        error="remote command failed",
        metadata={
            "tool_name": "ssh_exec",
            "failure_kind": "remote_command",
            "ssh_transport_succeeded": True,
            "exit_code": 255,
            "stderr": "unknown command 'pct'",
        },
    )
    assert _is_ssh_transport_impossibility(result) is False


def test_transport_marker_exit_255_is_ssh_transport_impossibility() -> None:
    result = ToolEnvelope(
        success=False,
        error="",
        metadata={
            "tool_name": "ssh_exec",
            "exit_code": 255,
            "stderr": "ssh: connect to host 192.168.1.161 port 22: Connection refused",
        },
    )
    assert _is_ssh_transport_impossibility(result) is True


def test_exit_255_empty_stderr_is_ssh_transport_impossibility() -> None:
    result = ToolEnvelope(
        success=False,
        error="",
        metadata={
            "tool_name": "ssh_exec",
            "exit_code": 255,
            "stderr": "",
        },
    )
    assert _is_ssh_transport_impossibility(result) is True


def test_repeated_exposure_does_not_advance_same_target_rejection_streak() -> None:
    """Only track_verifier_rejection should mutate the same-target streak."""
    from smallctl.harness.verifier_monitor import track_verifier_rejection

    state = LoopState()
    _activate_done_gate(state)
    command = "cd /repo && python3 ./temp/text_chunker.py"
    state.scratchpad["_fama_same_target_streak"] = {"command": command, "streak": 2}
    state.scratchpad["_verifier_rejection_count"] = 5
    state.last_verifier_verdict = {"verdict": "fail", "command": command, "failure_mode": "test"}

    for _ in range(5):
        apply_fama_tool_exposure(
            [_schema("task_complete"), _schema("task_fail"), _schema("file_read")],
            state=state,
            mode="loop",
            config=_Config(),
        )

    streak = state.scratchpad.get("_fama_same_target_streak")
    assert isinstance(streak, dict)
    assert streak["streak"] == 2

    track_verifier_rejection(state, {"verdict": "fail", "command": command})
    streak = state.scratchpad.get("_fama_same_target_streak")
    assert isinstance(streak, dict)
    assert streak["streak"] == 3


def _activate_tool_exposure_narrowing(state: LoopState) -> None:
    activate_mitigations(
        state,
        [
            ActiveMitigation(
                name="tool_exposure_narrowing",
                reason="loop_guard repeated_action",
                source_signal="looping:0",
                activated_step=0,
                expires_after_step=2,
            )
        ],
        max_active=3,
    )


def _push_looping_signal(state: LoopState, tool_name: str) -> None:
    push_fama_signal(
        state,
        FamaSignal(
            kind=FamaFailureKind.LOOPING,
            severity=2,
            source="loop_guard",
            evidence=f"repeated_tool={tool_name}; no_actionable_progress=4",
            step=1,
            tool_name=tool_name,
        ),
    )


def test_tool_exposure_narrowing_keeps_file_read_exposed_for_read_loop() -> None:
    """A repeated file_read loop must not hide file_read entirely (run aba990a3).

    The loop_guard already blocks the repeated call on the specific path; hiding
    the whole tool pushes the model onto worse substitutes like artifact_read.
    """
    state = LoopState()
    _activate_tool_exposure_narrowing(state)
    _push_looping_signal(state, "file_read")

    schemas = apply_fama_tool_exposure(
        [_schema("file_read"), _schema("artifact_read"), _schema("file_patch")],
        state=state,
        mode="loop",
        config=_Config(),
    )

    names = {entry["function"]["name"] for entry in schemas}
    assert names == {"file_read", "artifact_read", "file_patch"}


def test_tool_exposure_narrowing_still_hides_other_looping_read_tools() -> None:
    """Narrowing still applies to the other read-loop tools (e.g. artifact_read)."""
    state = LoopState()
    _activate_tool_exposure_narrowing(state)
    _push_looping_signal(state, "artifact_read")

    schemas = apply_fama_tool_exposure(
        [_schema("file_read"), _schema("artifact_read"), _schema("file_patch")],
        state=state,
        mode="loop",
        config=_Config(),
    )

    names = {entry["function"]["name"] for entry in schemas}
    assert "artifact_read" not in names
    assert "file_read" in names


def test_repeated_tool_loop_suppression_still_hides_other_looping_tools() -> None:
    """Non-file_read looping tools are still suppressed for the TTL window."""
    from smallctl.harness.tool_visibility import filter_tools_for_runtime_state

    state = LoopState()
    state.scratchpad["_repeated_tool_loop_suppressed_tool"] = "dir_list"
    state.scratchpad["_repeated_tool_loop_suppressed_ttl"] = 2

    filtered = filter_tools_for_runtime_state(
        [_schema("file_read"), _schema("dir_list")],
        state=state,
        mode="loop",
    )

    names = {entry["function"]["name"] for entry in filtered}
    assert "dir_list" not in names
    assert "file_read" in names


def test_swa_stable_tool_exposure_reads_scratchpad_model_and_provider() -> None:
    """SWA detection must fall back to state scratchpad model/provider."""
    from smallctl.fama.tool_policy import _swa_stable_tool_exposure

    state = LoopState()
    state.scratchpad["_model_name"] = "gemma-4-12b"
    state.scratchpad["_provider_profile"] = "llamacpp"

    assert _swa_stable_tool_exposure(state, config=None) is True

    # All tools should remain exposed for SWA models.
    schemas = apply_fama_tool_exposure(
        [_schema("task_complete"), _schema("task_fail"), _schema("file_read")],
        state=state,
        mode="loop",
        config=_Config(),
    )
    names = {entry["function"]["name"] for entry in schemas}
    assert names == {"task_complete", "task_fail", "file_read"}
