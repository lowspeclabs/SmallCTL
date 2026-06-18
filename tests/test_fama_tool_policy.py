from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.fama.fingerprints import install_verifier_passes_objective
from smallctl.fama.runtime import observe_tool_result
from smallctl.fama.signals import ActiveMitigation
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
