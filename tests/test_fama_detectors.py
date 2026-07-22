from __future__ import annotations

from smallctl.fama.detectors import (
    detect_early_stop_from_result,
    detect_generic_stuck_loop,
    detect_objective_verifier_mismatch,
    detect_ssh_host_key_verification_failure_from_result,
    detect_verifier_failure_from_result,
    detect_wrong_path,
)
from smallctl.fama.signals import FamaFailureKind
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


def test_detect_early_stop_from_task_complete_verifier_failure() -> None:
    state = LoopState(step_count=3)
    result = ToolEnvelope(
        success=False,
        error="Cannot complete the task while the latest verifier verdict is still failing.",
        metadata={
            "last_verifier_verdict": {
                "verdict": "fail",
                "command": "pytest tests/test_example.py",
            },
            "acceptance_checklist": [],
        },
    )

    signal = detect_early_stop_from_result(
        state,
        tool_name="task_complete",
        result=result,
        operation_id="op-1",
    )

    assert signal is not None
    assert signal.kind is FamaFailureKind.EARLY_STOP
    assert signal.step == 3
    assert signal.tool_name == "task_complete"
    assert "verifier verdict fail" in signal.evidence


def test_detect_early_stop_ignores_success_and_task_fail() -> None:
    state = LoopState()
    failed = ToolEnvelope(
        success=False,
        error="Cannot complete the task while the latest verifier verdict is still failing.",
        metadata={"last_verifier_verdict": {"verdict": "fail"}},
    )
    success = ToolEnvelope(success=True, output={"status": "complete"})

    assert detect_early_stop_from_result(state, tool_name="task_fail", result=failed) is None
    assert detect_early_stop_from_result(state, tool_name="task_complete", result=success) is None


def test_detect_early_stop_uses_pending_acceptance_metadata() -> None:
    state = LoopState(step_count=1)
    result = ToolEnvelope(
        success=False,
        error="Cannot complete the task until acceptance criteria are satisfied or waived.",
        metadata={"pending_acceptance_criteria": ["run tests"]},
    )

    signal = detect_early_stop_from_result(state, tool_name="task_complete", result=result)

    assert signal is not None
    assert signal.kind is FamaFailureKind.EARLY_STOP


def test_detect_verifier_failure_classifies_zero_discovered_tests() -> None:
    state = LoopState(step_count=2)
    result = ToolEnvelope(
        success=True,
        output={
            "stdout": "1s\n",
            "stderr": "\n----------------------------------------------------------------------\nRan 0 tests in 0.000s\n\nNO TESTS RAN\n",
            "exit_code": 0,
        },
        metadata={
            "last_verifier_verdict": {
                "verdict": "fail",
                "command": "python3 ./temp/uptime_formatter.py 1",
                "key_stdout": "1s\n",
                "key_stderr": "Ran 0 tests in 0.000s\n\nNO TESTS RAN\n",
            }
        },
    )

    signal = detect_verifier_failure_from_result(
        state,
        tool_name="shell_exec",
        result=result,
        operation_id="op-zero-tests",
    )

    assert signal is not None
    assert signal.kind is FamaFailureKind.EARLY_STOP
    assert signal.failure_class == "zero_tests_discovered"
    assert "zero tests discovered" in signal.evidence


def test_detect_verifier_failure_ignores_expected_diagnostic_failure() -> None:
    state = LoopState(step_count=2)
    state.run_brief.original_task = "rca why the fog service is not responding"
    result = ToolEnvelope(
        success=True,
        output={
            "stdout": "curl: (7) Failed to connect to 192.168.1.89 port 8080\n",
            "stderr": "",
            "exit_code": 0,
        },
        metadata={
            "last_verifier_verdict": {
                "verdict": "fail",
                "command": "curl -Is https://192.168.1.89:8080/fog/",
                "key_stdout": "curl: (7) Failed to connect to 192.168.1.89 port 8080",
            }
        },
    )

    signal = detect_verifier_failure_from_result(state, tool_name="ssh_exec", result=result)

    assert signal is None


def test_detect_early_stop_includes_ssh_auth_error_in_evidence() -> None:
    state = LoopState(step_count=1)
    state.task_mode = "remote_execute"
    result = ToolEnvelope(
        success=False,
        error="Permission denied (publickey,password).",
        metadata={
            "last_verifier_verdict": {
                "verdict": "fail",
                "command": "systemctl status docker",
                "key_stdout": "",
                "key_stderr": "Permission denied (publickey,password).",
                "failure_mode": "environment",
                "latest_blocker": {
                    "salient_error": "Permission denied (publickey,password).",
                },
            }
        },
    )

    signal = detect_early_stop_from_result(state, tool_name="task_complete", result=result)

    assert signal is not None
    assert signal.kind is FamaFailureKind.EARLY_STOP
    assert "verifier verdict fail" in signal.evidence
    assert "permission denied" in signal.evidence.lower()


def test_detect_ssh_host_key_failure_from_single_tool_result() -> None:
    state = LoopState(step_count=4)
    result = ToolEnvelope(
        success=False,
        error=(
            "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
            "@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @\n"
            "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
            "Offending RSA key in /home/stephen/.ssh/known_hosts:24\n"
            "  remove with:\n"
            "  ssh-keygen -f '/home/stephen/.ssh/known_hosts' -R '192.168.1.161'\n"
            "Host key verification failed."
        ),
    )

    signal = detect_ssh_host_key_verification_failure_from_result(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.161"},
    )

    assert signal is not None
    assert signal.kind is FamaFailureKind.SSH_HOST_KEY_VERIFICATION
    assert signal.failure_class == "ssh_host_key_verification"
    assert "192.168.1.161" in signal.evidence
    assert "known_hosts" in signal.evidence
    assert "local harness file" in str(signal.next_safe_action)
    assert signal.suggested_mitigations == ["ssh_host_key_recovery_capsule"]


def test_detect_generic_stuck_loop_fires_after_repeated_failures() -> None:
    state = LoopState(step_count=5)
    state.stagnation_counters = {"no_actionable_progress": 3}
    state.tool_history = [
        'shell_exec|{"command": "bad-cmd"}|error:command not found',
        'shell_exec|{"command": "bad-cmd"}|error:command not found',
        'shell_exec|{"command": "bad-cmd"}|error:command not found',
    ]
    state.recent_errors = ["command not found", "command not found", "command not found"]

    signal = detect_generic_stuck_loop(state, threshold=3)

    assert signal is not None
    assert signal.kind is FamaFailureKind.LOOPING
    assert signal.source == "loop_guard"
    assert signal.failure_class == "no_progress"
    assert "no_actionable_progress=3" in signal.evidence
    assert signal.tool_name == "shell_exec"


def test_detect_generic_stuck_loop_ignores_insufficient_failure_evidence() -> None:
    state = LoopState(step_count=2)
    state.stagnation_counters = {"no_actionable_progress": 1}
    state.tool_history = [
        'shell_exec|{"command": "bad-cmd"}|error:command not found',
    ]
    state.recent_errors = ["command not found"]

    assert detect_generic_stuck_loop(state, threshold=3) is None


def test_detect_wrong_path_keeps_remote_path_error_out_of_remote_scope_mitigation() -> None:
    state = LoopState(step_count=2)
    result = ToolEnvelope(
        success=False,
        error="grep: app.py: No such file or directory",
        metadata={"reason": "", "path": "/root/docker-medium-challenge/app.py"},
    )

    signal = detect_wrong_path(state, tool_name="ssh_exec", result=result)

    assert signal is not None
    assert signal.kind is FamaFailureKind.WRONG_PATH
    assert signal.failure_class == "wrong_path"


def test_detect_objective_verifier_mismatch_for_compose_repair_config_grep() -> None:
    state = LoopState(step_count=4)
    state.run_brief.original_task = "Repair a broken Docker Compose application and run the grader."
    state.last_verifier_verdict = {
        "verdict": "pass",
        "command": "cd /root/docker-medium-challenge && grep -n 'DB_HOST' compose.yaml",
    }

    signal = detect_objective_verifier_mismatch(state)

    assert signal is not None
    assert signal.kind is FamaFailureKind.OBJECTIVE_MISMATCH
