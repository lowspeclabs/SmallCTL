from __future__ import annotations

from smallctl.fama.detectors import detect_early_stop_from_result, detect_verifier_failure_from_result
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
    state.run_brief.original_task = "rca why the fog install is not working"
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
