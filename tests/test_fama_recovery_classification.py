from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.fama.detectors import (
    detect_empty_write,
    detect_remote_verification_pending,
    detect_repeated_tool_loop,
    detect_test_failure_from_verdict,
    detect_verifier_failure_from_result,
    detect_wrong_path,
)
from smallctl.fama.runtime import observe_tool_result
from smallctl.fama.signals import (
    FamaFailureKind,
    FamaSignal,
    signal_from_dict,
    signal_to_dict,
)
from smallctl.models.tool_result import ToolEnvelope
from smallctl.recovery_schema import Subtask, SubtaskLedger
from smallctl.state import LoopState


class _Config:
    fama_enabled = True
    fama_mode = "lite"
    fama_default_ttl_steps = 2
    fama_max_active_mitigations = 4
    fama_signal_window = 8
    fama_done_gate_on_failure = True
    fama_llm_judge_enabled = False
    fama_llm_judge_min_severity = 3
    loop_guard_stagnation_threshold = 3


class _DisabledConfig(_Config):
    fama_enabled = False


def _harness(state: LoopState, config: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        config=config or _Config(),
        _runlog=lambda *args, **kwargs: None,
    )


def test_fama_signal_round_trips_failure_class_and_next_action() -> None:
    signal = FamaSignal(
        kind=FamaFailureKind.REMOTE_LOCAL_CONFUSION,
        severity=2,
        source="test",
        evidence="path failure marker=no such file or directory; path=missing.py",
        step=4,
        tool_name="file_read",
        failure_class="wrong_path",
        next_safe_action="Verify the path first.",
    )

    payload = signal_to_dict(signal)
    restored = signal_from_dict(payload)

    assert payload["failure_class"] == "wrong_path"
    assert payload["next_safe_action"] == "Verify the path first."
    assert restored is not None
    assert restored.failure_class == "wrong_path"
    assert restored.next_safe_action == "Verify the path first."


def test_wrong_path_detector_sets_narrow_failure_class() -> None:
    state = LoopState(step_count=2)
    result = ToolEnvelope(
        success=False,
        error="FileNotFoundError: No such file or directory: 'src/missing.py'",
        metadata={"path": "src/missing.py"},
    )

    signal = detect_wrong_path(state, tool_name="file_read", result=result, operation_id="op-1")

    assert signal is not None
    assert signal.kind is FamaFailureKind.REMOTE_LOCAL_CONFUSION
    assert signal.failure_class == "wrong_path"
    assert "path=src/missing.py" in signal.evidence


def test_unittest_assertion_not_found_is_not_wrong_path() -> None:
    state = LoopState(step_count=2)
    result = ToolEnvelope(
        success=False,
        output={
            "exit_code": 1,
            "stdout": (
                "FAIL: test_preserve_word_boundaries (__main__.TestChunker.test_preserve_word_boundaries)\n"
                "Traceback (most recent call last):\n"
                "AssertionError: 'This is a sentence' not found in 'This is a sententence'\n"
                "FAILED (failures=1)\n"
            ),
            "stderr": "",
        },
        error="Shell command exited with code 1",
    )

    assert detect_wrong_path(state, tool_name="shell_exec", result=result, operation_id="op-test") is None


def test_ssh_missing_diagnostic_binary_is_not_wrong_path() -> None:
    state = LoopState(step_count=2)
    result = ToolEnvelope(
        success=False,
        output={
            "exit_code": 127,
            "stdout": "",
            "stderr": "bash: line 1: netstat: command not found\n",
        },
        error="bash: line 1: netstat: command not found",
    )

    assert (
        detect_wrong_path(
            state,
            tool_name="ssh_exec",
            result=result,
            operation_id="op-netstat",
        )
        is None
    )


def test_remote_mutation_requires_verification_uses_pending_classification() -> None:
    state = LoopState(step_count=3)
    result = ToolEnvelope(
        success=False,
        error="Remote mutation requires read-back verification before completion.",
        metadata={
            "reason": "remote_mutation_requires_verification",
            "host": "192.0.2.10",
            "pending_paths": ["/var/www/index.html"],
        },
    )

    signal = detect_remote_verification_pending(
        state,
        tool_name="task_complete",
        result=result,
        operation_id="op-verify",
    )

    assert signal is not None
    assert signal.kind is FamaFailureKind.REMOTE_VERIFICATION_PENDING
    assert signal.failure_class == "remote_verification_pending"
    assert signal.failure_class != "wrong_path"
    assert "read-back verification" in signal.next_safe_action
    assert detect_wrong_path(state, tool_name="task_complete", result=result) is None


def test_empty_write_detector_uses_arguments_and_result_markers() -> None:
    state = LoopState(step_count=3)
    result = ToolEnvelope(success=True, output={"bytes_written": 0, "message": "0 bytes written"})

    signal = detect_empty_write(
        state,
        tool_name="file_write",
        result=result,
        arguments={"path": "app.py", "content": ""},
    )

    assert signal is not None
    assert signal.kind is FamaFailureKind.WRITE_SESSION_STALL
    assert signal.failure_class == "empty_write"
    assert "path=app.py" in signal.evidence


def test_verifier_failure_detector_classifies_pytest_as_test_failed() -> None:
    state = LoopState(step_count=5)
    state.last_verifier_verdict = {
        "verdict": "fail",
        "command": "pytest tests/test_app.py",
        "key_stdout": "1 failed",
        "key_stderr": "",
    }
    result = ToolEnvelope(success=False, output={"exit_code": 1, "stdout": "1 failed"})

    signal = detect_verifier_failure_from_result(state, tool_name="shell_exec", result=result)

    assert detect_test_failure_from_verdict(state.last_verifier_verdict) == "test_failed"
    assert signal is not None
    assert signal.failure_class == "test_failed"
    assert signal.next_safe_action


def test_verifier_failure_detector_ignores_stale_state_for_unrelated_tool() -> None:
    state = LoopState(step_count=5)
    state.last_verifier_verdict = {
        "verdict": "fail",
        "command": "pytest tests/test_app.py",
    }

    assert (
        detect_verifier_failure_from_result(
            state,
            tool_name="file_read",
            result=ToolEnvelope(success=True, output="content"),
        )
        is None
    )


def test_repeated_tool_loop_sets_repeated_action_or_no_progress_class() -> None:
    state = LoopState(step_count=6)
    state.stagnation_counters["repeat_command"] = 3
    state.tool_history = ["file_read|path=src/app.py"] * 3

    signal = detect_repeated_tool_loop(state, threshold=3)

    assert signal is not None
    assert signal.failure_class == "repeated_action"
    assert signal.tool_name == "file_read"
    assert "repeated_fingerprint=file_read|path=src/app.py" in signal.evidence
    assert "Do not call file_read with the same target again" in signal.next_safe_action


def test_observe_tool_result_records_failure_class_in_fama_payload() -> None:
    state = LoopState(step_count=7)
    service = SimpleNamespace(harness=_harness(state))
    result = ToolEnvelope(
        success=False,
        error="cannot access '/tmp/missing': No such file or directory",
        metadata={"path": "/tmp/missing"},
    )

    asyncio.run(observe_tool_result(service, tool_name="file_read", result=result, operation_id="op-2"))

    signal = state.scratchpad["_fama"]["signals"][-1]
    assert signal["kind"] == "remote_local_confusion"
    assert signal["failure_class"] == "wrong_path"
    assert signal["next_safe_action"]
    assert state.failure_events[-1].failure_class == "wrong_path"
    assert state.last_failure_class == "wrong_path"
    assert "wrong_path" in state.working_memory.failures[-1]
    assert state.scratchpad["_recovery_metrics"]["fama_signals_by_kind"]["remote_local_confusion"] == 1
    assert state.scratchpad["_recovery_metrics"]["failure_events_by_class"]["wrong_path"] == 1


def test_fama_bridge_updates_active_subtask_from_signal() -> None:
    state = LoopState(step_count=7)
    state.subtask_ledger = SubtaskLedger(
        task_id="task-1",
        subtasks=[Subtask(subtask_id="S1", title="Read file", goal="Find the target", status="active")],
        active_subtask_id="S1",
    )
    service = SimpleNamespace(harness=_harness(state))

    asyncio.run(
        observe_tool_result(
            service,
            tool_name="file_read",
            result=ToolEnvelope(
                success=False,
                error="cannot access 'missing.py': No such file or directory",
                metadata={"path": "missing.py"},
            ),
            operation_id="op-3",
        )
    )

    active = state.subtask_ledger.active()
    assert active is not None
    assert active.attempts == 1
    assert active.failure_classes == ["wrong_path"]
    assert active.next_action


def test_fama_disabled_returns_no_classification_event() -> None:
    state = LoopState(step_count=8)
    service = SimpleNamespace(harness=_harness(state, _DisabledConfig()))

    asyncio.run(
        observe_tool_result(
            service,
            tool_name="file_read",
            result=ToolEnvelope(success=False, error="No such file or directory"),
        )
    )

    assert "_fama" not in state.scratchpad
