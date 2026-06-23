from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from smallctl.fama.capsules import (
    CAPSULE_TEXT,
    fama_capsule_health_warning,
    fama_fallback_recovery_guidance,
    render_fama_capsules,
)
from smallctl.fama.detectors import (
    detect_identical_tool_loop,
    detect_interactive_installer_stall,
    detect_loop_rewrite,
    detect_objective_verifier_mismatch,
    detect_preflight_contradiction,
    detect_preexisting_state_as_success,
    detect_repeated_remote_installer_failure,
    detect_stale_success_claim,
    detect_tool_plan_hard_route,
    detect_verifier_path_misclassification,
    detect_weak_verifier_logic,
)
from smallctl.fama.router import MITIGATION_RULES, route_signal
from smallctl.fama.signals import ActiveMitigation, FamaFailureKind, FamaSignal
from smallctl.fama.state import activate_mitigations
from smallctl.harness.config import HarnessConfig
from smallctl.state import LoopState


def _make_state(*, tool_history: list[str] | None = None, step_count: int = 10) -> LoopState:
    state = LoopState()
    state.step_count = step_count
    if tool_history:
        state.tool_history = list(tool_history)
    return state


def test_interactive_installer_stall_detected() -> None:
    """Detect repeated ssh_session_read with same prompt after send."""
    state = _make_state(
        tool_history=[
            "ssh_session_send|sess_abc|y",
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_send|sess_abc|y",
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_send|sess_abc|y",
            "ssh_session_read|sess_abc|prompt1|waiting",
        ]
    )
    signal = detect_interactive_installer_stall(state, threshold=2)
    assert signal is not None
    assert signal.kind == FamaFailureKind.INTERACTIVE_SESSION_STALL
    assert "same prompt" in signal.evidence.lower()


def test_interactive_installer_stall_not_detected_without_send() -> None:
    """No stall if no sends between reads."""
    state = _make_state(
        tool_history=[
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_read|sess_abc|prompt1|waiting",
        ]
    )
    signal = detect_interactive_installer_stall(state, threshold=2)
    assert signal is None


def test_weak_verifier_logic_detected() -> None:
    """Detect verifier pass when ssh_exec output shows interactive prompt."""
    state = _make_state()
    state.scratchpad["_last_verifier_verdict"] = {"verdict": "pass"}
    result = SimpleNamespace(
        success=True,
        metadata={"command": "apt-get install webmin"},
        output={"stdout": "Setting up...\n(y/N) ? ", "stderr": ""},
        error="",
    )
    signal = detect_weak_verifier_logic(state, tool_name="ssh_exec", result=result)
    assert signal is not None
    assert signal.kind == FamaFailureKind.EARLY_STOP
    assert "interactive prompt" in signal.evidence.lower()


def test_weak_verifier_logic_not_detected_when_verifier_fails() -> None:
    """No signal when verifier already failed."""
    state = _make_state()
    state.scratchpad["_last_verifier_verdict"] = {"verdict": "fail"}
    result = SimpleNamespace(
        success=True,
        metadata={"command": "apt-get install webmin"},
        output={"stdout": "Setting up...\n(y/N) ? ", "stderr": ""},
        error="",
    )
    signal = detect_weak_verifier_logic(state, tool_name="ssh_exec", result=result)
    assert signal is None


def test_identical_tool_loop_detected() -> None:
    """Detect same tool with same arguments called repeatedly."""
    state = _make_state(
        tool_history=[
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_read|sess_abc|prompt1|waiting",
        ]
    )
    _ = detect_identical_tool_loop(state, threshold=3)
    _ = detect_identical_tool_loop(state, threshold=3)
    signal = detect_identical_tool_loop(state, threshold=3)
    assert signal is not None
    assert signal.kind == FamaFailureKind.LOOPING
    assert "ssh_session_read" in signal.evidence


def test_identical_tool_loop_not_detected_with_variation() -> None:
    """No loop signal when arguments differ."""
    state = _make_state(
        tool_history=[
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_read|sess_abc|prompt2|waiting",
            "ssh_session_read|sess_abc|prompt3|waiting",
        ]
    )
    signal = detect_identical_tool_loop(state, threshold=3)
    assert signal is None


def test_fama_renders_interactive_installer_stall_capsule_by_turn_10() -> None:
    """At least one FAMA capsule appears in the prompt after stall detection."""
    state = LoopState()
    state.step_count = 10
    state.scratchpad["_fama_config"] = {"enabled": True, "mode": "lite", "capsule_token_budget": 180}
    state.tool_history = [
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
    ]

    signal = detect_interactive_installer_stall(state, threshold=2)
    assert signal is not None

    activate_mitigations(
        state,
        [
            ActiveMitigation(
                name="interactive_installer_stall_capsule",
                reason="stall_detected",
                source_signal=f"interactive_session_stall:{state.step_count}",
                activated_step=state.step_count,
                expires_after_step=state.step_count + 5,
            )
        ],
        max_active=5,
    )

    capsules = render_fama_capsules(state, token_budget=180)
    assert len(capsules) > 0
    combined = "\n".join(capsules)
    assert "interactive" in combined.lower() or "stall" in combined.lower() or "prompt" in combined.lower()
    assert "only `ssh_session_send`" in combined.lower()
    assert "one exact answer" in combined.lower()


def test_interactive_installer_stall_capsule_forbids_retry_installer() -> None:
    """The stall capsule explicitly forbids restarting the installer."""
    state = LoopState()
    state.step_count = 10
    state.scratchpad["_fama_config"] = {"enabled": True, "mode": "lite", "capsule_token_budget": 180}
    state.tool_history = [
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
    ]
    assert detect_interactive_installer_stall(state, threshold=2) is not None
    activate_mitigations(
        state,
        [
            ActiveMitigation(
                name="interactive_installer_stall_capsule",
                reason="stall_detected",
                source_signal=f"interactive_session_stall:{state.step_count}",
                activated_step=state.step_count,
                expires_after_step=state.step_count + 5,
            )
        ],
        max_active=5,
    )
    capsules = render_fama_capsules(state, token_budget=180)
    combined = "\n".join(capsules).lower()
    assert "do not run `ssh_exec`" in combined
    assert "curl" in combined or "wget" in combined
    assert "start another session" in combined


def test_loop_guard_context_aware_nudge_not_generic() -> None:
    """Loop guard for repeated ssh_session_read gives context-aware advice, not generic text."""
    state = LoopState()
    state.step_count = 10
    state.tool_history = [
        "ssh_session_read|sess_abc|prompt1|waiting",
        "ssh_session_read|sess_abc|prompt1|waiting",
        "ssh_session_read|sess_abc|prompt1|waiting",
    ]

    _ = detect_identical_tool_loop(state, threshold=3)
    _ = detect_identical_tool_loop(state, threshold=3)
    signal = detect_identical_tool_loop(state, threshold=3)
    assert signal is not None
    assert signal.kind == FamaFailureKind.LOOPING
    assert "explain the blocker" in signal.next_safe_action.lower() or "try a different" in signal.next_safe_action.lower()
    assert "do not call it again" in signal.next_safe_action.lower()


def test_detect_tool_plan_hard_route_triggers_on_repeated_shell_failures() -> None:
    state = SimpleNamespace(
        scratchpad={
            "_repeated_failure_observations": [
                {
                    "key": "shell_exec::argparse_error",
                    "tool_name": "shell_exec",
                    "domain": "",
                    "pattern": "argparse_error",
                    "count": 3,
                    "last_step": 5,
                    "first_step": 1,
                }
            ]
        },
        step_count=5,
    )
    assert detect_tool_plan_hard_route(state) is True
    assert state.scratchpad.get("_fama_force_tool_plan_next_turn") is True


def test_detect_tool_plan_hard_route_ignores_repeated_non_shell_failures() -> None:
    state = SimpleNamespace(
        scratchpad={
            "_repeated_failure_observations": [
                {
                    "key": "grep::not_found",
                    "tool_name": "grep",
                    "domain": "",
                    "pattern": "not_found",
                    "count": 3,
                    "last_step": 5,
                    "first_step": 1,
                }
            ]
        },
        step_count=5,
    )
    assert detect_tool_plan_hard_route(state) is False


def _apply_fama_loop_mode_guard(config: Any) -> None:
    """Mirror of the guard logic from initialization.py for testing."""
    if (
        str(config.run_mode or "").strip().lower() == "loop"
        and not config.fama_enabled
        and not config.fama_disabled
    ):
        config.fama_enabled = True


def test_fama_disabled_flag_blocks_loop_mode_guard() -> None:
    config = HarnessConfig(
        endpoint="http://localhost:11434",
        model="test",
        run_mode="loop",
        fama_enabled=False,
        fama_disabled=True,
    )
    _apply_fama_loop_mode_guard(config)
    assert config.fama_enabled is False


def test_loop_mode_guard_auto_enables_fama() -> None:
    config = HarnessConfig(
        endpoint="http://localhost:11434",
        model="test",
        run_mode="loop",
        fama_enabled=False,
        fama_disabled=False,
    )
    _apply_fama_loop_mode_guard(config)
    assert config.fama_enabled is True


def test_loop_mode_guard_respects_non_loop_mode() -> None:
    config = HarnessConfig(
        endpoint="http://localhost:11434",
        model="test",
        run_mode="chat",
        fama_enabled=False,
        fama_disabled=False,
    )
    _apply_fama_loop_mode_guard(config)
    assert config.fama_enabled is False


def test_detect_preflight_contradiction_fires_when_script_exists_in_preflight() -> None:
    state = LoopState()
    state.step_count = 10
    state.scratchpad["_remote_installer_preflight"] = {
        "pi.hole|root|/tmp": {
            "status": "clean",
            "script_path": "/tmp/install.sh",
            "host": "pi.hole",
            "user": "root",
            "cwd": "/tmp",
            "created_at_step": 5,
            "verified_by_ssh_file_write": True,
        }
    }
    result = SimpleNamespace(
        success=False,
        metadata={
            "preflight_probes": {
                "script_exists": False,
                "script_path": "/tmp/install.sh",
            },
            "host": "pi.hole",
            "user": "root",
        },
    )
    signal = detect_preflight_contradiction(
        state,
        tool_name="ssh_exec",
        result=result,
        operation_id="test-op-1",
    )
    assert signal is not None
    assert signal.kind == FamaFailureKind.PREFLIGHT_CONTRADICTION
    assert signal.failure_class == "preflight_contradiction"
    assert "ssh_file_write verified it" in signal.evidence


def test_detect_preflight_contradiction_no_fire_when_no_preflight() -> None:
    state = LoopState()
    result = SimpleNamespace(
        success=False,
        metadata={
            "preflight_probes": {
                "script_exists": False,
                "script_path": "/tmp/install.sh",
            },
            "host": "pi.hole",
            "user": "root",
        },
    )
    signal = detect_preflight_contradiction(
        state,
        tool_name="ssh_exec",
        result=result,
        operation_id="test-op-1",
    )
    assert signal is None


def test_detect_repeated_remote_installer_failure_fires_at_threshold() -> None:
    state = LoopState()
    state.step_count = 10
    state.scratchpad["_remote_installer_preflight"] = {
        "host1|user1|/tmp": {
            "status": "missing_critical_files",
            "host": "host1",
            "user": "user1",
            "script_path": "/tmp/install1.sh",
        },
        "host2|user2|/opt": {
            "status": "required",
            "host": "host2",
            "user": "user2",
            "script_path": "/opt/install2.sh",
        },
    }
    signal = detect_repeated_remote_installer_failure(state, threshold=2)
    assert signal is not None
    assert signal.kind == FamaFailureKind.REPEATED_REMOTE_INSTALLER_FAILURE
    assert signal.failure_class == "repeated_remote_installer_failure"
    assert "failed 2 times" in signal.evidence


def test_detect_repeated_remote_installer_failure_no_fire_below_threshold() -> None:
    state = LoopState()
    state.scratchpad["_remote_installer_preflight"] = {
        "host1|user1|/tmp": {
            "status": "missing_critical_files",
            "host": "host1",
            "user": "user1",
            "script_path": "/tmp/install1.sh",
        },
    }
    signal = detect_repeated_remote_installer_failure(state, threshold=2)
    assert signal is None


def test_detect_stale_success_claim_fires_on_blocked_task_complete() -> None:
    state = LoopState()
    state.messages = [
        {
            "role": "assistant",
            "content": "I have successfully installed Pi-hole on the server.",
            "tool_calls": [{"function": {"name": "task_complete"}}],
        }
    ]
    result = SimpleNamespace(
        success=False,
        metadata={"reason": "task_complete_blocked_in_staged_execution"},
    )
    signal = detect_stale_success_claim(
        state,
        tool_name="task_complete",
        result=result,
        operation_id="test-op-1",
    )
    assert signal is not None
    assert signal.kind == FamaFailureKind.STALE_SUCCESS_CLAIM
    assert signal.failure_class == "stale_success_claim"


def test_detect_stale_success_claim_no_fire_on_success() -> None:
    state = LoopState()
    result = SimpleNamespace(success=True, metadata={})
    signal = detect_stale_success_claim(
        state,
        tool_name="task_complete",
        result=result,
        operation_id="test-op-1",
    )
    assert signal is None


def test_detect_stale_success_claim_no_fire_on_wrong_tool() -> None:
    state = LoopState()
    result = SimpleNamespace(
        success=False,
        metadata={"reason": "task_complete_blocked_in_staged_execution"},
    )
    signal = detect_stale_success_claim(
        state,
        tool_name="shell_exec",
        result=result,
        operation_id="test-op-1",
    )
    assert signal is None


def test_detect_objective_verifier_mismatch_fires_on_weak_install_verifier() -> None:
    state = LoopState()
    state.run_brief = SimpleNamespace(original_task="Install Pi-hole on the remote server")
    state.last_verifier_verdict = {
        "verdict": "pass",
        "command": "ls -la /tmp/pihole-install.sh",
    }
    signal = detect_objective_verifier_mismatch(state)
    assert signal is not None
    assert signal.kind == FamaFailureKind.OBJECTIVE_MISMATCH
    assert signal.failure_class == "objective_mismatch"
    assert "file existence" in signal.evidence


def test_detect_objective_verifier_mismatch_no_fire_on_non_install_task() -> None:
    state = LoopState()
    state.run_brief = SimpleNamespace(original_task="Fix the bug in utils.py")
    state.last_verifier_verdict = {
        "verdict": "pass",
        "command": "ls -la /tmp/test.py",
    }
    signal = detect_objective_verifier_mismatch(state)
    assert signal is None


def test_detect_objective_verifier_mismatch_no_fire_on_strong_verifier() -> None:
    state = LoopState()
    state.run_brief = SimpleNamespace(original_task="Install Pi-hole on the remote server")
    state.last_verifier_verdict = {
        "verdict": "pass",
        "command": "pihole status",
    }
    signal = detect_objective_verifier_mismatch(state)
    assert signal is None


def test_detect_preexisting_state_as_success_fires_when_task_complete_after_preexisting() -> None:
    state = LoopState()
    state.scratchpad["_preexisting_remote_state_observed"] = {
        "host": "pi.hole",
        "path": "/usr/bin/pihole-FTL",
    }
    state.messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "task_complete"}}
            ],
        }
    ]
    signal = detect_preexisting_state_as_success(state)
    assert signal is not None
    assert signal.kind == FamaFailureKind.PREEXISTING_STATE_AS_SUCCESS
    assert signal.failure_class == "preexisting_state_as_success"
    assert "pre-existing remote state" in signal.evidence


def test_detect_preexisting_state_as_success_no_fire_without_task_complete() -> None:
    state = LoopState()
    state.scratchpad["_preexisting_remote_state_observed"] = {
        "host": "pi.hole",
        "path": "/usr/bin/pihole-FTL",
    }
    state.messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "shell_exec"}}
            ],
        }
    ]
    signal = detect_preexisting_state_as_success(state)
    assert signal is None


def test_detect_preexisting_state_as_success_no_fire_without_preexisting() -> None:
    state = LoopState()
    state.messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "task_complete"}}
            ],
        }
    ]
    signal = detect_preexisting_state_as_success(state)
    assert signal is None


def test_router_has_rules_for_all_new_kinds() -> None:
    assert FamaFailureKind.REPEATED_REMOTE_INSTALLER_FAILURE in MITIGATION_RULES
    assert FamaFailureKind.PREEXISTING_STATE_AS_SUCCESS in MITIGATION_RULES
    assert "repeated_remote_installer_failure_capsule" in MITIGATION_RULES[FamaFailureKind.REPEATED_REMOTE_INSTALLER_FAILURE]
    assert "preexisting_state_as_success_capsule" in MITIGATION_RULES[FamaFailureKind.PREEXISTING_STATE_AS_SUCCESS]


def test_route_signal_repeated_remote_installer_failure() -> None:
    state = LoopState()
    signal = FamaSignal(
        kind=FamaFailureKind.REPEATED_REMOTE_INSTALLER_FAILURE,
        severity=2,
        source="preflight",
        evidence="test",
        step=1,
    )
    mitigations = route_signal(signal, state=state, config=SimpleNamespace(loop_guard_stagnation_threshold=3))
    names = {m.name for m in mitigations}
    assert "repeated_remote_installer_failure_capsule" in names
    assert "evidence_reuse_capsule" in names


def test_route_signal_preexisting_state_as_success() -> None:
    state = LoopState()
    signal = FamaSignal(
        kind=FamaFailureKind.PREEXISTING_STATE_AS_SUCCESS,
        severity=2,
        source="task_result",
        evidence="test",
        step=1,
    )
    mitigations = route_signal(signal, state=state, config=SimpleNamespace(loop_guard_stagnation_threshold=3))
    names = {m.name for m in mitigations}
    assert "preexisting_state_as_success_capsule" in names
    assert "acceptance_checklist_capsule" in names


def test_route_signal_ssh_auth_failure_not_install_failure() -> None:
    """SSH authentication failures must route to remote_auth_failure_capsule,
    not be misclassified as repeated remote installer failures.
    """
    state = LoopState(step_count=2)
    state.task_mode = "remote_execute"
    signal = FamaSignal(
        kind=FamaFailureKind.EARLY_STOP,
        severity=2,
        source="verifier",
        evidence="task_complete rejected with verifier verdict fail: systemctl status docker [permission denied (publickey,password).]",
        step=1,
        tool_name="ssh_exec",
        failure_class="verifier_failed",
    )
    mitigations = route_signal(signal, state=state, config=SimpleNamespace(loop_guard_stagnation_threshold=3))
    names = {m.name for m in mitigations}
    assert "remote_auth_failure_capsule" in names
    assert "repeated_remote_installer_failure_capsule" not in names
    assert "interactive_installer_stall_capsule" not in names


def test_route_signal_ssh_exec_apt_install_is_install_failure() -> None:
    """ssh_exec failures that mention apt/install markers still route to install capsules."""
    state = LoopState(step_count=2)
    state.task_mode = "remote_execute"
    signal = FamaSignal(
        kind=FamaFailureKind.EARLY_STOP,
        severity=2,
        source="verifier",
        evidence="task_complete rejected with verifier verdict fail: apt install -y foo",
        step=1,
        tool_name="ssh_exec",
        failure_class="verifier_failed",
    )
    mitigations = route_signal(signal, state=state, config=SimpleNamespace(loop_guard_stagnation_threshold=3))
    names = {m.name for m in mitigations}
    assert "repeated_remote_installer_failure_capsule" in names


def test_capsule_text_has_entries_for_new_kinds() -> None:
    assert "repeated_remote_installer_failure_capsule" in CAPSULE_TEXT
    assert "preexisting_state_as_success_capsule" in CAPSULE_TEXT
    assert "repeatedly" in CAPSULE_TEXT["repeated_remote_installer_failure_capsule"].lower()
    assert "already existed" in CAPSULE_TEXT["preexisting_state_as_success_capsule"].lower()


def test_fama_capsule_health_warning_after_three_empty_prompts() -> None:
    state = LoopState()
    for _ in range(3):
        render_fama_capsules(state, token_budget=180)
    warning = fama_capsule_health_warning(state)
    assert warning is not None
    assert "no mitigations have been rendered for 3 consecutive prompts" in warning


def test_fama_capsule_health_warning_resets_after_capsules() -> None:
    state = LoopState()
    activate_mitigations(
        state,
        [ActiveMitigation(name="done_gate", reason="test", source_signal="early_stop:0", activated_step=0, expires_after_step=10)],
        max_active=5,
    )
    for _ in range(3):
        render_fama_capsules(state, token_budget=180)
    warning = fama_capsule_health_warning(state)
    assert warning is None


def test_fama_fallback_recovery_guidance_provides_path_mitigation() -> None:
    state = LoopState()
    state.scratchpad["_fama_empty_streak"] = 3
    state.last_failure_class = "path"
    lines = fama_fallback_recovery_guidance(state)
    assert any("path failure" in line.lower() for line in lines)


def test_fama_fallback_recovery_guidance_empty_when_streak_below_three() -> None:
    state = LoopState()
    state.scratchpad["_fama_empty_streak"] = 2
    lines = fama_fallback_recovery_guidance(state)
    assert lines == []


def test_detect_loop_rewrite_fires_on_third_similar_write() -> None:
    state = LoopState()
    content = "#!/bin/bash\necho hello\n"
    for _ in range(3):
        signal = detect_loop_rewrite(
            state,
            tool_name="ssh_file_write",
            arguments={"path": "/tmp/test.sh", "content": content},
        )
    assert signal is not None
    assert signal.kind == FamaFailureKind.LOOPING
    assert "rewritten" in signal.evidence.lower()


def test_detect_loop_rewrite_no_fire_on_dissimilar_writes() -> None:
    state = LoopState()
    for i in range(3):
        signal = detect_loop_rewrite(
            state,
            tool_name="ssh_file_write",
            arguments={"path": "/tmp/test.sh", "content": f"#!/bin/bash\necho {i}\n"},
        )
    assert signal is None


def test_detect_verifier_path_misclassification_fires_on_recent_write() -> None:
    state = LoopState()
    state.last_verifier_verdict = {
        "verdict": "fail",
        "failure_mode": "path",
        "command": "bash /tmp/pihole-install.sh",
    }
    state.recent_messages.append(
        SimpleNamespace(
            role="tool",
            name="ssh_file_write",
            metadata={
                "success": True,
                "path": "/tmp/pihole-install.sh",
            },
        )
    )
    signal = detect_verifier_path_misclassification(state)
    assert signal is not None
    assert signal.failure_class == "verifier_misclassification"
    assert "ssh_file_write confirmed it exists" in signal.evidence


def test_detect_verifier_path_misclassification_no_fire_without_path_failure() -> None:
    state = LoopState()
    state.last_verifier_verdict = {
        "verdict": "fail",
        "failure_mode": "permission_denied",
        "command": "bash /tmp/pihole-install.sh",
    }
    signal = detect_verifier_path_misclassification(state)
    assert signal is None
