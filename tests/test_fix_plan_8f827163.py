"""Tests for repair plan 8f827163 — Codebase Shape Analysis fixes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from smallctl.fama.capsules import CAPSULE_TEXT, render_fama_capsules
from smallctl.fama.detectors import (
    detect_objective_verifier_mismatch,
    detect_preflight_contradiction,
    detect_preexisting_state_as_success,
    detect_repeated_remote_installer_failure,
    detect_stale_success_claim,
)
from smallctl.fama.router import MITIGATION_RULES, route_signal
from smallctl.fama.signals import FamaFailureKind, FamaSignal
from smallctl.harness.core_facade import _run_metric_flags
from smallctl.harness.tool_result_verification_store import (
    _verifier_path_failure_is_false_negative,
)
from smallctl.models.events import UIEvent, UIEventType
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools.shell_support_apt_and_outcome import (
    _apt_sources_list_d_guard,
    record_apt_update_result,
    record_sources_list_d_modification,
)
from smallctl.tools.shell_support_installer_guards import (
    _mark_remote_installer_preflight_clean_from_write,
    _remote_installer_preflight_guard,
    _remote_installer_preflight_has_verified_write,
)
from smallctl.ui.display import _build_backend_rca_strip, should_render_event


# ───────────────────────────────────────────────────────────────
# 1. Installer preflight + ssh_file_write integration
# ───────────────────────────────────────────────────────────────


def test_preflight_guard_allows_verified_ssh_file_write() -> None:
    state = LoopState()
    state.step_count = 5
    _mark_remote_installer_preflight_clean_from_write(
        state, host="pi.hole", user="root", script_path="/tmp/install.sh"
    )
    result = _remote_installer_preflight_guard(
        "bash /tmp/install.sh",
        host="pi.hole",
        user="root",
        state=state,
    )
    assert result is None, "Guard should allow verified installer script"


def test_preflight_guard_blocks_unverified_installer() -> None:
    state = LoopState()
    state.step_count = 5
    result = _remote_installer_preflight_guard(
        "bash /tmp/install.sh",
        host="pi.hole",
        user="root",
        state=state,
    )
    assert result is not None
    assert result.get("success") is False
    metadata = result.get("metadata") or {}
    assert metadata.get("reason") == "remote_installer_preflight_required"


def test_preflight_has_verified_write_checks_path_and_host() -> None:
    state = LoopState()
    state.step_count = 5
    _mark_remote_installer_preflight_clean_from_write(
        state, host="pi.hole", user="root", script_path="/tmp/install.sh"
    )
    # Same host, same path
    assert _remote_installer_preflight_has_verified_write(
        state, host="pi.hole", user="root", script_path="/tmp/install.sh"
    ) is True
    # Different host
    assert _remote_installer_preflight_has_verified_write(
        state, host="other.host", user="root", script_path="/tmp/install.sh"
    ) is False
    # Different path
    assert _remote_installer_preflight_has_verified_write(
        state, host="pi.hole", user="root", script_path="/tmp/other.sh"
    ) is False


def test_preflight_verified_write_expires_after_8_steps() -> None:
    state = LoopState()
    state.step_count = 5
    _mark_remote_installer_preflight_clean_from_write(
        state, host="pi.hole", user="root", script_path="/tmp/install.sh"
    )
    state.step_count = 13  # 5 + 8 = 13, still within window
    assert _remote_installer_preflight_has_verified_write(
        state, host="pi.hole", user="root", script_path="/tmp/install.sh"
    ) is True
    state.step_count = 14  # 5 + 9 = 14, expired
    assert _remote_installer_preflight_has_verified_write(
        state, host="pi.hole", user="root", script_path="/tmp/install.sh"
    ) is False


# ───────────────────────────────────────────────────────────────
# 2. False-negative path verifier
# ───────────────────────────────────────────────────────────────


def test_verifier_path_false_negative_when_recent_ssh_file_write() -> None:
    state = LoopState()
    state.recent_messages.append(
        SimpleNamespace(
            role="tool",
            name="ssh_file_write",
            metadata={
                "success": True,
                "host": "pi.hole",
                "path": "/tmp/pihole-install.sh",
            },
        )
    )
    assert _verifier_path_failure_is_false_negative(
        state,
        host="pi.hole",
        command="bash /tmp/pihole-install.sh",
        stdout="",
        stderr="bash: /tmp/pihole-install.sh: No such file or directory",
    ) is True


def test_verifier_path_false_negative_no_recent_write() -> None:
    state = LoopState()
    assert _verifier_path_failure_is_false_negative(
        state,
        host="pi.hole",
        command="bash /tmp/pihole-install.sh",
        stdout="",
        stderr="bash: /tmp/pihole-install.sh: No such file or directory",
    ) is False


def test_verifier_path_false_negative_different_host() -> None:
    state = LoopState()
    state.recent_messages.append(
        SimpleNamespace(
            role="tool",
            name="ssh_file_write",
            metadata={
                "success": True,
                "host": "other.host",
                "path": "/tmp/pihole-install.sh",
            },
        )
    )
    assert _verifier_path_failure_is_false_negative(
        state,
        host="pi.hole",
        command="bash /tmp/pihole-install.sh",
        stdout="",
        stderr="bash: /tmp/pihole-install.sh: No such file or directory",
    ) is False


def test_verifier_path_false_negative_no_path_in_output() -> None:
    state = LoopState()
    state.recent_messages.append(
        SimpleNamespace(
            role="tool",
            name="ssh_file_write",
            metadata={
                "success": True,
                "host": "pi.hole",
                "path": "/tmp/pihole-install.sh",
            },
        )
    )
    assert _verifier_path_failure_is_false_negative(
        state,
        host="pi.hole",
        command="bash /tmp/pihole-install.sh",
        stdout="some other error",
        stderr="permission denied",
    ) is False


# ───────────────────────────────────────────────────────────────
# 3. FAMA detectors
# ───────────────────────────────────────────────────────────────


def test_detect_preflight_contradiction_fires_when_script_exists_in_preflight() -> None:
    state = LoopState()
    state.step_count = 10
    # Set up preflight with verified write
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
    # Simulate a preflight probe result that says NOT FOUND
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
    # Simulate assistant message with task_complete
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
    # Simulate assistant message with different tool
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


# ───────────────────────────────────────────────────────────────
# 4. FAMA router rules for new kinds
# ───────────────────────────────────────────────────────────────


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


# ───────────────────────────────────────────────────────────────
# 5. Capsule text for new kinds
# ───────────────────────────────────────────────────────────────


def test_capsule_text_has_entries_for_new_kinds() -> None:
    assert "repeated_remote_installer_failure_capsule" in CAPSULE_TEXT
    assert "preexisting_state_as_success_capsule" in CAPSULE_TEXT
    assert "repeatedly" in CAPSULE_TEXT["repeated_remote_installer_failure_capsule"].lower()
    assert "already existed" in CAPSULE_TEXT["preexisting_state_as_success_capsule"].lower()


# ───────────────────────────────────────────────────────────────
# 6. TUI display policy — critical events
# ───────────────────────────────────────────────────────────────


def test_should_render_event_shows_critical_events_despite_system_messages_off() -> None:
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        data={"ui_kind": "context_lane_dropped", "message": "Dropped"},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=False) is True

    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        data={"ui_kind": "fama_health_warning", "message": "FAMA warning"},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=False) is True

    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        data={"ui_kind": "terminal_control_failed", "message": "Control failed"},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=False) is True


def test_should_render_event_hides_normal_system_when_off() -> None:
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        data={"ui_kind": "info", "message": "Some info"},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=False) is False


# ───────────────────────────────────────────────────────────────
# 7. Apt sources.list.d guard
# ───────────────────────────────────────────────────────────────


def test_apt_sources_list_d_guard_blocks_install_after_modification() -> None:
    state = LoopState()
    record_sources_list_d_modification(
        state, path="/etc/apt/sources.list.d/bitnami.list", host="pi.hole", user="root"
    )
    result = _apt_sources_list_d_guard(
        "apt-get install -y nginx",
        tool_name="ssh_exec",
        state=state,
        host="pi.hole",
        user="root",
    )
    assert result is not None
    assert result.get("success") is False
    metadata = result.get("metadata") or {}
    assert metadata.get("reason") == "apt_update_required_after_sources_change"


def test_apt_sources_list_d_guard_allows_update_after_modification() -> None:
    state = LoopState()
    record_sources_list_d_modification(
        state, path="/etc/apt/sources.list.d/bitnami.list", host="pi.hole", user="root"
    )
    result = _apt_sources_list_d_guard(
        "apt-get update",
        tool_name="ssh_exec",
        state=state,
        host="pi.hole",
        user="root",
    )
    assert result is None, "apt-get update should be allowed even after modification"


def test_apt_sources_list_d_guard_allows_install_after_successful_update() -> None:
    state = LoopState()
    record_sources_list_d_modification(
        state, path="/etc/apt/sources.list.d/bitnami.list", host="pi.hole", user="root"
    )
    record_apt_update_result(
        state, command="apt-get update", success=True, stderr="", host="pi.hole", user="root"
    )
    result = _apt_sources_list_d_guard(
        "apt-get install -y nginx",
        tool_name="ssh_exec",
        state=state,
        host="pi.hole",
        user="root",
    )
    assert result is None, "apt-get install should be allowed after successful update"


def test_apt_sources_list_d_guard_blocks_on_malformed_entry() -> None:
    state = LoopState()
    record_sources_list_d_modification(
        state, path="/etc/apt/sources.list.d/bitnami.list", host="pi.hole", user="root"
    )
    record_apt_update_result(
        state,
        command="apt-get update",
        success=False,
        stderr="E: Malformed entry 1 in list file /etc/apt/sources.list.d/bitnami.list",
        host="pi.hole",
        user="root",
    )
    result = _apt_sources_list_d_guard(
        "apt-get install -y nginx",
        tool_name="ssh_exec",
        state=state,
        host="pi.hole",
        user="root",
    )
    assert result is not None
    assert result.get("success") is False
    metadata = result.get("metadata") or {}
    assert metadata.get("reason") == "apt_malformed_sources_list"
    assert "bitnami.list" in str(metadata.get("malformed_file", ""))


def test_apt_sources_list_d_guard_no_fire_for_unrelated_commands() -> None:
    state = LoopState()
    result = _apt_sources_list_d_guard(
        "ls -la /etc/apt/sources.list.d/",
        tool_name="ssh_exec",
        state=state,
        host="pi.hole",
        user="root",
    )
    assert result is None


# ───────────────────────────────────────────────────────────────
# 8. deliverable_verified on cancellation
# ───────────────────────────────────────────────────────────────


def test_deliverable_verified_false_on_cancel_without_passing_verifier() -> None:
    state = LoopState()
    challenge_progress = {"verified_after_last_change": True}
    flags = _run_metric_flags(state, challenge_progress, status="cancelled")
    assert flags["deliverable_verified"] is False


def test_deliverable_verified_true_on_cancel_with_passing_verifier() -> None:
    state = LoopState()
    state.last_verifier_verdict = {"verdict": "pass", "command": "pihole status"}
    challenge_progress = {"verified_after_last_change": True}
    flags = _run_metric_flags(state, challenge_progress, status="cancelled")
    assert flags["deliverable_verified"] is True


def test_deliverable_verified_preserved_on_non_cancelled() -> None:
    state = LoopState()
    challenge_progress = {"verified_after_last_change": True}
    flags = _run_metric_flags(state, challenge_progress, status="completed")
    assert flags["deliverable_verified"] is True


def test_deliverable_verified_false_when_not_verified() -> None:
    state = LoopState()
    challenge_progress = {"verified_after_last_change": False}
    flags = _run_metric_flags(state, challenge_progress, status="completed")
    assert flags["deliverable_verified"] is False


# ───────────────────────────────────────────────────────────────
# 9. RCA strip building
# ───────────────────────────────────────────────────────────────


def test_build_backend_rca_strip_shows_failing_verifier() -> None:
    harness = SimpleNamespace(
        state=LoopState(),
    )
    harness.state.last_verifier_verdict = {
        "verdict": "fail",
        "command": "pytest tests/",
    }
    rca = _build_backend_rca_strip(harness)
    assert "Last failing verifier: `pytest tests/`" in rca


def test_build_backend_rca_strip_shows_passing_verifier() -> None:
    harness = SimpleNamespace(
        state=LoopState(),
    )
    harness.state.last_verifier_verdict = {
        "verdict": "pass",
        "command": "pihole status",
    }
    rca = _build_backend_rca_strip(harness)
    assert "Last passing verifier: `pihole status`" in rca


def test_build_backend_rca_strip_shows_recent_failures() -> None:
    harness = SimpleNamespace(
        state=LoopState(),
    )
    harness.state.recent_errors = ["Error 1", "Error 2", "Error 3"]
    rca = _build_backend_rca_strip(harness)
    assert "Recent failures:" in rca
    assert "Error 1" in rca
    assert "Error 2" in rca
    assert "Error 3" in rca


def test_build_backend_rca_strip_shows_fama_health() -> None:
    harness = SimpleNamespace(
        state=LoopState(),
    )
    harness.state.scratchpad["_fama"] = {"signals": [{"kind": "preflight_contradiction"}]}
    rca = _build_backend_rca_strip(harness)
    assert "FAMA signals: 1" in rca


def test_build_backend_rca_strip_shows_hidden_artifacts() -> None:
    harness = SimpleNamespace(
        state=LoopState(),
    )
    harness.state.scratchpad["suppressed_truncated_artifact_ids"] = ["art-1", "art-2"]
    rca = _build_backend_rca_strip(harness)
    assert "Hidden artifacts:" in rca
    assert "art-1" in rca


def test_build_backend_rca_strip_empty_when_no_data() -> None:
    harness = SimpleNamespace(
        state=LoopState(),
    )
    rca = _build_backend_rca_strip(harness)
    assert rca == ""


# ───────────────────────────────────────────────────────────────
# 10. Objective-aware verifier matching (challenge_progress)
# ───────────────────────────────────────────────────────────────


def test_verifier_matches_user_objective_allows_strong_verifier_for_install() -> None:
    state = LoopState()
    state.run_brief = SimpleNamespace(original_task="Install Pi-hole")
    from smallctl.challenge_progress import _verifier_matches_user_objective
    assert _verifier_matches_user_objective(state, "pihole status") is True
    assert _verifier_matches_user_objective(state, "pihole-FTL --version") is True


def test_verifier_matches_user_objective_blocks_weak_verifier_for_install() -> None:
    state = LoopState()
    state.run_brief = SimpleNamespace(original_task="Install Pi-hole")
    from smallctl.challenge_progress import _verifier_matches_user_objective
    assert _verifier_matches_user_objective(state, "ls -la /tmp/install.sh") is False
    assert _verifier_matches_user_objective(state, "test -f /tmp/install.sh") is False
    assert _verifier_matches_user_objective(state, "cat /tmp/install.sh") is False


def test_verifier_matches_user_objective_allows_any_verifier_for_non_install() -> None:
    state = LoopState()
    state.run_brief = SimpleNamespace(original_task="Fix the bug in utils.py")
    from smallctl.challenge_progress import _verifier_matches_user_objective
    assert _verifier_matches_user_objective(state, "ls -la /tmp/test.py") is True
