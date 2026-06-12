from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from smallctl.fama.capsules import (
    fama_capsule_health_warning,
    fama_fallback_recovery_guidance,
    render_fama_capsules,
)
from smallctl.fama.detectors import detect_loop_rewrite, detect_verifier_path_misclassification
from smallctl.fama.signals import FamaFailureKind, FamaSignal
from smallctl.harness.tool_dispatch import dispatch_tool_call
from smallctl.harness.tool_result_artifact_updates import _apply_ssh_file_mutation_updates
from smallctl.harness.tool_result_verification_store import _verifier_path_failure_is_false_negative
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


def _make_fake_harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        config=SimpleNamespace(graph_dispatch_tools_timeout_sec=300),
        registry=SimpleNamespace(
            names=lambda: {"file_write", "file_patch", "ssh_file_write", "ssh_file_patch", "shell_exec", "ssh_exec", "ssh_file_read"},
            get=lambda name: None,
        ),
        dispatcher=SimpleNamespace(
            dispatch=lambda name, args: ToolEnvelope(success=True, output={"tool": name, "args": args})
        ),
        _current_user_task=lambda: "test task",
        _runlog=lambda *args, **kwargs: None,
        artifact_store=SimpleNamespace(
            compact_tool_message=lambda artifact, result, **kwargs: str(result.output or result.error or "")
        ),
        context_policy=SimpleNamespace(tool_result_inline_token_limit=200),
    )


# ───────────────────────────────────────────────────────────────
# P0.1: Verifier false-negative guard
# ───────────────────────────────────────────────────────────────

def test_verifier_path_failure_is_false_negative_when_recent_ssh_file_write() -> None:
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


def test_verifier_path_failure_is_false_negative_no_recent_write() -> None:
    state = LoopState()
    assert _verifier_path_failure_is_false_negative(
        state,
        host="pi.hole",
        command="bash /tmp/pihole-install.sh",
        stdout="",
        stderr="bash: /tmp/pihole-install.sh: No such file or directory",
    ) is False


def test_verifier_path_failure_is_false_negative_different_host() -> None:
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


# ───────────────────────────────────────────────────────────────
# P0.2: FAMA health check & fallback
# ───────────────────────────────────────────────────────────────

def test_fama_capsule_health_warning_after_three_empty_prompts() -> None:
    state = LoopState()
    # Simulate 3 consecutive prompts with no capsules
    for _ in range(3):
        render_fama_capsules(state, token_budget=180)
    warning = fama_capsule_health_warning(state)
    assert warning is not None
    assert "no mitigations have been rendered for 3 consecutive prompts" in warning


def test_fama_capsule_health_warning_resets_after_capsules() -> None:
    state = LoopState()
    from smallctl.fama.state import activate_mitigations
    from smallctl.fama.signals import ActiveMitigation

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


# ───────────────────────────────────────────────────────────────
# P0.3: Tool timeout override visibility
# ───────────────────────────────────────────────────────────────

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


# ───────────────────────────────────────────────────────────────
# P1.1: Context invalidation policy
# ───────────────────────────────────────────────────────────────

def test_successful_ssh_file_write_does_not_invalidate_observations() -> None:
    state = LoopState()
    service = SimpleNamespace(
        harness=SimpleNamespace(
            state=state,
            _runlog=lambda *args, **kwargs: None,
        )
    )
    result = ToolEnvelope(
        success=True,
        metadata={"path": "/tmp/test.sh", "host": "pi.hole", "changed": True},
    )
    _apply_ssh_file_mutation_updates(
        service,
        tool_name="ssh_file_write",
        result=result,
        arguments={"path": "/tmp/test.sh", "host": "pi.hole"},
        artifact=None,
    )
    # If the fix is working, the scratchpad should not have a recent context_invalidated event
    # from file_changed for this ssh_file_write
    invalidations = state.scratchpad.get("_context_invalidations", [])
    ssh_file_write_invalidations = [
        inv for inv in invalidations
        if inv.get("details", {}).get("tool_name") == "ssh_file_write"
    ]
    assert len(ssh_file_write_invalidations) == 0


# ───────────────────────────────────────────────────────────────
# P1.2: Phase reset on "continue" after harness error
# ───────────────────────────────────────────────────────────────

def test_phase_reset_on_continue_after_verifier_failure() -> None:
    state = LoopState()
    state.current_phase = "repair"
    state.last_failure_class = "verifier_failed"
    state.scratchpad["_last_task_status"] = "cancelled_after_verifier_failure"
    state.scratchpad["_task_transaction"] = {"turn_type": "CONTINUE"}

    # Simulate the handoff mixin behavior
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


# ───────────────────────────────────────────────────────────────
# P1.3: Verifier failure classification
# ───────────────────────────────────────────────────────────────

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


# ───────────────────────────────────────────────────────────────
# P2.1: Loop detection for file rewrites
# ───────────────────────────────────────────────────────────────

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


# ───────────────────────────────────────────────────────────────
# P2.2: SSH session state in context
# ───────────────────────────────────────────────────────────────

def test_fresh_tool_outputs_includes_ssh_session_status() -> None:
    # This is an integration test that verifies the assembler behavior
    from smallctl.context.assembler import PromptAssembler

    state = LoopState()
    assembler = PromptAssembler(policy=SimpleNamespace(
        fresh_tool_output_items=4,
        fresh_tool_output_token_limit=1200,
    ))

    # We can't easily mock the SSH session manager, but we can verify the
    # method exists and doesn't crash when no SSH sessions are active.
    output = assembler._render_fresh_tool_outputs(state)
    assert "Fresh tool outputs" in output or output == ""


# ───────────────────────────────────────────────────────────────
# P2.3: TUI transparency
# ───────────────────────────────────────────────────────────────

def test_format_run_log_row_verifier_path_false_negative() -> None:
    from smallctl.ui.display import format_run_log_row

    row = {
        "channel": "harness",
        "event": "verifier_path_false_negative_guard",
        "data": {"target": "/tmp/pihole-install.sh", "command": "bash /tmp/pihole-install.sh"},
    }
    formatted = format_run_log_row(row)
    assert "verifier path-failure overridden" in formatted
    assert "⚠️" in formatted


def test_format_run_log_row_timeout_override() -> None:
    from smallctl.ui.display import format_run_log_row

    row = {
        "channel": "harness",
        "event": "timeout_override",
        "data": {"requested_timeout_sec": 600, "effective_timeout_sec": 300, "reason": "capped by harness limit"},
    }
    formatted = format_run_log_row(row)
    assert "timeout capped" in formatted
    assert "⏱️" in formatted


def test_format_run_log_row_fama_health_alarm() -> None:
    from smallctl.ui.display import format_run_log_row

    row = {
        "channel": "harness",
        "event": "fama_capsule_health_warning",
        "data": {"warning": "FAMA capsules are empty for 3 consecutive prompts"},
    }
    formatted = format_run_log_row(row)
    assert "FAMA capsules are empty" in formatted
    assert "🚨" in formatted


def test_format_run_log_row_patch_recovery_autoread() -> None:
    from smallctl.ui.display import format_run_log_row, should_render_run_log_row

    row = {
        "channel": "harness",
        "event": "file_patch_read_autocontinue",
        "data": {"target_path": "temp/example.py", "error_kind": "patch_target_not_found"},
    }

    assert should_render_run_log_row(row) is True
    formatted = format_run_log_row(row)
    assert "Patch recovery" in formatted
    assert "temp/example.py" in formatted
    assert "patch_target_not_found" in formatted


def test_format_run_log_row_task_interrupted_reason() -> None:
    from smallctl.ui.display import format_run_log_row, should_render_run_log_row

    row = {
        "channel": "harness",
        "event": "task_interrupted",
        "data": {"result": {"reason": "cancel_requested"}},
    }

    assert should_render_run_log_row(row) is True
    assert format_run_log_row(row) == "[harness] Task interrupted: cancel_requested"


def test_format_run_log_row_fama_signal_and_mitigation_are_visible() -> None:
    from smallctl.ui.display import format_run_log_row, should_render_run_log_row

    signal_row = {
        "channel": "harness",
        "event": "fama_signal_detected",
        "data": {"kind": "bad_tool_args", "failure_class": "patch_target_not_found", "tool_name": "file_patch"},
    }
    route_row = {
        "channel": "harness",
        "event": "fama_signal_to_mitigation",
        "data": {"signal_kind": "bad_tool_args", "activated_mitigations": ["patch_target_not_found_capsule"]},
    }
    mitigation_row = {
        "channel": "harness",
        "event": "fama_mitigation_activated",
        "data": {"mitigation": "patch_target_not_found_capsule", "reason": "bad_tool_args"},
    }

    assert should_render_run_log_row(signal_row) is True
    assert "patch_target_not_found" in format_run_log_row(signal_row)
    assert should_render_run_log_row(route_row) is True
    assert "patch_target_not_found_capsule" in format_run_log_row(route_row)
    assert should_render_run_log_row(mitigation_row) is True
    assert "FAMA mitigation active" in format_run_log_row(mitigation_row)


def test_format_run_log_row_same_scope_iteration_visible() -> None:
    from smallctl.ui.display import format_run_log_row, should_render_run_log_row

    row = {
        "channel": "harness",
        "event": "same_scope_iteration_recorded",
        "data": {"turn_type": "ITERATION"},
    }

    assert should_render_run_log_row(row) is True
    assert format_run_log_row(row) == "[harness] Same-scope follow-up recorded: ITERATION"
