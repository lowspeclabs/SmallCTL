from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from smallctl.models.events import UIEvent, UIEventType
from smallctl.state import LoopState
from smallctl.tools import control
from smallctl.ui.app_flow import _terminal_status_detail
from smallctl.ui.display import (
    _build_backend_rca_strip,
    format_recovery_banner,
    format_run_log_row,
    should_render_event,
    should_render_run_log_row,
)


def test_tui_formats_phase_change_and_context_invalidation_events() -> None:
    phase_row = {
        "event": "context_invalidated",
        "data": {
            "reason": "phase_advanced",
            "details": {"from_phase": "execute", "to_phase": "repair"},
        },
    }
    invalidation_row = {
        "event": "context_invalidated",
        "data": {
            "reason": "verifier_failed",
            "invalidated_artifact_count": 2,
            "invalidated_observation_count": 1,
            "invalidated_summary_count": 0,
        },
    }
    subtask_row = {
        "event": "subtask_transition",
        "data": {"subtask_id": "S2", "title": "Escalate to bigger model", "old_status": "active", "new_status": "done"},
    }

    assert "Phase changed: execute -> repair" in format_run_log_row(phase_row)
    assert "Context invalidated: artifacts=2, observations=1, summaries=0" in format_run_log_row(invalidation_row)
    assert "Subtask update: S2 | Escalate to bigger model | active->done" in format_run_log_row(subtask_row)


def test_task_complete_blocks_fake_escalation_completion() -> None:
    state = LoopState()
    state.run_brief.original_task = "Escalate to a bigger model for help with this bug"
    harness = SimpleNamespace(
        config=SimpleNamespace(escalation_enabled=False, escalation_expose_tool=False),
    )

    blocked = asyncio.run(control.task_complete("Escalated to a bigger model for help.", state=state, harness=harness))

    assert blocked["success"] is False
    assert blocked["metadata"]["reason"] == "escalation_not_performed"


def test_ui_status_detail_and_rca_surface_install_source_blocker() -> None:
    state = LoopState()
    state.current_phase = "repair"
    state.scratchpad["_install_source_invalid_blocker"] = {
        "reason": "upstream_install_source_invalid",
        "host": "apt.fogproject.org",
    }
    state.scratchpad["_context_invalidations"] = [
        {
            "invalidated_artifact_count": 2,
            "invalidated_observation_count": 1,
            "invalidated_summary_count": 0,
        }
    ]
    harness = SimpleNamespace(state=state)

    detail = _terminal_status_detail({"status": "stopped", "reason": "no_tool_calls"}, harness)
    rca = _build_backend_rca_strip(harness)

    assert detail.startswith("Blocked: installer source invalid or unavailable")
    assert "Phase: repair" in rca
    assert "Blocked: invalid upstream install source (apt.fogproject.org)" in rca
    assert "Context invalidated: artifacts=2, observations=1, summaries=0" in rca


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


def test_format_run_log_row_verifier_path_false_negative() -> None:
    row = {
        "channel": "harness",
        "event": "verifier_path_false_negative_guard",
        "data": {"target": "/tmp/pihole-install.sh", "command": "bash /tmp/pihole-install.sh"},
    }
    formatted = format_run_log_row(row)
    assert "verifier path-failure overridden" in formatted
    assert "⚠️" in formatted


def test_format_run_log_row_timeout_override() -> None:
    row = {
        "channel": "harness",
        "event": "timeout_override",
        "data": {"requested_timeout_sec": 600, "effective_timeout_sec": 300, "reason": "capped by harness limit"},
    }
    formatted = format_run_log_row(row)
    assert "timeout capped" in formatted
    assert "⏱️" in formatted


def test_format_run_log_row_fama_health_alarm() -> None:
    row = {
        "channel": "harness",
        "event": "fama_capsule_health_warning",
        "data": {"warning": "FAMA capsules are empty for 3 consecutive prompts"},
    }
    formatted = format_run_log_row(row)
    assert "FAMA capsules are empty" in formatted
    assert "🚨" in formatted


def test_format_run_log_row_patch_recovery_autoread() -> None:
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
    row = {
        "channel": "harness",
        "event": "task_interrupted",
        "data": {"result": {"reason": "cancel_requested"}},
    }

    assert should_render_run_log_row(row) is True
    assert format_run_log_row(row) == "[harness] Task interrupted: cancel_requested"


def test_format_run_log_row_fama_signal_and_mitigation_are_visible() -> None:
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
    row = {
        "channel": "harness",
        "event": "same_scope_iteration_recorded",
        "data": {"turn_type": "ITERATION"},
    }

    assert should_render_run_log_row(row) is True
    assert format_run_log_row(row) == "[harness] Same-scope follow-up recorded: ITERATION"
