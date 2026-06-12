from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.fama.detectors import detect_upstream_install_source_invalid
from smallctl.fama.signals import FamaFailureKind
from smallctl.graph.error_hardening import _maybe_pivot_upstream_install_source_invalid
from smallctl.graph.state import GraphRunState
from smallctl.state import LoopState
from smallctl.tools import control
from smallctl.ui.app_flow import _terminal_status_detail
from smallctl.ui.display import _build_backend_rca_strip, format_recovery_banner


def test_detect_upstream_install_source_invalid_from_accumulated_diagnosis() -> None:
    state = LoopState()
    state.step_count = 6
    state.run_brief.original_task = "Install FOG on Debian"
    state.scratchpad["_install_source_diagnosis"] = {
        "invalid_fetch_count": 2,
        "resolve_fail_count": 1,
        "public_dns_nxdomain": True,
        "network_ok": True,
        "source_host": "apt.fogproject.org",
    }

    signal = detect_upstream_install_source_invalid(state)

    assert signal is not None
    assert signal.kind is FamaFailureKind.UPSTREAM_INSTALL_SOURCE_INVALID
    assert "apt.fogproject.org" in signal.evidence
    assert "general network access still works" in signal.evidence


def test_detect_upstream_install_source_invalid_requires_public_dns_and_network_evidence() -> None:
    state = LoopState()
    state.run_brief.original_task = "Install a package from an upstream repo"
    state.scratchpad["_install_source_diagnosis"] = {
        "invalid_fetch_count": 1,
        "resolve_fail_count": 1,
        "install_context_resolve_failed": True,
        "source_host": "repo.example.invalid",
    }

    assert detect_upstream_install_source_invalid(state) is None


def test_pivot_upstream_install_source_invalid_pauses_for_clarification() -> None:
    state = LoopState()
    state.run_brief.original_task = "Install FOG on Debian"
    state.scratchpad["_install_source_diagnosis"] = {
        "invalid_fetch_count": 2,
        "resolve_fail_count": 1,
        "public_dns_nxdomain": True,
        "network_ok": True,
        "source_host": "apt.fogproject.org",
    }
    runlog_events: list[tuple[str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(names=lambda: ["web_search"]),
        _runlog=lambda event, _message, **data: runlog_events.append((event, data)),
    )
    graph_state = GraphRunState(loop_state=state, thread_id="thread-1", run_mode="loop")

    pivoted = _maybe_pivot_upstream_install_source_invalid(graph_state, harness)

    assert pivoted is True
    assert graph_state.pending_tool_calls == []
    assert graph_state.interrupt_payload is not None
    assert graph_state.interrupt_payload["kind"] == "ask_human"
    assert graph_state.interrupt_payload["reason"] == "upstream_install_source_invalid"
    assert "apt.fogproject.org" in graph_state.interrupt_payload["question"]
    assert state.pending_interrupt == graph_state.interrupt_payload
    assert state.scratchpad["_ask_human"] is True
    assert format_recovery_banner("upstream_install_source_invalid_pivot", {}) == "Blocked: installer source invalid or unavailable"
    assert any(event == "upstream_install_source_invalid_pivot" for event, _data in runlog_events)


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
