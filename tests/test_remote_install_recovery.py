from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from smallctl.context.retrieval import LexicalRetriever
from smallctl.fama.detectors import detect_upstream_install_source_invalid
from smallctl.fama.signals import FamaFailureKind
from smallctl.graph.error_hardening import _maybe_pivot_upstream_install_source_invalid
from smallctl.graph.state import GraphRunState
from smallctl.graph.tool_outcomes import _maybe_request_apt_deb822_validator
from smallctl.harness import Harness
from smallctl.harness.tool_result_artifact_updates import _handle_verifier_loop_hard_stop
from smallctl.harness.tool_visibility import resolve_turn_tool_exposure
from smallctl.models.events import UIEvent, UIEventType
from smallctl.models.tool_result import ToolEnvelope
from smallctl.recovery_schema import Subtask, SubtaskLedger
from smallctl.state import ArtifactRecord, EpisodicSummary, ExperienceMemory, LoopState
from smallctl.tools.register import build_registry
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


def _make_harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        memory=SimpleNamespace(prime_write_policy=lambda _task: None),
        _initial_phase="execute",
        _configured_planning_mode=False,
        _configured_tool_profiles=None,
        _runlog=lambda *args, **kwargs: None,
    )


def _build_registry(harness: SimpleNamespace) -> None:
    harness.registry = build_registry(
        SimpleNamespace(state=harness.state, log=SimpleNamespace(info=lambda *args, **kwargs: None)),
        registry_profiles=None,
    )


def test_remote_fog_install_exposes_web_search_in_execute_phase() -> None:
    state = LoopState(cwd="/tmp")
    harness = _make_harness(state)
    task = "install FOG PXE server on root@192.168.1.162"

    Harness._initialize_run_brief(harness, task, raw_task=task)
    Harness._activate_tool_profiles(harness, task)
    _build_registry(harness)

    assert state.task_mode == "remote_execute"
    assert "network_read" in state.active_tool_profiles
    assert state.scratchpad.get("_expose_interactive_session_tools") is True
    exposure = resolve_turn_tool_exposure(harness, "loop")
    assert "web_search" in exposure["names"]
    assert "web_fetch" in exposure["names"]
    assert "ssh_session_start" in exposure["names"]


def test_remote_fog_install_exposes_web_search_in_repair_phase() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    harness = _make_harness(state)
    task = "install FOG PXE server on root@192.168.1.162"

    Harness._initialize_run_brief(harness, task, raw_task=task)
    Harness._activate_tool_profiles(harness, task)
    _build_registry(harness)

    assert state.current_phase == "repair"
    assert state.task_mode == "remote_execute"
    assert "network_read" in state.active_tool_profiles
    exposure = resolve_turn_tool_exposure(harness, "loop")
    assert "web_search" in exposure["names"]
    assert "web_fetch" in exposure["names"]


def test_plain_remote_command_does_not_auto_expose_network_read() -> None:
    state = LoopState(cwd="/tmp")
    harness = _make_harness(state)
    task = "ssh root@192.168.1.162 systemctl restart nginx"

    Harness._initialize_run_brief(harness, task, raw_task=task)
    Harness._activate_tool_profiles(harness, task)
    _build_registry(harness)

    assert state.task_mode == "remote_execute"
    assert "network_read" not in state.active_tool_profiles
    assert "network" in state.active_tool_profiles
    exposure = resolve_turn_tool_exposure(harness, "loop")
    assert "web_search" not in exposure["names"]
    assert "ssh_exec" in exposure["names"]


def test_install_recovery_signals_expose_network_read_without_install_keyword() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_install_source_diagnosis"] = {
        "invalid_fetch_count": 2,
        "resolve_fail_count": 1,
        "public_dns_nxdomain": True,
        "network_ok": True,
        "source_host": "apt.fogproject.org",
    }
    harness = _make_harness(state)
    task = "continue the remote task on root@192.168.1.162"

    Harness._initialize_run_brief(harness, task, raw_task=task)
    state.task_mode = "remote_execute"
    Harness._activate_tool_profiles(harness, task)
    _build_registry(harness)

    assert state.task_mode == "remote_execute"
    assert "network_read" in state.active_tool_profiles
    exposure = resolve_turn_tool_exposure(harness, "loop")
    assert "web_search" in exposure["names"]


def test_empty_install_source_diagnosis_does_not_expose_network_read() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_install_source_diagnosis"] = {}
    harness = _make_harness(state)
    task = "ssh root@192.168.1.162 systemctl restart nginx"

    Harness._initialize_run_brief(harness, task, raw_task=task)
    Harness._activate_tool_profiles(harness, task)
    _build_registry(harness)

    assert "network_read" not in state.active_tool_profiles
    exposure = resolve_turn_tool_exposure(harness, "loop")
    assert "web_search" not in exposure["names"]


def test_repeated_remote_install_failure_counters_expose_network_read() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_remote_install_fetch_fail_count"] = 2
    state.scratchpad["_remote_install_resolve_fail_count"] = 1
    harness = _make_harness(state)
    task = "continue the remote task on root@192.168.1.162"

    Harness._initialize_run_brief(harness, task, raw_task=task)
    state.task_mode = "remote_execute"
    Harness._activate_tool_profiles(harness, task)
    _build_registry(harness)

    assert "network_read" in state.active_tool_profiles
    exposure = resolve_turn_tool_exposure(harness, "loop")
    assert "web_search" in exposure["names"]


def test_remote_installer_preflight_required_signals_expose_network_read() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_remote_installer_preflight"] = {
        "192.168.1.162|root|/opt/fogproject/bin": {
            "status": "required",
            "host": "192.168.1.162",
            "user": "root",
            "cwd": "/opt/fogproject/bin",
        }
    }
    harness = _make_harness(state)
    task = "run the installer on root@192.168.1.162"

    Harness._initialize_run_brief(harness, task, raw_task=task)
    Harness._activate_tool_profiles(harness, task)
    _build_registry(harness)

    assert state.task_mode == "remote_execute"
    assert "network_read" in state.active_tool_profiles
    exposure = resolve_turn_tool_exposure(harness, "loop")
    assert "web_search" in exposure["names"]


def test_remote_interactive_installer_uses_interactive_program_signal() -> None:
    state = LoopState(cwd="/tmp")
    harness = _make_harness(state)
    task = "run the interactive installer on root@192.168.1.162"

    Harness._initialize_run_brief(harness, task, raw_task=task)
    Harness._activate_tool_profiles(harness, task)
    _build_registry(harness)

    assert state.task_mode == "remote_execute"
    assert state.scratchpad.get("_expose_interactive_session_tools") is True
    exposure = resolve_turn_tool_exposure(harness, "loop")
    assert "ssh_session_start" in exposure["names"]


def test_ordinary_remote_command_hides_interactive_ssh_tools() -> None:
    state = LoopState(cwd="/tmp")
    harness = _make_harness(state)
    task = "ssh root@192.168.1.162 systemctl restart nginx"

    Harness._initialize_run_brief(harness, task, raw_task=task)
    Harness._activate_tool_profiles(harness, task)
    _build_registry(harness)

    assert state.task_mode == "remote_execute"
    assert state.scratchpad.get("_expose_interactive_session_tools") is not True
    exposure = resolve_turn_tool_exposure(harness, "loop")
    assert "ssh_session_start" not in exposure["names"]
    assert "ssh_session_read" not in exposure["names"]
    assert "ssh_session_send" not in exposure["names"]
    assert "ssh_session_close" not in exposure["names"]


def test_remote_fog_failure_eventually_pivots_to_research_or_human() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "install FOG PXE server on root@192.168.1.162"
    state.scratchpad["_install_source_diagnosis"] = {
        "invalid_fetch_count": 2,
        "resolve_fail_count": 1,
        "public_dns_nxdomain": True,
        "network_ok": True,
        "source_host": "apt.fogproject.org",
    }
    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(names=lambda: ["web_search"]),
        _runlog=lambda *args, **kwargs: None,
    )
    graph_state = GraphRunState(loop_state=state, thread_id="thread-1", run_mode="loop")

    pivoted = _maybe_pivot_upstream_install_source_invalid(graph_state, harness)

    assert pivoted is True
    assert graph_state.interrupt_payload is not None
    assert graph_state.interrupt_payload["kind"] == "ask_human"
    assert state.scratchpad["_ask_human"] is True


def test_verifier_loop_hard_stop_for_fog_install_requires_research_action() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "install FOG PXE server on root@192.168.1.162"
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    service = SimpleNamespace(harness=harness)
    verdict = {
        "verdict": "fail",
        "command": "apt-cache search fog",
        "tool": "ssh_exec",
        "target": "192.168.1.162 :: apt-cache search fog",
    }

    _handle_verifier_loop_hard_stop(service, verdict, rejection_count=3)

    assert "research" in state.scratchpad["_verifier_loop_required_action_classes"]
    assert "mutation" in state.scratchpad["_verifier_loop_required_action_classes"]
    assert "ask_user" in state.scratchpad["_verifier_loop_required_action_classes"]


def test_retrieval_decay_repeated_artifacts_and_summaries() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.active_intent = "requested_ssh_exec"
    state.run_brief.original_task = "install FOG PXE server on root@192.168.1.162"
    state.working_memory.current_goal = "find valid FOG install source"

    state.artifacts = {
        "art-stale-guess": ArtifactRecord(
            artifact_id="art-stale-guess",
            kind="web_fetch",
            source="http://fogproject.org/fog_1.5.9.tar.gz",
            created_at="2026-06-12T00:00:00+00:00",
            size_bytes=0,
            summary="failed fetch from guessed FOG URL",
            tool_name="web_fetch",
            keywords=["fog", "fetch", "404"],
            metadata={"success": False, "confidence": 0.7},
        ),
        "art-fresh-search": ArtifactRecord(
            artifact_id="art-fresh-search",
            kind="web_search",
            source="web_search:fog project official download",
            created_at="2026-06-12T00:00:01+00:00",
            size_bytes=0,
            summary="official FOG Project download page results",
            tool_name="web_search",
            keywords=["fog", "download", "official"],
            metadata={"success": True, "confidence": 0.9},
        ),
    }
    state.episodic_summaries = [
        EpisodicSummary(
            summary_id="sum-stale-guess",
            created_at="2026-06-12T00:00:00+00:00",
            decisions=["Guessed fogproject.org URL"],
            files_touched=[],
            failed_approaches=["URL returned 404"],
            remaining_plan=["find official download"],
            notes=["stale guessed source"],
        ),
        EpisodicSummary(
            summary_id="sum-fresh-search",
            created_at="2026-06-12T00:00:01+00:00",
            decisions=["Searched official FOG documentation"],
            files_touched=[],
            failed_approaches=[],
            remaining_plan=["download from github releases"],
            notes=["fresh research"],
        ),
    ]

    state.scratchpad["_retrieved_artifact_history"] = [
        {"id": "art-stale-guess", "retrieved_at_step": i} for i in range(4)
    ]
    state.scratchpad["_retrieved_summary_history"] = [
        {"id": "sum-stale-guess", "retrieved_at_step": i} for i in range(4)
    ]

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query="find official FOG download source",
        include_experiences=False,
    )

    selected_artifact_ids = [snippet.artifact_id for snippet in bundle.artifacts]
    selected_summary_ids = [summary.summary_id for summary in bundle.summaries]

    assert "art-fresh-search" in selected_artifact_ids
    assert "art-stale-guess" not in selected_artifact_ids
    assert selected_summary_ids[0] == "sum-fresh-search"
    assert bundle.repeat_counts["artifacts"].get("art-stale-guess", 0) >= 4
    assert bundle.applied_decays["artifacts"].get("art-stale-guess") is not None
    assert bundle.applied_decays["artifacts"]["art-stale-guess"] < 1.0


def test_retrieval_repeat_decay_does_not_suppress_first_selection() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.run_brief.original_task = "install FOG PXE server on root@192.168.1.162"
    state.working_memory.current_goal = "find official FOG download source"

    state.artifacts = {
        "art-official": ArtifactRecord(
            artifact_id="art-official",
            kind="web_fetch",
            source="https://github.com/FOGProject/fogproject/releases",
            created_at="2026-06-12T00:00:00+00:00",
            size_bytes=0,
            summary="official FOG GitHub releases page",
            tool_name="web_fetch",
            keywords=["fog", "github", "releases"],
            metadata={"success": True, "confidence": 0.95},
        ),
    }

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query="official FOG download",
        include_experiences=False,
    )

    assert any(snippet.artifact_id == "art-official" for snippet in bundle.artifacts)
    assert bundle.applied_decays["artifacts"].get("art-official", 1.0) == 1.0


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
    assert any(event == "upstream_install_source_invalid_pivot" for event, _data in runlog_events)


def test_pivot_suppresses_unrelated_completed_subtasks() -> None:
    state = LoopState()
    state.run_brief.original_task = "Install FOG on Debian"
    state.scratchpad["_install_source_diagnosis"] = {
        "invalid_fetch_count": 2,
        "resolve_fail_count": 1,
        "public_dns_nxdomain": True,
        "network_ok": True,
        "source_host": "apt.fogproject.org",
    }
    ledger = SubtaskLedger(
        task_id="task-123",
        subtasks=[
            Subtask(subtask_id="S1", title="Complete user task", goal="Install FOG on Debian", status="active"),
            Subtask(subtask_id="S2", title="Escalate to bigger model", goal="Get help", status="done"),
            Subtask(subtask_id="S3", title="Check disk space", goal="Check disk space", status="done"),
        ],
        active_subtask_id="S1",
    )
    state.subtask_ledger = ledger
    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(names=lambda: ["web_search"]),
        _runlog=lambda *args, **kwargs: None,
    )
    graph_state = GraphRunState(loop_state=state, thread_id="thread-1", run_mode="loop")

    pivoted = _maybe_pivot_upstream_install_source_invalid(graph_state, harness)

    assert pivoted is True
    suppressed = set(state.scratchpad.get("_pivot_suppressed_subtask_ids", []))
    assert "S2" in suppressed
    assert "S3" in suppressed
    assert "S1" not in suppressed


def test_verifier_loop_hard_stop_sets_required_action_classes() -> None:
    state = LoopState()
    state.run_brief.original_task = "Install FOG on Debian"
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    service = SimpleNamespace(harness=harness)
    verdict = {
        "verdict": "fail",
        "command": "dpkg -l | grep -i fog",
        "tool": "ssh_exec",
        "target": "192.168.1.10 :: dpkg -l | grep -i fog",
    }

    _handle_verifier_loop_hard_stop(service, verdict, rejection_count=3)

    assert state.scratchpad["_verifier_loop_required_action_classes"] == [
        "research", "mutation", "ask_user", "stop_blocked"
    ]
    assert state.scratchpad["_verifier_loop_rejection_count"] == 3
    messages = [m for m in state.recent_messages if getattr(m, "role", None) == "system"]
    assert messages
    assert "VERIFIER LOOP HARD STOP" in messages[-1].content
    assert "research" in messages[-1].content


def test_apt_deb822_preflight_creates_approval_interrupt_instead_of_autocontinue() -> None:
    events: list[UIEvent] = []
    runlog_events: list[tuple[str, str, dict[str, object]]] = []
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.active_tool_profiles = ["network"]

    async def _emit(handler, event):
        if handler is not None:
            handler(event)

    harness = SimpleNamespace(
        state=state,
        _emit=_emit,
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
    )
    deps = SimpleNamespace(harness=harness, event_handler=lambda event: events.append(event))
    graph_state = GraphRunState(loop_state=state, thread_id="thread-1", run_mode="loop")
    record = SimpleNamespace(
        tool_name="ssh_exec",
        tool_call_id="call-1",
        operation_id="op-1",
        result=ToolEnvelope(
            success=False,
            error="APT blocked pending deb822 validation.",
            metadata={"reason": "apt_deb822_preflight_required", "host": "192.168.1.162", "user": "root"},
        ),
    )

    created = asyncio.run(_maybe_request_apt_deb822_validator(graph_state, deps, record))

    assert created is True
    assert graph_state.pending_tool_calls == []
    assert state.pending_interrupt["kind"] == "apt_deb822_validator_approval"
    assert state.pending_interrupt["tool_name"] == "ssh_exec"
    assert "debian.sources" in state.pending_interrupt["question"]
    assert any(event.event_type == UIEventType.ALERT for event in events)
    assert any(event == "apt_deb822_validator_approval_requested" for event, _message, _data in runlog_events)


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
    assert _remote_installer_preflight_has_verified_write(
        state, host="pi.hole", user="root", script_path="/tmp/install.sh"
    ) is True
    assert _remote_installer_preflight_has_verified_write(
        state, host="other.host", user="root", script_path="/tmp/install.sh"
    ) is False
    assert _remote_installer_preflight_has_verified_write(
        state, host="pi.hole", user="root", script_path="/tmp/other.sh"
    ) is False


def test_preflight_verified_write_expires_after_8_steps() -> None:
    state = LoopState()
    state.step_count = 5
    _mark_remote_installer_preflight_clean_from_write(
        state, host="pi.hole", user="root", script_path="/tmp/install.sh"
    )
    state.step_count = 13
    assert _remote_installer_preflight_has_verified_write(
        state, host="pi.hole", user="root", script_path="/tmp/install.sh"
    ) is True
    state.step_count = 14
    assert _remote_installer_preflight_has_verified_write(
        state, host="pi.hole", user="root", script_path="/tmp/install.sh"
    ) is False


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
