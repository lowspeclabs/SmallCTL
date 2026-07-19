from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Agent-Tools"))

from run_diagnose import _classify_failure, _file_patch_target_loop_count
from runscan import _classify_run
from agent_tools_lib import detect_background_state_changing_shell, detect_primary_blockers, detect_shell_execution_anomalies, warn_on_schema_mismatch
from trace_call import _filter_records, _trace_step, _trace_task


def _make_failed_dispatch(tool_name: str, error_kind: str, error: str = "") -> dict:
    return {
        "event": "dispatch_complete",
        "data": {
            "tool_name": tool_name,
            "success": False,
            "error": error,
            "metadata": {"error_kind": error_kind},
        },
    }


def test_file_patch_target_loop_count_detects_stale_patch_failures() -> None:
    dispatches = [
        _make_failed_dispatch("file_patch", "patch_target_not_found"),
        _make_failed_dispatch("file_patch", "patch_target_not_found"),
        _make_failed_dispatch("file_patch", "repeat_sensitive_patch_already_applied"),
        _make_failed_dispatch("file_read", "not_found"),
    ]

    assert _file_patch_target_loop_count(dispatches) == 3


def test_file_patch_target_loop_count_ignores_non_file_patch_failures() -> None:
    dispatches = [
        _make_failed_dispatch("shell_exec", "command_failed"),
        _make_failed_dispatch("file_read", "not_found"),
    ]

    assert _file_patch_target_loop_count(dispatches) == 0


def test_diagnose_classifies_stale_patch_loop_before_model_degeneration() -> None:
    dispatches = [
        _make_failed_dispatch("file_patch", "patch_target_not_found"),
        _make_failed_dispatch("file_patch", "patch_target_not_found"),
        _make_failed_dispatch("file_patch", "patch_target_not_found"),
    ]
    events: Counter[str] = Counter({"model_output_degenerate_loop_exhausted": 1})
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    harness_records: list[dict] = []

    classification = _classify_failure(events, errors, session, dispatches, harness_records)

    assert classification == "file_patch_target_not_found_loop"


def test_runscan_classifies_stale_patch_loop_before_model_degeneration() -> None:
    dispatches = [
        _make_failed_dispatch("file_patch", "patch_target_not_found"),
        _make_failed_dispatch("file_patch", "patch_target_not_found"),
        _make_failed_dispatch("file_patch", "patch_target_not_found"),
    ]
    events: Counter[str] = Counter({"model_output_degenerate_loop_exhausted": 1})
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    task_summary: dict = {}
    harness_records: list[dict] = []

    classification = _classify_run(events, errors, session, task_summary, dispatches, harness_records)

    assert classification == "file_patch_target_not_found_loop"


def test_diagnose_classifies_completed_recovered_run_as_success_with_errors() -> None:
    events = Counter({"model_output_degenerate_loop_exhausted": 1})
    errors = [{"event": "model_output_degenerate_loop_exhausted"}]
    session: dict = {
        "overall_objective_status": "completed",
        "deliverable_verified": False,
        "incomplete_task_ids": [],
    }
    dispatches: list[dict] = []
    harness_records: list[dict] = []

    classification = _classify_failure(events, errors, session, dispatches, harness_records)

    assert classification == "success_with_errors"


def test_runscan_classifies_completed_recovered_run_as_success_with_errors() -> None:
    events = Counter({"model_output_degenerate_loop_exhausted": 1})
    errors = [{"event": "model_output_degenerate_loop_exhausted"}]
    session: dict = {
        "overall_objective_status": "completed",
        "deliverable_verified": False,
        "incomplete_task_ids": [],
    }
    task_summary: dict = {"final_task_status": "completed"}
    dispatches: list[dict] = []
    harness_records: list[dict] = []

    classification = _classify_run(events, errors, session, task_summary, dispatches, harness_records)

    assert classification == "success_with_errors"


def test_diagnose_recommends_fresh_read_for_stale_patch_loop() -> None:
    from run_diagnose import _recommend_next_steps

    dispatches = [
        _make_failed_dispatch("file_patch", "patch_target_not_found"),
        _make_failed_dispatch("file_patch", "patch_target_not_found"),
        _make_failed_dispatch("file_patch", "patch_target_not_found"),
    ]
    events = Counter()
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    harness_records: list[dict] = []

    steps = _recommend_next_steps(events, errors, session, dispatches, harness_records)

    assert any("file_patch" in step for step in steps)
    assert any("non-cached file_read" in step for step in steps)


def test_diagnose_recommends_last_verifier_for_unverified_changes() -> None:
    from run_diagnose import _recommend_next_steps

    dispatches: list[dict] = []
    events = Counter()
    errors: list[dict] = []
    session: dict = {
        "overall_objective_status": "incomplete",
        "deliverable_verified": False,
        "challenge_progress": {
            "code_change_count": 2,
            "verified_after_last_change": False,
            "last_verifier_command": "python3 ./temp/vikunja-9b.py info",
        },
    }
    harness_records: list[dict] = []

    steps = _recommend_next_steps(events, errors, session, dispatches, harness_records)

    assert any("not verified" in step for step in steps)
    assert any("python3 ./temp/vikunja-9b.py info" in step for step in steps)


def _make_failed_write(tool_name: str = "file_write", reason: str = "patch_first_required", error: str = "") -> dict:
    return {
        "event": "dispatch_complete",
        "data": {
            "tool_name": tool_name,
            "success": False,
            "error": error,
            "metadata": {"reason": reason},
        },
    }


def test_diagnose_classifies_patch_first_policy_loop() -> None:
    dispatches = [
        _make_failed_write("file_write", "patch_first_required"),
        _make_failed_write("file_write", "patch_first_required"),
    ]
    events: Counter[str] = Counter({"model_output_degenerate_loop_exhausted": 1})
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    harness_records: list[dict] = []

    classification = _classify_failure(events, errors, session, dispatches, harness_records)

    assert classification == "patch_first_policy_loop"


def test_runscan_classifies_patch_first_policy_loop() -> None:
    dispatches = [
        _make_failed_write("file_write", "patch_first_required"),
        _make_failed_write("file_write", "patch_first_required"),
    ]
    events: Counter[str] = Counter({"model_output_degenerate_loop_exhausted": 1})
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    task_summary: dict = {}
    harness_records: list[dict] = []

    classification = _classify_run(events, errors, session, task_summary, dispatches, harness_records)

    assert classification == "patch_first_policy_loop"


def test_diagnose_classifies_environment_blocker_before_model_degeneration() -> None:
    dispatches = [
        {
            "event": "dispatch_complete",
            "data": {
                "tool_name": "shell_exec",
                "success": False,
                "error": "curl: (7) Failed to connect to localhost port 3456: Connection refused",
            },
        },
        {
            "event": "dispatch_complete",
            "data": {
                "tool_name": "shell_exec",
                "success": False,
                "error": "curl: (7) Failed to connect to localhost port 3456: Connection refused",
            },
        },
    ]
    events: Counter[str] = Counter({"model_output_degenerate_loop_exhausted": 1})
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    harness_records: list[dict] = []

    classification = _classify_failure(events, errors, session, dispatches, harness_records)

    assert classification == "environment_blocker"


def test_diagnose_classifies_circuit_breaker_before_model_degeneration() -> None:
    events: Counter[str] = Counter({"model_output_degenerate_loop_exhausted": 1})
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    dispatches: list[dict] = []
    harness_records: list[dict] = [
        {
            "event": "dispatch_complete",
            "data": {"_stderr_signature_circuit_breaker": {"signature": "% Total % Received"}},
        }
    ]

    classification = _classify_failure(events, errors, session, dispatches, harness_records)

    assert classification == "harness_circuit_breaker_false_positive"


def test_diagnose_classifies_fama_ssh_circuit_breaker_before_model_degeneration() -> None:
    events: Counter[str] = Counter({"model_output_degenerate_loop_exhausted": 1, "fama_ssh_transport_circuit_breaker": 3})
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    dispatches: list[dict] = []
    harness_records: list[dict] = [{"event": "fama_ssh_transport_circuit_breaker"}]

    classification = _classify_failure(events, errors, session, dispatches, harness_records)

    assert classification == "ssh_transport_circuit_breaker_false_positive"


def test_diagnose_classifies_prompt_budget_overflow_before_incomplete() -> None:
    events: Counter[str] = Counter()
    errors: list[dict] = []
    session: dict = {
        "overall_objective_status": "incomplete",
        "deliverable_verified": False,
        "postmortem_summary": "PROMPT BUDGET OVERFLOW: 12651 tokens assembled",
    }
    dispatches: list[dict] = []
    harness_records: list[dict] = []

    classification = _classify_failure(events, errors, session, dispatches, harness_records)

    assert classification == "prompt_budget_overflow"


def test_runscan_classifies_prompt_budget_overflow_before_incomplete() -> None:
    events: Counter[str] = Counter()
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    task_summary: dict = {"postmortem_summary": "PROMPT BUDGET OVERFLOW: 12651 tokens assembled"}
    dispatches: list[dict] = []
    harness_records: list[dict] = []

    classification = _classify_run(events, errors, session, task_summary, dispatches, harness_records)

    assert classification == "prompt_budget_overflow"


def test_primary_blockers_ignore_timeout_in_cli_usage() -> None:
    failed_dispatches = [
        {
            "event": "dispatch_complete",
            "data": {
                "tool_name": "shell_exec",
                "success": False,
                "output": {
                    "stderr": "usage: proxmox-cli lxc create [-h] [--timeout TIMEOUT]",
                    "exit_code": 1,
                },
            },
        }
    ]

    blockers = detect_primary_blockers([], failed_dispatches)

    assert blockers == []


def test_detect_background_state_changing_shell() -> None:
    tools_records = [
        {
            "event": "dispatch_start",
            "trace_id": "abc:task-1:step-8:call-8",
            "data": {
                "tool_name": "shell_exec",
                "arguments": {
                    "command": "python scripts/Proxmox-cli.py lxcs create --node pve --vmid 105",
                    "background": True,
                },
            },
        }
    ]

    blockers = detect_background_state_changing_shell(tools_records)

    assert blockers
    assert blockers[0]["pattern"] == "background_state_changing_shell"


def test_detect_shell_execution_anomalies_flags_masked_cli_error_and_endpoint_loop() -> None:
    tools_records = [
        {
            "event": "dispatch_start",
            "trace_id": "abc:task-1:step-1:call-1",
            "data": {"tool_name": "shell_exec", "arguments": {"command": "cli containers list 2>&1 | tail -40"}},
        },
        {
            "event": "dispatch_complete",
            "trace_id": "abc:task-1:step-1:call-1",
            "data": {"tool_name": "shell_exec", "success": True, "output": {"stdout": "usage: cli [-h]\ncli: error: unrecognized arguments: list"}},
        },
    ]
    for step in range(2, 5):
        tools_records.append({
            "event": "dispatch_start",
            "trace_id": f"abc:task-1:step-{step}:call-{step}",
            "data": {"tool_name": "shell_exec", "arguments": {"command": "curl https://192.0.2.1:8006/api2/json/nodes"}},
        })
        tools_records.append({
            "event": "dispatch_complete",
            "trace_id": f"abc:task-1:step-{step}:call-{step}",
            "data": {"tool_name": "shell_exec", "success": True, "output": {"stdout": ""}},
        })

    patterns = {item["pattern"] for item in detect_shell_execution_anomalies(tools_records)}

    assert "semantic_cli_failure_reported_success" in patterns
    assert "pipeline_may_mask_command_failure" in patterns
    assert "repeated_endpoint_probe_without_new_evidence" in patterns


def test_detect_shell_execution_anomalies_flags_observed_placeholder_config() -> None:
    tools_records = [
        {
            "event": "dispatch_start",
            "trace_id": "abc:task-1:step-1:call-1",
            "data": {"tool_name": "shell_exec", "arguments": {"command": "cat .env"}},
        },
        {
            "event": "dispatch_complete",
            "trace_id": "abc:task-1:step-1:call-1",
            "data": {"tool_name": "shell_exec", "success": True, "output": {"stdout": "PROXMOX_API_URL=https://proxmox.example.local"}},
        },
    ]

    patterns = {item["pattern"] for item in detect_shell_execution_anomalies(tools_records)}

    assert "placeholder_configuration_observed" in patterns


def test_diagnose_classifies_background_state_changing_shell_before_recovery() -> None:
    events: Counter[str] = Counter({"recovery_failure_event_recorded": 1})
    errors: list[dict] = [{"event": "recovery_failure_event_recorded"}]
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    dispatches: list[dict] = []
    harness_records: list[dict] = []
    tools_records = [
        {
            "event": "dispatch_start",
            "data": {
                "tool_name": "shell_exec",
                "arguments": {
                    "command": "python scripts/Proxmox-cli.py lxcs create --node pve --vmid 105",
                    "background": True,
                },
            },
        }
    ]

    classification = _classify_failure(events, errors, session, dispatches, harness_records, tools_records=tools_records)

    assert classification == "background_state_change_unverified"


def test_diagnose_classifies_chat_terminal_repetition_stall() -> None:
    session = {"latest_task_id": "task-0005", "overall_objective_status": "incomplete"}
    harness_records: list[dict] = [
        {"trace_id": "abc:task-0005:step-1:call-1", "event": "chat_tool_selection", "data": {"reason": "non_lookup_chat_terminal_only", "tool_names": ["task_complete", "task_fail"]}},
        {"trace_id": "abc:task-0005:step-1:call-1", "event": "model_output_degenerate_loop_exhausted", "data": {"repeated_phrase": "I understand"}},
        {"trace_id": "abc:task-0005:step-1:call-1", "event": "task_finalize", "data": {"result": {"status": "chat_completed"}}},
    ]
    classification = _classify_failure(Counter(), [], session, [], harness_records)

    assert classification == "chat_terminal_repetition_stall"


def test_detect_apt_deb822_guard_misfire_finds_block_after_validator_passed() -> None:
    from agent_tools_lib import detect_apt_deb822_guard_misfire

    records = [
        {"event": "apt_deb822_preflight_validator_passed", "trace_id": "abc:task-1:step-1:call-1"},
        {"event": "apt_deb822_preflight_blocked", "trace_id": "abc:task-1:step-1:call-1"},
    ]
    misfires = detect_apt_deb822_guard_misfire(records)

    assert len(misfires) == 1
    assert misfires[0]["event"] == "apt_deb822_preflight_blocked"


def test_detect_apt_deb822_guard_misfire_ignores_block_before_pass() -> None:
    from agent_tools_lib import detect_apt_deb822_guard_misfire

    records = [
        {"event": "apt_deb822_preflight_blocked", "trace_id": "abc:task-1:step-1:call-1"},
        {"event": "apt_deb822_preflight_validator_passed", "trace_id": "abc:task-1:step-1:call-1"},
    ]
    misfires = detect_apt_deb822_guard_misfire(records)

    assert len(misfires) == 0


def test_diagnose_classifies_guard_misfire_before_model_degeneration() -> None:
    harness_records: list[dict] = [
        {"event": "apt_deb822_preflight_validator_passed", "trace_id": "abc:task-1:step-1:call-1"},
        {"event": "apt_deb822_preflight_blocked", "trace_id": "abc:task-1:step-1:call-1"},
    ]
    events: Counter[str] = Counter({"model_output_degenerate_loop_exhausted": 1})
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    dispatches: list[dict] = []

    classification = _classify_failure(events, errors, session, dispatches, harness_records)

    assert classification == "guard_misfire"


def test_runscan_classifies_guard_misfire_before_model_degeneration() -> None:
    harness_records: list[dict] = [
        {"event": "apt_deb822_preflight_validator_passed", "trace_id": "abc:task-1:step-1:call-1"},
        {"event": "apt_deb822_preflight_blocked", "trace_id": "abc:task-1:step-1:call-1"},
    ]
    events: Counter[str] = Counter({"model_output_degenerate_loop_exhausted": 1})
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    task_summary: dict = {}
    dispatches: list[dict] = []

    classification = _classify_run(events, errors, session, task_summary, dispatches, harness_records)

    assert classification == "guard_misfire"


def test_warn_on_schema_mismatch_detects_version_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    header = {"event_schema_version": 2, "channels": ["harness"]}
    (run_dir / "run_header.json").write_text(json.dumps(header), encoding="utf-8")
    (run_dir / "harness.jsonl").write_text(
        json.dumps({"event": "test", "event_schema_version": 2}) + "\n",
        encoding="utf-8",
    )
    warnings = warn_on_schema_mismatch(run_dir)
    assert any("schema version 2" in w for w in warnings)


def test_warn_on_schema_mismatch_silent_for_current_version(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    header = {"event_schema_version": 1, "channels": ["harness"]}
    (run_dir / "run_header.json").write_text(json.dumps(header), encoding="utf-8")
    warnings = warn_on_schema_mismatch(run_dir)
    assert warnings == []


def test_trace_call_filter_by_step() -> None:
    grouped = {
        "harness": [
            {"event": "mode_decision", "trace_id": "s:t:step-1:call-1"},
            {"event": "phase_transition", "trace_id": "s:t:step-2:call-1"},
        ],
        "tools": [],
    }
    filtered = _filter_records(grouped, trace_id="s:t:step-1:call-1", step=1, task=None, events=None)
    assert [r["event"] for r in filtered["harness"]] == ["mode_decision"]


def test_trace_call_filter_by_event() -> None:
    grouped = {
        "harness": [
            {"event": "mode_decision", "trace_id": "s:t:step-1:call-1"},
            {"event": "phase_transition", "trace_id": "s:t:step-1:call-1"},
        ],
        "tools": [],
    }
    filtered = _filter_records(grouped, trace_id="s:t:step-1:call-1", step=None, task=None, events={"mode_decision"})
    assert [r["event"] for r in filtered["harness"]] == ["mode_decision"]


def test_trace_step_parsing() -> None:
    assert _trace_step("abc:task-1:step-5:call-2") == 5
    assert _trace_step("abc:task-1:step-5:ctx") == 5
    assert _trace_step("invalid") is None


def test_trace_task_parsing() -> None:
    assert _trace_task("abc:task-1:step-5:call-2") == "task-1"
    assert _trace_task("invalid") is None


def test_diagnose_classifies_policy_block_before_model_tool_loop_stall() -> None:
    events: Counter[str] = Counter({
        "tool_blocked_not_exposed": 1,
        "action_stall": 1,
        "no_tool_recovery": 1,
    })
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    dispatches: list[dict] = []
    harness_records: list[dict] = []

    classification = _classify_failure(events, errors, session, dispatches, harness_records)

    assert classification == "policy_block"


def test_diagnose_classifies_fama_block_before_model_tool_loop_stall() -> None:
    events: Counter[str] = Counter({
        "fama_tool_call_blocked": 1,
        "action_stall": 1,
    })
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    dispatches: list[dict] = []
    harness_records: list[dict] = []

    classification = _classify_failure(events, errors, session, dispatches, harness_records)

    assert classification == "fama_block"


def test_runscan_classifies_policy_block_before_model_tool_loop_stall() -> None:
    events: Counter[str] = Counter({
        "tool_blocked_not_exposed": 1,
        "action_stall": 1,
        "no_tool_recovery": 1,
    })
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    task_summary: dict = {}
    dispatches: list[dict] = []
    harness_records: list[dict] = []

    classification = _classify_run(events, errors, session, task_summary, dispatches, harness_records)

    assert classification == "policy_block"


def test_runscan_classifies_fama_block_before_model_tool_loop_stall() -> None:
    events: Counter[str] = Counter({
        "fama_tool_call_blocked": 1,
        "action_stall": 1,
    })
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    task_summary: dict = {}
    dispatches: list[dict] = []
    harness_records: list[dict] = []

    classification = _classify_run(events, errors, session, task_summary, dispatches, harness_records)

    assert classification == "fama_block"


def test_diagnose_classifies_reasoning_only_exhaustion_before_incomplete() -> None:
    events: Counter[str] = Counter({"reasoning_only_stream_exhausted": 1})
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    dispatches: list[dict] = []
    harness_records: list[dict] = []

    classification = _classify_failure(events, errors, session, dispatches, harness_records)

    assert classification == "model_stream_stall"


def test_runscan_classifies_reasoning_only_exhaustion_before_incomplete() -> None:
    events: Counter[str] = Counter({"reasoning_only_stream_exhausted": 1})
    errors: list[dict] = []
    session: dict = {"overall_objective_status": "incomplete", "deliverable_verified": False}
    task_summary: dict = {"final_task_status": "failed"}
    dispatches: list[dict] = []
    harness_records: list[dict] = []

    classification = _classify_run(events, errors, session, task_summary, dispatches, harness_records)

    assert classification == "model_stream_stall"


def test_harness_reported_blocker_from_session_summary() -> None:
    from agent_tools_lib import harness_reported_blocker

    summaries = {
        "session_summary": {"primary_blocker": "ls: cannot access '/tmp/x/agents.md': No such file or directory"},
        "task_summary": {},
    }

    assert harness_reported_blocker(summaries) == "ls: cannot access '/tmp/x/agents.md': No such file or directory"


def test_harness_reported_blocker_falls_back_to_task_summaries() -> None:
    from agent_tools_lib import harness_reported_blocker

    summaries = {"session_summary": {}, "task_summary": {"primary_blocker": ""}}
    task_summaries = [{"task_id": "task-0001", "primary_blocker": "disk full"}]

    assert harness_reported_blocker(summaries, task_summaries) == "disk full"


def test_harness_reported_blocker_none_when_absent() -> None:
    from agent_tools_lib import harness_reported_blocker

    assert harness_reported_blocker({"session_summary": {}, "task_summary": {}}) is None


def test_detect_phantom_code_changes_flags_staged_missing_file(tmp_path: Path) -> None:
    from agent_tools_lib import detect_phantom_code_changes

    missing = str(tmp_path / "agents.md")
    summaries = {
        "session_summary": {
            "challenge_progress": {"last_code_change_paths": [missing]},
        },
        "task_summary": {},
    }
    tools_records = [
        {
            "event": "dispatch_complete",
            "data": {
                "tool_name": "file_write",
                "success": True,
                "metadata": {"path": missing, "staged_only": True, "write_session_finalized": False},
            },
        }
    ]

    findings = detect_phantom_code_changes(summaries, tools_records)

    assert len(findings) == 1
    assert findings[0]["path"] == missing
    assert findings[0]["staged_evidence"] is True


def test_detect_phantom_code_changes_ignores_existing_files(tmp_path: Path) -> None:
    from agent_tools_lib import detect_phantom_code_changes

    existing = tmp_path / "bookstack_client.py"
    existing.write_text("x = 1\n", encoding="utf-8")
    summaries = {
        "session_summary": {
            "challenge_progress": {"last_code_change_paths": [str(existing)]},
        },
        "task_summary": {},
    }

    assert detect_phantom_code_changes(summaries, []) == []


def test_detect_phantom_code_changes_lists_staging_copies(tmp_path: Path) -> None:
    from agent_tools_lib import detect_phantom_code_changes

    workspace = tmp_path / "repo"
    run_dir = workspace / "logs" / "run-1"
    staging_dir = workspace / ".smallctl" / "write_sessions"
    staging_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (staging_dir / "ws_abc__agents__stage.md").write_text("staged\n", encoding="utf-8")

    missing = str(workspace / "agents.md")
    summaries = {
        "session_summary": {
            "challenge_progress": {"last_code_change_paths": [missing]},
        },
        "task_summary": {},
    }

    findings = detect_phantom_code_changes(summaries, [], run_dir)

    assert len(findings) == 1
    assert findings[0]["staged_evidence"] is False
    assert findings[0]["staging_files"] == ["ws_abc__agents__stage.md"]


def test_diagnose_surfaces_reported_blocker_and_phantom_changes(tmp_path: Path) -> None:
    from run_diagnose import _diagnose

    workspace = tmp_path / "repo"
    run_dir = workspace / "logs" / "run-1"
    run_dir.mkdir(parents=True)
    missing = str(workspace / "agents.md")
    (run_dir / "run_header.json").write_text(
        json.dumps({"channels": ["harness", "tools"], "event_schema_version": 1}),
        encoding="utf-8",
    )
    (run_dir / "harness.jsonl").write_text("", encoding="utf-8")
    (run_dir / "tools.jsonl").write_text(
        json.dumps(
            {
                "event": "dispatch_complete",
                "data": {
                    "tool_name": "file_write",
                    "success": True,
                    "metadata": {"path": missing, "staged_only": True, "write_session_finalized": False},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "session_summary.json").write_text(
        json.dumps(
            {
                "overall_objective_status": "incomplete",
                "deliverable_verified": False,
                "primary_blocker": f"ls: cannot access '{missing}': No such file or directory",
                "challenge_progress": {"last_code_change_paths": [missing]},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "task_summary.json").write_text(
        json.dumps({"final_task_status": "cancelled", "total_tool_calls": 1}),
        encoding="utf-8",
    )

    diagnosis = _diagnose(run_dir)

    assert diagnosis["harness_reported_blocker"].startswith("ls: cannot access")
    assert [p["path"] for p in diagnosis["phantom_code_changes"]] == [missing]
    assert any("non-environmental" in step for step in diagnosis["recommended_next_steps"])
    assert any("missing on disk" in step for step in diagnosis["recommended_next_steps"])


def test_detect_phantom_code_changes_staging_match_excludes_unrelated_sessions(tmp_path: Path) -> None:
    from agent_tools_lib import detect_phantom_code_changes

    workspace = tmp_path / "repo"
    run_dir = workspace / "logs" / "run-1"
    staging_dir = workspace / ".smallctl" / "write_sessions"
    staging_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (staging_dir / "ws_8dcbbc__env_sanitizer__stage.py").write_text("staged\n", encoding="utf-8")
    (staging_dir / "ws_cbb9c2___env__stage.example").write_text("staged\n", encoding="utf-8")

    missing = str(workspace / ".env.example")
    summaries = {
        "session_summary": {
            "challenge_progress": {"last_code_change_paths": [missing]},
        },
        "task_summary": {},
    }

    findings = detect_phantom_code_changes(summaries, [], run_dir)

    assert len(findings) == 1
    assert findings[0]["staging_files"] == ["ws_cbb9c2___env__stage.example"]


def test_detect_phantom_code_changes_staged_evidence_from_output_text(tmp_path: Path) -> None:
    from agent_tools_lib import detect_phantom_code_changes

    missing = str(tmp_path / "agents.md")
    summaries = {
        "session_summary": {
            "challenge_progress": {"last_code_change_paths": [missing]},
        },
        "task_summary": {},
    }
    tools_records = [
        {
            "event": "dispatch_complete",
            "data": {
                "tool_name": "file_write",
                "success": True,
                "output": f"Section `content` written to `{missing}`. Next section inferred: `header`. Staged copy: `/x/.smallctl/write_sessions/ws_1__agents__stage.md`.",
            },
        }
    ]

    findings = detect_phantom_code_changes(summaries, tools_records)

    assert findings[0]["staged_evidence"] is True
