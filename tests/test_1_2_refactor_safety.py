from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from smallctl.config import SmallctlConfig, resolve_config
from smallctl.config_support import _env_config_key_names
from smallctl.client.client_transport_helpers import (
    context_pressure_diagnostics,
    extract_available_tool_names,
    latest_user_message_audit,
    llamacpp_model_unloaded_details,
    parse_retry_after_seconds,
    provider_root,
    request_first_token_timeout_sec,
    tool_name,
)
from smallctl.harness import HarnessConfig
from smallctl.harness.remote_mutation_helpers import (
    bounded_region_not_found,
    readback_content_satisfies_requirement,
    remote_missing_file_markers,
    remote_mutation_guessed_paths,
    remote_mutation_target_matches,
    should_emit_small_file_rewrite_nudge,
    tool_result_path_host,
)
from smallctl.harness.task_boundary_summary import (
    clip_task_summary_text,
    extract_task_terminal_message,
    task_duration_seconds,
)
from smallctl.main import build_harness_config_kwargs
from smallctl.search_server.config import SearchServerConfig
from smallctl.graph.progress_guard import _maybe_inject_verifier_success_nudge, _verifier_success_nudge_key
from smallctl.graph.state import GraphRunState, ToolExecutionRecord
from smallctl.graph.lifecycle_step_budget import STEP_BUDGET_NUDGE_THRESHOLD, _maybe_inject_step_budget_nudge
from smallctl.harness.run_mode import _approved_plan_matches_plan_interrupt
from smallctl.harness.runtime_facade import _approved_plan_matches_interrupt
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools import network, shell
from smallctl.tools.dispatcher import normalize_tool_request
from smallctl.tools.dispatcher_artifact_normalization import (
    normalize_artifact_read_request,
    normalize_web_fetch_request,
)
from smallctl.tools.ast_patch_results import (
    build_ast_patch_metadata,
    build_diff_preview,
    dedupe_symbols,
    supported_ast_patch_operations,
    unsupported_language_failure,
)
from smallctl.tools import ast_patch_parsing
from smallctl.tools import ast_patch as ast_patch_module
from smallctl.tools.ast_patch_operations import apply_python_ast_patch
from smallctl.tools.ast_patch_parsing import (
    find_function_candidates,
    parse_expression,
    parse_python_module,
    parse_statement_block,
)
from smallctl.tools.control_remote_mutation import (
    remote_mutation_block_payload,
    remote_mutation_verification_requirement,
)
from smallctl.tools.control_objective_ledger import (
    ensure_multi_objective_ledger,
    multi_objective_completion_block,
)
from smallctl.tools.control_phase_gates import (
    mutation_expectation_block,
    phase_promotion_gate_block,
    task_involves_interactive_program,
)
from smallctl.tools.control_phase_contracts import (
    normalize_phase_contract_payload,
    phase_check_quality_from_command,
    phase_contract_validation_error,
)
from smallctl.tools.control_plan_subtasks import plan_subtask_completion_block
from smallctl.tools.control_post_change import (
    focused_verifier_command_for_path,
    missing_dependency_block,
    post_change_verification_block,
)
from smallctl.tools.control_verifier_helpers import (
    normalized_verifier_verdict,
    verifier_failure_summary,
    verifier_requires_human_approval,
)
from smallctl.tools.control_weather import (
    has_specific_weather_answer,
    is_weather_lookup_task,
    looks_like_weather_search_meta_completion,
)
from smallctl.tools.control_write_session_helpers import (
    write_session_resume_action,
    write_session_schema_failure,
    write_session_warning,
)
from smallctl.tools.control_loop_status_helpers import (
    max_steps_progress,
    subtask_ledger_status,
    write_session_status_payload,
)
from smallctl.tools.dispatcher_request_normalization import (
    normalize_initial_tool_request,
    repair_ssh_exec_malformed_args,
    repair_write_session_path_from_state,
)
from smallctl.tools.dispatcher_remote_paths import (
    command_mentions_remote_absolute_path,
    looks_like_remote_absolute_path,
    looks_like_remote_infrastructure_probe_command,
)
from smallctl.tools.dispatcher_shell_guards import (
    guard_harness_tool_as_ssh_shell_command,
    guard_nested_raw_ssh_in_ssh_exec,
    looks_like_raw_ssh_shell_command,
    raw_ssh_shell_block_envelope,
)
from smallctl.tools.dispatcher_ssh_auth import (
    password_fingerprint,
    ssh_auth_debug_metadata,
    ssh_auth_recovery_entry_key,
)
from smallctl.tools.dispatcher_ssh_context import (
    infer_ssh_password_from_state_context,
    infer_ssh_user_from_state_context,
    looks_like_ssh_password,
    ssh_task_context_texts,
    text_mentions_ssh_target,
)
from smallctl.tools.dispatcher_ssh_memory import (
    infer_ssh_password,
    infer_ssh_password_from_execution_records,
    infer_ssh_password_from_session_memory,
    infer_ssh_user_from_execution_records,
    infer_ssh_user_from_session_memory,
    session_ssh_target_record,
    ssh_record_likely_authenticated,
)
from smallctl.tools.process_lifecycle import build_process_output, cancel_tasks, truncate_output
from smallctl.tools.shell_support import (
    _apt_deb822_preflight_guard,
    _foreground_command_guard,
    _interactive_installer_yes_pipe_guard,
    _is_deb822_preflight_clean,
    _looks_like_deb822_validator,
    _mark_deb822_preflight_clean,
    _remote_installer_preflight_guard,
    _shell_execution_authoring_guard,
    guard_fail,
    record_apt_update_result,
    validate_sources_file,
)
from smallctl.tools.web import _resolve_fetch_selector
from smallctl.tools.web_artifact_refs import resolve_search_result_from_artifact_reference
from smallctl.tools.web_budget import budget_remaining, ensure_budget, mark_web_fetch_budget_exhausted
from smallctl.tools.web_fetch_artifacts import persist_fetch_artifact
from smallctl.tools.web_result_index import (
    assign_fetch_ids,
    load_fetch_artifact_index,
    load_result_index,
    record_fetch_artifact_mapping,
    unknown_result_id_error,
    update_result_index,
)
from smallctl.tools.fs_loop_guard import LoopGuardDecision, _emit_block
from smallctl.tools.fs_loop_guard_status import active_loop_guard_paths, recent_complete_reads


class _EmptyStream:
    async def read(self, _chunk_size: int) -> bytes:
        return b""

    def close(self) -> None:
        pass


class _TimeoutProbeProcess:
    def __init__(self) -> None:
        self.stdout = _EmptyStream()
        self.stderr = _EmptyStream()
        self.stdin = _EmptyStream()
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            raise asyncio.TimeoutError
        return self.returncode


class _FakeArtifactStore:
    def persist_generated_text(self, **kwargs):
        return SimpleNamespace(artifact_id="A1", **kwargs)


class _FakePlan:
    def iter_steps(self):
        return [SimpleNamespace(step_id="S1")]


def test_local_preflight_timeout_kills_probe(monkeypatch) -> None:
    proc = _TimeoutProbeProcess()
    harness = SimpleNamespace(_active_processes={proc})

    async def _fake_create_process(**_kwargs):
        return proc

    monkeypatch.setattr(shell, "_create_process", _fake_create_process)

    result = asyncio.run(
        shell._run_local_installer_preflight_probes(
            "./install.sh",
            state=LoopState(cwd="/tmp"),
            harness=harness,
        )
    )

    assert result["probe_error"] == "Preflight probes timed out after 30s"
    assert proc.terminate_calls == 1
    assert proc.wait_calls >= 2
    assert proc not in harness._active_processes


def test_remote_preflight_timeout_kills_and_unregisters_probe(monkeypatch) -> None:
    proc = _TimeoutProbeProcess()
    harness = SimpleNamespace(_active_processes={proc})

    async def _fake_create_process(**_kwargs):
        return proc

    from smallctl.tools import shell as _shell_module
    monkeypatch.setattr(_shell_module, "create_process", _fake_create_process)
    monkeypatch.setattr(network, "_build_ssh_command", lambda **_kwargs: ("ssh example", {}))

    result = asyncio.run(
        network._run_remote_installer_preflight_probes(
            host="example.test",
            command="./install.sh",
            state=LoopState(cwd="/tmp"),
            harness=harness,
        )
    )

    assert result["probe_error"] == "Preflight probes timed out after 30s"
    assert proc.terminate_calls == 1
    assert proc not in harness._active_processes


def test_verifier_success_nudge_key_is_stable_for_same_context() -> None:
    state = LoopState(cwd="/tmp")
    state.files_changed_this_cycle = ["src/app.py"]
    graph_state = GraphRunState(loop_state=state, thread_id="thread-1", run_mode="loop")
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="shell_exec",
            args={"command": "python -m py_compile src/app.py"},
            tool_call_id=None,
            result=ToolEnvelope(success=True, output={"exit_code": 0}),
        )
    ]

    _maybe_inject_verifier_success_nudge(state, graph_state)
    first_keys = [key for key in state.scratchpad if key.startswith("_verifier_nudge_")]
    _maybe_inject_verifier_success_nudge(state, graph_state)

    assert len(first_keys) == 1
    assert [key for key in state.scratchpad if key.startswith("_verifier_nudge_")] == first_keys
    assert len([message for message in state.recent_messages if message.metadata.get("recovery_kind") == "verifier_success_completion_prompt"]) == 1


def test_ast_patch_result_helpers_preserve_metadata_shape() -> None:
    assert supported_ast_patch_operations() == [
        "add_import",
        "replace_function",
        "insert_in_function",
        "update_call_keyword",
        "add_dataclass_field",
    ]
    assert dedupe_symbols(["Example", "", "Example", "field"]) == ["Example", "field"]

    diff_preview = build_diff_preview("old\n", "new\n")
    assert "--- before" in diff_preview
    assert "+++ after" in diff_preview

    metadata = build_ast_patch_metadata(
        path=Path("src/app.py"),
        requested_path="src/app.py",
        source_path=Path("src/app.py"),
        session=None,
        staged_only=False,
        language="python",
        operation="add_import",
        target={"module": "os"},
        payload={},
        changed=True,
        updated_text="new\n",
        original_text="old\n",
        matched_node_count=1,
        touched_symbols=["os"],
        dry_run=False,
        expected_followup_verifier=None,
        staging_path=None,
        status_block=None,
    )
    assert metadata["diff_preview"] == diff_preview
    assert metadata["changed"] is True
    assert metadata["touched_symbols"] == ["os"]

    failure = unsupported_language_failure(
        path=Path("src/app.rb"),
        requested_path="src/app.rb",
        language="ruby",
        operation="add_import",
    )
    assert failure["success"] is False
    assert failure["metadata"]["error_kind"] == "unsupported_language"


def test_ast_patch_parsing_helpers_are_importable_and_parse_when_libcst_available() -> None:
    assert callable(parse_python_module)
    assert callable(find_function_candidates)
    assert callable(parse_statement_block)
    assert callable(parse_expression)

    if not hasattr(ast_patch_parsing.cst, "parse_module"):
        return

    module = parse_python_module("class Example:\n    def run(self):\n        call(value=1)\n")
    matches = find_function_candidates(module, function_name="run", class_name="Example")
    assert len(matches) == 1
    assert matches[0].qualified_name == "Example.run"
    assert parse_statement_block("value = 1")
    assert parse_expression("1 + 2", error_kind="replacement_parse_failed")


def test_ast_patch_operations_import_and_missing_libcst_guard() -> None:
    assert callable(apply_python_ast_patch)
    if ast_patch_module.cst is not None:
        return

    result = asyncio.run(
        ast_patch_module.handle_ast_patch(
            path="src/app.py",
            operation="add_import",
            target={"module": "os"},
            state=LoopState(cwd="/tmp"),
        )
    )
    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "ast_operation_invalid"
    assert "libcst" in result["error"]


def test_verifier_success_nudge_key_is_stable_across_interpreters() -> None:
    command = "python   -m py_compile   src/app.py"
    changed_paths = ["src/app.py", "src/other.py"]
    expected = _verifier_success_nudge_key(command, changed_paths)
    script = (
        "from smallctl.graph.progress_guard import _verifier_success_nudge_key; "
        "print(_verifier_success_nudge_key('python   -m py_compile   src/app.py', ['src/other.py', 'src/app.py']))"
    )
    env = dict(os.environ)
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == expected


def test_loop_guard_env_values_are_converted_like_cli(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMALLCTL_LOOP_GUARD_ENABLED", "false")
    monkeypatch.setenv("SMALLCTL_LOOP_GUARD_CUMULATIVE_WRITE_GATE", "0")
    monkeypatch.setenv("SMALLCTL_LOOP_GUARD_STAGNATION_THRESHOLD", "7")
    monkeypatch.setenv("SMALLCTL_LOOP_GUARD_SIMILARITY_THRESHOLD", "0.75")
    monkeypatch.setenv("SMALLCTL_GRAPH_RECURSION_LIMIT", "77")
    monkeypatch.setenv("SMALLCTL_GRAPH_CODING_RECURSION_LIMIT", "88")
    monkeypatch.setenv("SMALLCTL_RUNTIME_CONTEXT_PROBE", "false")
    monkeypatch.setenv("SMALLCTL_STAGED_REASONING", "false")
    monkeypatch.setenv("SMALLCTL_SOLVER_REFINE_ENABLED", "true")
    monkeypatch.setenv("SMALLCTL_SOLVER_REFINE_MAX_PASSES", "2")
    monkeypatch.setenv("SMALLCTL_TEST_TIME_SCALING_RUNTIMES", "alpha, beta")

    config = resolve_config({})

    assert config.loop_guard_enabled is False
    assert config.loop_guard_cumulative_write_gate is False
    assert config.loop_guard_stagnation_threshold == 7
    assert config.loop_guard_similarity_threshold == 0.75
    assert config.graph_recursion_limit == 77
    assert config.graph_coding_recursion_limit == 88
    assert config.runtime_context_probe is False
    assert config.staged_reasoning is False
    assert config.staged_execution_enabled is False
    assert config.solver_refine_enabled is True
    assert config.solver_refine_max_passes == 2
    assert config.test_time_scaling_runtimes == ["alpha", "beta"]


def test_cli_and_env_numeric_conversion_use_same_schema(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMALLCTL_MAX_PROMPT_TOKENS", "1234")
    monkeypatch.setenv("SMALLCTL_SUMMARIZE_AT_RATIO", "0.65")

    env_config = resolve_config({})
    cli_config = resolve_config({"max_prompt_tokens": "1234", "summarize_at_ratio": "0.65"})

    assert env_config.max_prompt_tokens == cli_config.max_prompt_tokens == 1234
    assert env_config.summarize_at_ratio == cli_config.summarize_at_ratio == 0.65


def test_yaml_config_uses_shared_conversion_and_aliases(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".smallctl.yaml").write_text(
        "\n".join(
            [
                "backend_healthcheck_url: http://localhost:1234/health",
                "backend_restart_grace_sec: '9'",
                "solver_refine_enabled: 'true'",
                "solver_refine_max_passes: '3'",
                "test_time_scaling_runtimes: alpha, beta",
            ]
        ),
        encoding="utf-8",
    )

    config = resolve_config({})

    assert config.healthcheck_url == "http://localhost:1234/health"
    assert config.startup_grace_period_sec == 9
    assert config.solver_refine_enabled is True
    assert config.solver_refine_max_passes == 3
    assert config.test_time_scaling_runtimes == ["alpha", "beta"]


def test_env_config_mapping_covers_config_fields() -> None:
    ignored = {
        "cleanup",
        "compatibility_warnings",
        "max_prompt_tokens_explicit",
        "no_fama",
        "task",
        "tui",
    }
    field_names = {field.name for field in fields(SmallctlConfig)} - ignored

    assert field_names <= _env_config_key_names()


def test_interactive_ssh_cleanup_unregisters_process_and_cancels_collectors() -> None:
    async def _run() -> None:
        proc = _TimeoutProbeProcess()
        harness = SimpleNamespace(_active_processes={proc})

        async def _collector() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(_collector())
        session_id = "sshint-test"
        session = {"proc": proc, "tasks": [task], "stdout": [], "stderr": []}
        network._SSH_INTERACTIVE_SESSIONS[session_id] = session

        await network._cleanup_interactive_session(
            session_id,
            session,
            harness=harness,
            terminate=True,
        )

        assert session_id not in network._SSH_INTERACTIVE_SESSIONS
        assert proc not in harness._active_processes
        assert task.cancelled()
        assert proc.terminate_calls == 1

    asyncio.run(_run())


def test_truthy_plan_approval_metadata_matches_interrupt() -> None:
    state = SimpleNamespace(
        active_plan=SimpleNamespace(approved=1, plan_id="plan-1"),
        draft_plan=None,
    )
    harness = SimpleNamespace(state=state)
    interrupt = {"kind": "plan_execute_approval", "plan_id": "plan-1"}

    assert _approved_plan_matches_interrupt(harness, interrupt) is True
    assert _approved_plan_matches_plan_interrupt(state, interrupt) is True


def test_step_budget_nudge_message_matches_threshold() -> None:
    state = LoopState(cwd="/tmp")
    state.step_count = STEP_BUDGET_NUDGE_THRESHOLD + 1
    state.scratchpad["_model_name"] = "qwen2.5:7b"
    harness = SimpleNamespace(state=state, _runlog=lambda *_args, **_kwargs: None, client=None)
    graph_state = SimpleNamespace(final_result=None)

    injected = _maybe_inject_step_budget_nudge(harness, graph_state)

    assert injected is True
    assert f"more than {STEP_BUDGET_NUDGE_THRESHOLD} steps" in state.recent_messages[-1].content


def test_truncate_output_does_not_modify_short_text() -> None:
    assert truncate_output("hello") == "hello"


def test_truncate_output_truncates_long_text() -> None:
    long_text = "x" * (300 * 1024)
    result = truncate_output(long_text)
    assert len(result) < len(long_text)
    assert result.endswith("[OUTPUT TRUNCATED - TOO LARGE]")


def test_build_process_output_truncates_and_preserves_metrics() -> None:
    long_text = "x" * (300 * 1024)
    result = build_process_output(
        stdout=long_text,
        stderr="err",
        exit_code=0,
        metrics={"duration_sec": 1.25},
    )

    assert result["stdout"].endswith("[OUTPUT TRUNCATED - TOO LARGE]")
    assert result["stderr"] == "err"
    assert result["exit_code"] == 0
    assert result["metrics"] == {"duration_sec": 1.25}


def test_cancel_tasks_cancels_pending_and_ignores_done_tasks() -> None:
    async def _run() -> None:
        async def _pending() -> None:
            await asyncio.sleep(60)

        async def _done() -> str:
            return "done"

        pending_task = asyncio.create_task(_pending())
        done_task = asyncio.create_task(_done())
        await done_task

        await cancel_tasks([pending_task, done_task, object()])

        assert pending_task.cancelled()
        assert done_task.done()
        assert done_task.result() == "done"

    asyncio.run(_run())


def test_tui_and_cli_harness_kwargs_parity() -> None:
    import logging
    run_logger = logging.getLogger("test")
    config = SmallctlConfig(
        task="test task",
        planning_mode=True,
        runtime_context_probe=False,
        summarizer_endpoint="http://summary.test/v1",
        loop_guard_enabled=False,
        loop_guard_stagnation_threshold=9,
        reflexion_enabled=False,
        reflexion_max_items=7,
        subtask_ledger_enabled=False,
        subtask_max_history=4,
    )
    tui_kwargs = build_harness_config_kwargs(config, run_logger=run_logger, task=config.task)
    cli_kwargs = build_harness_config_kwargs(config, run_logger=run_logger)
    # TUI intentionally includes task; CLI omits it (passed to run_auto separately)
    assert tui_kwargs.get("task") == "test task"
    assert "task" not in cli_kwargs
    assert tui_kwargs["planning_mode"] is True
    assert cli_kwargs["planning_mode"] is True
    assert tui_kwargs["runtime_context_probe"] is False
    assert cli_kwargs["runtime_context_probe"] is False
    assert cli_kwargs["summarizer_endpoint"] == "http://summary.test/v1"
    assert cli_kwargs["loop_guard_enabled"] is False
    assert cli_kwargs["loop_guard_stagnation_threshold"] == 9
    assert cli_kwargs["reflexion_enabled"] is False
    assert cli_kwargs["reflexion_max_items"] == 7
    assert cli_kwargs["subtask_ledger_enabled"] is False
    assert cli_kwargs["subtask_max_history"] == 4
    # Verify core fields are identical when task is removed from TUI
    tui_without_task = {k: v for k, v in tui_kwargs.items() if k != "task"}
    assert tui_without_task == cli_kwargs


def test_harness_kwargs_cover_shared_config_fields() -> None:
    import logging
    shared_fields = {field.name for field in fields(SmallctlConfig)} & {field.name for field in fields(HarnessConfig)}
    kwargs = build_harness_config_kwargs(SmallctlConfig(), run_logger=logging.getLogger("test"))

    assert shared_fields <= set(kwargs)


def test_guard_fail_builds_consistent_metadata() -> None:
    result = guard_fail(
        "blocked",
        reason="test_reason",
        command="rm -rf /",
        error_kind="destructive_delete",
        next_required_tool={"tool_name": "file_read"},
        next_required_action="ask_human",
        extra_metadata={"blocked_targets": ["/"]},
    )
    assert result["success"] is False
    meta = result["metadata"]
    assert meta["reason"] == "test_reason"
    assert meta["error_kind"] == "destructive_delete"
    assert meta["next_required_tool"]["tool_name"] == "file_read"
    assert meta["next_required_action"] == "ask_human"
    assert meta["blocked_targets"] == ["/"]


def test_shell_authoring_guard_uses_consistent_metadata() -> None:
    state = LoopState(cwd="/tmp")
    state.active_plan = SimpleNamespace(approved=0, plan_id="plan-1")

    result = _shell_execution_authoring_guard(state, "python -m pytest")

    assert result is not None
    assert result["success"] is False
    assert result["metadata"] == {
        "reason": "spec_not_approved",
        "command": "python -m pytest",
        "plan_id": "plan-1",
    }


def test_shell_installer_pipe_guard_uses_consistent_metadata() -> None:
    result = _interactive_installer_yes_pipe_guard("yes | ./installfog.sh", tool_name="shell_exec")

    assert result is not None
    assert result["success"] is False
    assert result["metadata"]["reason"] == "unsafe_yes_pipe_interactive_installer"
    assert result["metadata"]["command"] == "yes | ./installfog.sh"
    assert result["metadata"]["detected_target"] == "./installfog.sh"
    assert result["metadata"]["next_required_action"]["strategy"] == "use_structured_noninteractive_install"


def test_apt_deb822_preflight_guard_uses_consistent_metadata() -> None:
    state = LoopState(cwd="/tmp")
    record_apt_update_result(
        state,
        command="apt-get update",
        success=False,
        stderr="malformed entry in list file /etc/apt/sources.list.d/debian.sources",
        host="localhost",
        user="root",
    )
    result = _apt_deb822_preflight_guard(
        "apt-get update", tool_name="shell_exec", state=state, host="localhost", user="root"
    )

    assert result is not None
    assert result["success"] is False
    assert result["metadata"]["reason"] == "apt_deb822_preflight_required"
    assert result["metadata"]["command"] == "apt-get update"
    assert result["metadata"]["required_fields"] == ["Types:", "URIs:", "Suites:", "Components:"]
    assert result["metadata"]["next_required_action"]["tool_name"] == "shell_exec"
    # One-liner validator (no heredoc) so it can be chained with &&
    validator = result["metadata"]["next_required_action"]["required_arguments"]["command"]
    assert "python3 -c" in validator
    assert "<<'PY'" not in validator


def test_apt_deb822_preflight_allows_after_session_validation() -> None:
    state = LoopState(cwd="/tmp")
    record_apt_update_result(
        state,
        command="apt-get update",
        success=False,
        stderr="malformed entry in list file /etc/apt/sources.list.d/debian.sources",
        host="localhost",
        user="root",
    )
    # Initially blocked because apt update failed
    result = _apt_deb822_preflight_guard(
        "apt-get install foo",
        tool_name="shell_exec",
        state=state,
        host="localhost",
        user="root",
    )
    assert result is not None
    assert result["success"] is False

    # Mark clean
    _mark_deb822_preflight_clean(state, host="localhost", user="root")

    # Now allowed
    result2 = _apt_deb822_preflight_guard(
        "apt-get install foo",
        tool_name="shell_exec",
        state=state,
        host="localhost",
        user="root",
    )
    assert result2 is None


def test_apt_deb822_preflight_respects_host_user_isolation() -> None:
    state = LoopState(cwd="/tmp")
    record_apt_update_result(
        state,
        command="apt-get update",
        success=False,
        stderr="malformed entry in list file /etc/apt/sources.list.d/debian.sources",
        host="host-a",
        user="root",
    )
    record_apt_update_result(
        state,
        command="apt-get update",
        success=False,
        stderr="malformed entry in list file /etc/apt/sources.list.d/debian.sources",
        host="host-b",
        user="root",
    )
    record_apt_update_result(
        state,
        command="apt-get update",
        success=False,
        stderr="malformed entry in list file /etc/apt/sources.list.d/debian.sources",
        host="host-a",
        user="other",
    )
    _mark_deb822_preflight_clean(state, host="host-a", user="root")

    # Different host still blocked
    result = _apt_deb822_preflight_guard(
        "apt-get install foo",
        tool_name="ssh_exec",
        state=state,
        host="host-b",
        user="root",
    )
    assert result is not None
    assert result["success"] is False

    # Different user still blocked
    result2 = _apt_deb822_preflight_guard(
        "apt-get install foo",
        tool_name="ssh_exec",
        state=state,
        host="host-a",
        user="other",
    )
    assert result2 is not None
    assert result2["success"] is False


def test_looks_like_deb822_validator() -> None:
    assert _looks_like_deb822_validator(
        "python3 -c \"from pathlib import Path; p = Path('/etc/apt/sources.list.d/debian.sources'); print('deb822 OK')\""
    )
    assert not _looks_like_deb822_validator("apt-get update")
    assert not _looks_like_deb822_validator("python3 -c 'print(1)'")


def test_validate_sources_file_with_debian_deb822() -> None:
    valid = """Types: deb
URIs: http://deb.debian.org/debian
Suites: stable
Components: main
"""
    result = validate_sources_file(valid)
    assert result["valid"] is True

    missing_fields = """Types: deb
URIs: http://deb.debian.org/debian
"""
    result2 = validate_sources_file(missing_fields)
    assert result2["valid"] is False
    assert "missing" in result2["error"].lower()
    assert "Suites:" in result2.get("missing_fields", [])
    assert "Components:" in result2.get("missing_fields", [])


def test_remote_installer_preflight_guard_uses_consistent_metadata() -> None:
    state = LoopState(cwd="/tmp/project")
    result = _remote_installer_preflight_guard(
        "cd /tmp/project && ./installfog.sh",
        host="example.test",
        user="root",
        state=state,
    )

    assert result is not None
    assert result["success"] is False
    assert result["metadata"]["reason"] == "remote_installer_preflight_required"
    assert result["metadata"]["command"] == "cd /tmp/project && ./installfog.sh"
    assert result["metadata"]["host"] == "example.test"
    assert result["metadata"]["user"] == "root"
    assert result["metadata"]["cwd"] == "/tmp/project"
    assert result["metadata"]["script_path"] == "/tmp/project/installfog.sh"
    assert result["metadata"]["next_required_action"]


def test_shell_foreground_guard_uses_consistent_metadata() -> None:
    result = _foreground_command_guard("npm run dev", tool_name="shell_exec", allow_background_parameter=True)

    assert result is not None
    assert result["success"] is False
    assert result["metadata"]["reason"] == "long_running_foreground_command"
    assert result["metadata"]["command"] == "npm run dev"
    assert result["metadata"]["foreground_detection"] == "package_runner_foreground"
    assert result["metadata"]["next_required_action"]["strategy"] == "detach_or_bound_then_verify"


def test_loop_guard_decision_and_emit_block() -> None:
    state = LoopState(cwd="/tmp")
    path_state: dict[str, Any] = {"blocked_attempts": 0, "escalation_level": 0}
    result = _emit_block(
        state,
        path_state,
        LoopGuardDecision(
            action="block",
            message="test block",
            error_kind="test_error",
        ),
        resolved_path="/tmp/test.py",
        session_id="s1",
        section_name="sec1",
        next_section_name="sec2",
        score=3,
        signals={"hash_stagnation": True},
        tail_excerpt="tail",
        level=2,
        outline_required=True,
    )
    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "test_error"
    assert path_state["escalation_level"] == 2
    assert path_state["blocked_attempts"] == 1
    assert path_state["outline_required"] is True


def test_task_classification_rules_table_preserves_precedence() -> None:
    from smallctl.harness.task_classifier import _TASK_CLASSIFICATION_RULES, classify_task_mode

    # Precedence: local_execute > hybrid_execute > chat > plan_only > remote_execute > debug_inspect > analysis
    assert _TASK_CLASSIFICATION_RULES[0].mode == "local_execute"
    assert _TASK_CLASSIFICATION_RULES[1].mode == "local_execute"
    assert _TASK_CLASSIFICATION_RULES[2].mode == "local_execute"
    assert _TASK_CLASSIFICATION_RULES[3].mode == "hybrid_execute"
    assert _TASK_CLASSIFICATION_RULES[4].mode == "local_execute"
    assert _TASK_CLASSIFICATION_RULES[5].mode == "chat"
    assert _TASK_CLASSIFICATION_RULES[6].mode == "plan_only"
    assert _TASK_CLASSIFICATION_RULES[7].mode == "remote_execute"
    assert _TASK_CLASSIFICATION_RULES[8].mode == "local_execute"
    assert _TASK_CLASSIFICATION_RULES[9].mode == "debug_inspect"
    assert _TASK_CLASSIFICATION_RULES[10].mode == "local_execute"
    assert _TASK_CLASSIFICATION_RULES[11].mode == "analysis"
    # Default fallback
    assert classify_task_mode("") == "chat"
    assert classify_task_mode("hi") == "chat"


def test_resolve_fetch_selector_prefers_single_argument() -> None:
    url, rid, fid, warnings = _resolve_fetch_selector(None, url="http://example.com", result_id=None, fetch_id=None)
    assert url == "http://example.com"
    assert not warnings


def test_resolve_fetch_selector_resolves_multiple_arguments() -> None:
    url, rid, fid, warnings = _resolve_fetch_selector(None, url="http://a.com", result_id="r1", fetch_id="f1")
    assert fid == "f1"
    assert url is None
    assert rid is None
    assert any("fetch_id" in w for w in warnings)


def test_web_artifact_ref_resolves_search_result_alias() -> None:
    artifact = SimpleNamespace(
        artifact_id="A1",
        preview_text=(
            "1. Example\n"
            "Result ID: r1\n"
            "Fetch ID: f1\n"
            "URL: https://example.com/a\n"
            "Domain: example.com\n"
        ),
        inline_content="",
        content_path="",
        metadata={},
    )
    state = SimpleNamespace(artifacts={"A1": artifact}, scratchpad={})

    entry, metadata = resolve_search_result_from_artifact_reference(state, "A1-1")

    assert entry is not None
    assert entry["result_id"] == "r1"
    assert entry["fetch_id"] == "f1"
    assert entry["canonical_url"] == "https://example.com/a"
    assert metadata["argument_repair"] == "web_fetch_result_alias_to_search_result"
    assert metadata["resolved_search_artifact_id"] == "A1"


def test_web_budget_helpers_track_usage_and_terminal_exhaustion() -> None:
    state = SimpleNamespace(scratchpad={})
    config = SearchServerConfig(
        max_searches_per_run=2,
        max_fetches_per_run=1,
        max_total_fetched_chars=5,
    )

    ensure_budget(state, config=config, action="search")
    ensure_budget(state, config=config, action="fetch")
    ensure_budget(state, config=config, action="fetch_chars", chars=3)

    remaining = budget_remaining(state, config)
    assert remaining == {
        "searches_remaining": 1,
        "fetches_remaining": 0,
        "chars_remaining": 2,
    }

    mark_web_fetch_budget_exhausted(state, "Web fetch budget exhausted for this run")
    assert state.scratchpad["_web_fetch_budget_exhausted"]["terminal"] is True


def test_web_result_index_helpers_track_fetch_ids_and_artifacts() -> None:
    state = SimpleNamespace(scratchpad={}, artifacts={})

    assigned = assign_fetch_ids(
        state,
        [{"result_id": "r10", "url": "https://example.com/a"}],
    )
    assert assigned[0]["fetch_id"] == "r1"

    update_result_index(state, assigned)
    index = load_result_index(state)
    assert index["r10"]["canonical_result_id"] == "r10"
    assert index["r1"]["canonical_result_id"] == "r10"

    record_fetch_artifact_mapping(state, "r10", "r1", "A1")
    assert load_fetch_artifact_index(state)["r10"] == "A1"
    assert load_fetch_artifact_index(state)["r1"] == "A1"

    message, metadata = unknown_result_id_error(state, "missing")
    assert metadata["valid_fetch_ids"] == ["r1"]
    assert "Valid fetch IDs" in message


def test_web_fetch_artifact_persistence_preserves_body_metadata() -> None:
    harness = SimpleNamespace(artifact_store=_FakeArtifactStore())
    state = SimpleNamespace(thread_id="thread-1", artifacts={})
    response = SimpleNamespace(
        source_id="src-1",
        url="https://example.com/a",
        canonical_url="https://example.com/a",
        domain="example.com",
        title="Example",
        byline="",
        content_type="text/html",
        content_sha256="abc123",
        published_at="",
        fetched_at="now",
        text_excerpt="excerpt",
        untrusted_text="",
    )
    citation = SimpleNamespace(provider="provider", extractor="extractor")

    artifact_id = persist_fetch_artifact(
        harness,
        state,
        response,
        "full body\nline two",
        citation,
        "r1",
        {"fetch_id": "f1", "_bounded_max_chars": 10},
    )

    assert artifact_id == "A1"
    artifact = state.artifacts["A1"]
    assert artifact.metadata["render_mode"] == "body_with_preview"
    assert artifact.metadata["body_total_lines"] == 2
    assert artifact.metadata["result_id"] == "r1"
    assert artifact.metadata["fetch_id"] == "f1"


def test_dispatcher_artifact_read_routes_file_path_to_file_read() -> None:
    tool_name, arguments, metadata = normalize_artifact_read_request(
        {"artifact_id": "src/app.py", "start_line": 3, "max_chars": 1000}
    )

    assert tool_name == "file_read"
    assert arguments == {"path": "src/app.py", "start_line": 3, "max_bytes": 1000}
    assert metadata["routing_reason"] == "artifact_id_to_file_read"


def test_dispatcher_artifact_read_uses_recent_artifact_fallback() -> None:
    state = SimpleNamespace(artifacts={"A1": object(), "A2": object()}, retrieval_cache=["A1"])

    tool_name, arguments, metadata = normalize_artifact_read_request({"start_line": 1}, state=state)

    assert tool_name == "artifact_read"
    assert arguments["artifact_id"] == "A1"
    assert metadata["argument_repair"] == "artifact_read_recent_fallback"


def test_dispatcher_web_fetch_normalizes_fetch_id_and_result_alias() -> None:
    state = SimpleNamespace(
        scratchpad={
            "_web_result_index": {"r1": {"url": "http://example.com"}},
            "_web_last_search_result_ids": ["r1"],
            "_web_last_search_artifact_id": "A1",
        }
    )

    arguments, metadata = normalize_web_fetch_request({"fetch_id": "result 1"}, state=state)

    assert arguments == {"result_id": "r1"}
    assert metadata["field_alias_repair"] == "web_fetch_fetch_id_to_result_id"
    assert metadata["argument_repair"] == "web_fetch_result_alias_to_search_result"


def test_initial_tool_request_normalization_repairs_aliases_and_none_sentinels() -> None:
    tool_name, arguments, metadata = normalize_initial_tool_request(
        "artifact_write",
        {
            "path": "app.py",
            "content": "print('ok')",
            "write_session_id": "none",
            "next_section_name": "null",
        },
    )

    assert tool_name == "file_write"
    assert arguments == {"path": "app.py", "content": "print('ok')"}
    assert metadata["repaired_tool_alias_from"] == "artifact_write"
    assert metadata["repaired_tool_alias_to"] == "file_write"
    assert metadata["routing_reason"] == "tool_alias_repair"
    assert metadata["optional_none_sentinel_removed"] == ["next_section_name", "write_session_id"]


def test_initial_tool_request_normalization_repairs_patch_aliases() -> None:
    tool_name, arguments, metadata = normalize_initial_tool_request(
        "file_patch",
        {"path": "app.py", "old": "before", "new_text": "after"},
    )

    assert tool_name == "file_patch"
    assert arguments == {"path": "app.py", "target_text": "before", "replacement_text": "after"}
    assert metadata["argument_alias_repair"] == {"old": "target_text", "new_text": "replacement_text"}


def test_repair_ssh_exec_malformed_args_unwraps_nested_command_and_name() -> None:
    arguments, metadata = repair_ssh_exec_malformed_args(
        {
            "host": "example.test",
            "name": "whoami",
            "arguments": {"arg": "id"},
        }
    )

    assert arguments == {"host": "example.test", "command": "id"}
    assert metadata == {
        "repaired_ssh_exec_nested_args": True,
        "repaired_ssh_exec_hallucinated_name": True,
        "routing_reason": "ssh_exec_malformed_args_repair",
    }


def test_repair_ssh_exec_malformed_args_preserves_explicit_command() -> None:
    arguments, metadata = repair_ssh_exec_malformed_args(
        {
            "host": "example.test",
            "command": "whoami",
            "name": "ssh_exec",
            "arguments": {"arg": "id"},
        }
    )

    assert arguments == {"host": "example.test", "command": "whoami", "name": "ssh_exec"}
    assert metadata == {
        "repaired_ssh_exec_nested_args": True,
        "routing_reason": "ssh_exec_malformed_args_repair",
    }


def test_repair_write_session_path_from_state_uses_active_session_target() -> None:
    state = SimpleNamespace(
        write_session=SimpleNamespace(
            status="open",
            write_session_id="ws-1",
            write_target_path="src/app.py",
        )
    )

    arguments, metadata = repair_write_session_path_from_state(
        "file_patch",
        {"write_session_id": "ws-1", "target_text": "a", "replacement_text": "b"},
        state=state,
    )

    assert arguments["path"] == "src/app.py"
    assert metadata == {
        "argument_repair": "active_write_session_path",
        "repaired_write_session_path": True,
        "write_session_id": "ws-1",
        "target_path": "src/app.py",
    }


def test_dispatcher_shell_guard_blocks_harness_tool_inside_ssh_exec() -> None:
    result = guard_harness_tool_as_ssh_shell_command("file_read {'path': 'app.py'}")

    assert result is not None
    assert result.success is False
    assert result.metadata == {
        "tool_name": "ssh_exec",
        "reason": "harness_tool_as_remote_shell_command",
        "suggested_tool": "file_read",
    }


def test_dispatcher_shell_guard_blocks_nested_raw_ssh() -> None:
    assert looks_like_raw_ssh_shell_command("ssh root@example.test whoami") is True
    assert looks_like_raw_ssh_shell_command("ssh-keygen -R 192.168.1.161 -f ~/.ssh/known_hosts") is False
    result = guard_nested_raw_ssh_in_ssh_exec("ssh root@example.test whoami")

    assert result is not None
    assert result.success is False
    assert result.metadata["reason"] == "nested_raw_ssh_in_ssh_exec"
    assert result.metadata["suggested_command"] == "whoami"


def test_dispatcher_shell_guard_raw_ssh_block_metadata() -> None:
    result = raw_ssh_shell_block_envelope("scp a b", ssh_available=False)

    assert result.success is False
    assert result.metadata["reason"] == "raw_ssh_shell_blocked"
    assert result.metadata["command"] == "scp a b"
    assert "ssh_exec" in result.metadata["suggested_tools"]


def test_dispatcher_remote_path_detectors_exclude_local_cwd_and_tmp() -> None:
    state = SimpleNamespace(cwd="/home/stephen/project")

    assert looks_like_remote_absolute_path("/etc/nginx/nginx.conf", state=state) is True
    assert looks_like_remote_absolute_path("/home/stephen/project/app.py", state=state) is False
    assert looks_like_remote_absolute_path("/tmp/install.log", state=state) is False


def test_dispatcher_remote_path_detectors_find_shell_mentions_and_infra_probes() -> None:
    state = SimpleNamespace(cwd="/home/stephen/project")

    assert command_mentions_remote_absolute_path("test -f /etc/nginx/nginx.conf", state=state) is True
    assert command_mentions_remote_absolute_path("test -f /home/stephen/project/app.py", state=state) is False
    assert looks_like_remote_infrastructure_probe_command("systemctl status nginx") is True
    assert looks_like_remote_infrastructure_probe_command("python -m pytest") is False


def test_dispatcher_ssh_auth_helpers_are_stable() -> None:
    assert ssh_auth_recovery_entry_key(" Example.TEST ", " Root ") == "root@example.test"
    assert ssh_auth_recovery_entry_key(" Example.TEST ", "") == "example.test"
    assert password_fingerprint(" secret ") == password_fingerprint("secret")
    assert len(password_fingerprint("secret")) == 16
    assert password_fingerprint("") == ""


def test_dispatcher_ssh_auth_debug_metadata_reports_transport() -> None:
    password_meta = ssh_auth_debug_metadata(
        {"password": "secret", "identity_file": "/tmp/key"},
        password_source="task_context",
    )
    key_meta = ssh_auth_debug_metadata({"identity_file": "/tmp/key"}, password_source="")

    assert password_meta == {
        "ssh_auth_mode": "password",
        "ssh_auth_transport": "sshpass_env",
        "ssh_password_origin": "task_context",
        "ssh_password_recovered": True,
        "ssh_identity_file_supplied": True,
    }
    assert key_meta["ssh_auth_mode"] == "key"
    assert key_meta["ssh_auth_transport"] == "ssh"
    assert key_meta["ssh_password_origin"] == "none"
    assert key_meta["ssh_password_recovered"] is False


def test_dispatcher_ssh_context_helpers_extract_user_and_password() -> None:
    state = SimpleNamespace(
        run_brief=SimpleNamespace(original_task="SSH to root@example.test password is secret"),
        working_memory=SimpleNamespace(current_goal=""),
        recent_messages=[],
    )

    assert ssh_task_context_texts(state) == ["SSH to root@example.test password is secret"]
    assert infer_ssh_user_from_state_context("example.test", state=state) == "root"
    assert infer_ssh_password_from_state_context("example.test", user="root", state=state) == "secret"
    assert text_mentions_ssh_target("connect as root@example.test", host="example.test", user="root") is True


def test_dispatcher_ssh_context_helpers_filter_password_tokens() -> None:
    state = SimpleNamespace(
        run_brief=SimpleNamespace(original_task="host example.test user root password required"),
        working_memory=SimpleNamespace(current_goal=""),
        recent_messages=[],
    )

    assert looks_like_ssh_password("secret") is True
    assert looks_like_ssh_password("required") is False
    assert infer_ssh_password_from_state_context("example.test", user="root", state=state) == ""


def test_dispatcher_ssh_memory_helpers_infer_from_records_and_session() -> None:
    state = SimpleNamespace(
        tool_execution_records={
            "1": {
                "tool_name": "ssh_exec",
                "args": {"host": "example.test", "user": "root", "password": "secret"},
                "result": {"success": True},
            }
        },
        scratchpad={
            "_session_ssh_targets": {
                "example.test": {"user": "admin", "password": "session-secret"},
            }
        },
    )

    assert ssh_record_likely_authenticated({"result": {"success": True}}) is True
    assert infer_ssh_user_from_execution_records("example.test", state=state) == "root"
    assert infer_ssh_password_from_execution_records("example.test", user="root", state=state) == "secret"
    assert session_ssh_target_record("EXAMPLE.test", state=state)["user"] == "admin"
    assert infer_ssh_user_from_session_memory("example.test", state=state) == "admin"
    assert infer_ssh_password_from_session_memory("example.test", user="admin", state=state) == "session-secret"


def test_dispatcher_ssh_memory_password_precedence_prefers_records_then_context() -> None:
    record_state = SimpleNamespace(
        tool_execution_records={
            "1": {
                "tool_name": "ssh_exec",
                "args": {"host": "example.test", "user": "root", "password": "record-secret"},
                "result": {"success": True},
            }
        },
        scratchpad={"_session_ssh_targets": {"example.test": {"user": "root", "password": "session-secret"}}},
        run_brief=SimpleNamespace(original_task="ssh to root@example.test password is task-secret"),
        working_memory=SimpleNamespace(current_goal=""),
        recent_messages=[],
    )
    context_state = SimpleNamespace(
        tool_execution_records={},
        scratchpad={"_session_ssh_targets": {"example.test": {"user": "root", "password": "session-secret"}}},
        run_brief=SimpleNamespace(original_task="ssh to root@example.test password is task-secret"),
        working_memory=SimpleNamespace(current_goal=""),
        recent_messages=[],
    )

    assert infer_ssh_password("example.test", user="root", state=record_state) == ("record-secret", "prior_ssh_exec")
    assert infer_ssh_password("example.test", user="root", state=context_state) == ("task-secret", "task_context")


def test_normalize_tool_request_routes_artifact_file_path() -> None:
    tool_name, arguments, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "artifact_read",
        {"artifact_id": "src/app.py", "max_chars": 1000},
        phase="execute",
        state=LoopState(cwd="."),
    )

    assert intercepted is None
    assert tool_name == "file_read"
    assert arguments == {"path": "src/app.py", "max_bytes": 1000}
    assert metadata["routing_reason"] == "artifact_id_to_file_read"


def test_normalize_tool_request_repairs_web_fetch_alias() -> None:
    state = LoopState(cwd=".")
    state.scratchpad["_web_result_index"] = {"webres-test-1": {"url": "https://example.com/one"}}
    state.scratchpad["_web_last_search_result_ids"] = ["webres-test-1"]
    state.scratchpad["_web_last_search_artifact_id"] = "A1"

    tool_name, arguments, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "web_fetch",
        {"fetch_id": "result 1"},
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "web_fetch"
    assert arguments == {"result_id": "webres-test-1"}
    assert metadata["field_alias_repair"] == "web_fetch_fetch_id_to_result_id"
    assert metadata["argument_repair"] == "web_fetch_result_alias_to_search_result"


def test_schema_type_matches_respects_boolean_exclusion() -> None:
    from smallctl.tools.dispatcher_schema_helpers import type_matches

    assert type_matches("number", 42) is True
    assert type_matches("number", True) is False
    assert type_matches("integer", 42) is True
    assert type_matches("integer", True) is False
    assert type_matches("boolean", True) is True
    assert type_matches("string", "hello") is True
    assert type_matches("array", []) is True
    assert type_matches("object", {}) is True
    assert type_matches("unknown", "anything") is True


def test_schema_coerce_value_converts_strings_and_numbers() -> None:
    from smallctl.tools.dispatcher_schema_helpers import coerce_value

    assert coerce_value("boolean", "true") is True
    assert coerce_value("boolean", "false") is False
    assert coerce_value("boolean", "maybe") == "maybe"
    assert coerce_value("string", 42) == "42"
    assert coerce_value("string", True) == "true"
    assert coerce_value("integer", "42") == 42
    assert coerce_value("integer", "-3") == -3
    assert coerce_value("integer", "not_a_number") == "not_a_number"
    assert coerce_value("number", "3.14") == 3.14
    assert coerce_value("number", "invalid") == "invalid"
    assert coerce_value("unknown", "keep") == "keep"


def test_network_ssh_helpers_classify_failures_and_build_command() -> None:
    from smallctl.tools.network_ssh_helpers import (
        build_ssh_command,
        ssh_diagnostic_not_found,
        ssh_error_class,
        ssh_failure_kind,
    )

    cmd, env = build_ssh_command(
        host="example.test",
        command="whoami",
        user="root",
        port=22,
        identity_file=None,
        password=None,
    )
    assert cmd.startswith("ssh")
    assert "root@example.test" in cmd
    assert env is None

    assert ssh_failure_kind(exit_code=1, stderr="permission denied") == "transport"
    assert ssh_failure_kind(exit_code=1, stderr="hello") == "remote_command"
    assert ssh_error_class(exit_code=1, stderr="permission denied") == "auth_permission_denied"
    assert ssh_error_class(exit_code=1, stderr="could not resolve hostname") == "dns_resolution"
    assert ssh_diagnostic_not_found("test", {"stdout": "", "stderr": "command not found"}) is True
    assert ssh_diagnostic_not_found("test", {"stdout": "ok", "stderr": ""}) is False


def test_control_objectives_extract_and_match() -> None:
    from smallctl.tools.control_objectives import (
        extract_multi_objectives,
        objective_matches_text,
        objective_tokens,
    )

    assert extract_multi_objectives("fix issues:\n- critical: bug A\n- high: bug B") == [
        "critical: bug A",
        "high: bug B",
    ]
    assert objective_tokens("handle missing import") == {"import"}
    assert objective_matches_text({"title": "fix bug"}, "need to fix the bug") is False
    assert objective_matches_text({"title": "refactor parser"}, "refactor the parser logic") is True


def test_control_objective_ledger_tracks_remaining_objectives() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Fix these items:\n- refactor parser\n- update tests"

    ledger = ensure_multi_objective_ledger(state)
    assert ledger is not None
    assert [item["objective_id"] for item in ledger["objectives"]] == ["O1", "O2"]

    block = multi_objective_completion_block(
        state,
        message="refactor the parser logic",
        verifier_verdict={"verdict": "pass"},
    )

    assert block is not None
    assert block["completed_now"] == ["O1"]
    assert block["remaining_objectives"] == [{"objective_id": "O2", "title": "update tests"}]


def test_control_remote_mutation_block_requires_binary_presence_probe() -> None:
    requirement = {
        "host": "root@example.com",
        "mutation_type": "write",
        "guessed_paths": ["/etc/demo.key"],
        "verified_paths": [],
    }
    state = SimpleNamespace(scratchpad={"_remote_mutation_requires_verification": requirement})

    assert remote_mutation_verification_requirement(state) is requirement
    payload = remote_mutation_block_payload(requirement)
    action = payload["next_required_action"]

    assert action["tool_names"] == ["ssh_exec"]
    assert action["required_arguments"]["target"] == "root@example.com"
    assert "sha256sum" in action["required_arguments"]["command"]
    assert "Next required verifier" in payload["error"]


def test_control_verifier_helpers_normalize_stale_verdict_and_approval() -> None:
    state = LoopState(cwd="/tmp")
    state.last_verifier_verdict = {
        "verdict": "fail",
        "command": "pytest tests/test_app.py",
        "key_stderr": "assertion failed",
    }
    state.scratchpad["_last_verifier_stale_after_mutation"] = {
        "paths": ["src/app.py"],
        "tool_name": "file_write",
    }

    verdict = normalized_verifier_verdict(state)

    assert verdict is not None
    assert verdict["stale"] is True
    assert verdict["stale_after_paths"] == ["src/app.py"]
    assert verdict["next_required_action"]["tool_name"] == "shell_exec"
    assert "check=pytest tests/test_app.py" in verifier_failure_summary(verdict)
    assert verifier_requires_human_approval({"approval_denied": True}) is True


def test_control_weather_helpers_distinguish_meta_search_from_answer() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "What is the weather in Paris today?"

    assert is_weather_lookup_task(state) is True
    assert looks_like_weather_search_meta_completion("Web search completed and returned 3 results.") is True
    assert has_specific_weather_answer("Currently 72 degrees and sunny.") is True
    assert looks_like_weather_search_meta_completion("Currently 72 degrees and sunny.") is False


def test_control_plan_subtask_block_surfaces_blocked_subtask() -> None:
    state = LoopState(cwd="/tmp")
    state.active_plan = _FakePlan()
    state.subtask_ledger = SimpleNamespace(
        subtasks=[
            SimpleNamespace(
                subtask_id="S1",
                title="Investigate failure",
                goal="Find root cause",
                status="blocked",
                acceptance=["root cause known"],
                evidence=["trace"],
                blockers=["missing credential"],
                next_action="ask user",
                attempts=2,
            )
        ]
    )

    block = plan_subtask_completion_block(state, verifier_verdict={"verdict": "pass"})

    assert block is not None
    assert block["next_required_subtask"]["subtask_id"] == "S1"
    assert block["next_required_action"]["tool_names"] == ["escalate_to_bigger_model", "ask_human", "task_fail"]
    assert block["open_plan_subtasks"][0]["blockers"] == ["missing credential"]


def test_control_post_change_helpers_build_focused_verifier_and_dependency_block() -> None:
    state = LoopState(cwd="/tmp")

    assert focused_verifier_command_for_path("src/app.py", state=state) == "python3 -m py_compile src/app.py"

    state.challenge_progress.code_change_count = 1
    state.challenge_progress.verified_after_last_change = False
    state.challenge_progress.last_code_change_paths = ["src/app.py"]
    block = post_change_verification_block(state)
    assert block is not None
    assert block["reason"] == "post_change_verification_required"
    assert block["next_required_action"]["required_arguments"]["command"] == "python3 -m py_compile src/app.py"

    state.last_verifier_verdict = {
        "verdict": "fail",
        "key_stderr": "ModuleNotFoundError: No module named 'demo_dep'",
    }
    dependency = missing_dependency_block(state)
    assert dependency is not None
    assert dependency["module"] == "demo_dep"
    assert dependency["command"] == "python3 -m pip install demo_dep"


def test_control_phase_gates_block_zero_change_and_weak_verifier() -> None:
    state = LoopState(cwd="/tmp")
    state.challenge_progress.task_category = "coding"
    state.run_brief.original_task = "Implement phase 2 parser work"

    mutation_block = mutation_expectation_block(state, message="Phase 2 complete")
    assert mutation_block is not None
    assert mutation_block["reason"] == "mutation_expected_but_no_code_changes"

    phase_block = phase_promotion_gate_block(
        state,
        message="Phase 2 complete",
        verifier_verdict={"verdict": "pass", "command": "python -m py_compile app.py"},
    )
    assert phase_block is not None
    assert phase_block["reason"] == "phase_promotion_behavioral_verifier_required"

    state.run_brief.original_task = "Build a pygame game loop"
    assert task_involves_interactive_program(state) is True


def test_control_phase_contract_helpers_normalize_and_validate() -> None:
    contract = {
        "active_phase": "phase_1",
        "phases": {
            "phase_1": {
                "name": "Parser",
                "status": "active",
                "promotion": "syntax",
                "expected_files": ["src/app.py"],
                "checks": ["python3 -m py_compile app.py", "python -c 'assert game.move()'"],
            }
        },
    }

    normalized = normalize_phase_contract_payload(contract)
    phase = normalized["phases"]["phase_1"]
    assert phase["title"] == "Parser"
    assert phase["promotion"] == {"required_quality": "syntax"}
    assert phase["checks"][0]["command"] == "python3 -m py_compile src/app.py"
    assert phase["checks"][0]["quality"] == "syntax"
    assert phase["checks"][1]["quality"] == "behavioral"
    assert phase_contract_validation_error(normalized) == ""

    invalid = {"phases": {"phase_1": {"status": "active", "checks": "pytest"}}}
    assert phase_contract_validation_error(invalid) == "Phase `phase_1` field `checks` must be a list when provided."
    assert phase_check_quality_from_command("python -m unittest tests/test_app.py") == "behavioral"


def test_control_write_session_helpers_build_resume_actions_and_warning() -> None:
    session = SimpleNamespace(
        status="open",
        write_sections_completed=False,
        write_next_section="body",
        write_current_section="imports",
        write_target_path="src/app.py",
        write_session_id="ws-1",
    )
    state = SimpleNamespace(
        write_session=session,
        scratchpad={
            "_last_write_session_schema_failure": {
                "target_path": "src/app.py",
                "recommended_section_name": "body",
                "required_fields": ["content"],
            }
        },
    )

    failure = write_session_schema_failure(state)
    action = write_session_resume_action(state, failure)
    assert action is not None
    assert action["tool_name"] == "file_write"
    assert action["required_arguments"] == {
        "path": "src/app.py",
        "section_name": "body",
    }
    assert "Last schema failure was missing: content" in action["notes"]
    assert "Next expected section: `body`" in str(write_session_warning(state))

    session.write_sections_completed = True
    session.write_next_section = ""
    finalize_action = write_session_resume_action(state, None)
    assert finalize_action is not None
    assert finalize_action["tool_name"] == "finalize_write_session"


def test_control_loop_status_helper_shapes_subtask_ledger() -> None:
    subtasks = [
        SimpleNamespace(
            subtask_id=f"S{i}",
            title=f"Task {i}",
            goal="goal",
            status="done" if i == 1 else "pending",
            acceptance=["ok"],
            evidence=["old", "mid", "new", "latest"],
            blockers=["b1", "b2", "b3", "b4"],
            next_action="continue",
            attempts=i,
        )
        for i in range(1, 15)
    ]
    state = SimpleNamespace(
        subtask_ledger=SimpleNamespace(
            task_id="root",
            active_subtask_id="S14",
            subtasks=subtasks,
        )
    )

    status = subtask_ledger_status(state)
    assert status is not None
    assert status["task_id"] == "root"
    assert status["active_subtask"]["subtask_id"] == "S14"
    assert status["done_subtask_ids"] == ["S1"]
    assert "S14" in status["pending_subtask_ids"]
    assert len(status["subtasks"]) == 12
    assert status["subtasks"][-1]["evidence"] == ["mid", "new", "latest"]
    assert status["subtasks"][-1]["blockers"] == ["b2", "b3", "b4"]


def test_control_loop_status_helpers_shape_progress_and_write_session_payload(monkeypatch) -> None:
    session = SimpleNamespace(
        write_target_path="src/app.py",
        to_dict=lambda: {"write_session_id": "ws-1", "status": "open"},
    )
    state = SimpleNamespace(
        scratchpad={"_max_steps": "8"},
        step_count=3,
        write_session=session,
    )
    monkeypatch.setattr(
        "smallctl.tools.control_loop_status_helpers.write_session_contract",
        lambda _session: {"target_path": "src/app.py"},
    )

    max_steps, progress = max_steps_progress(state)
    assert max_steps == 8
    assert progress == 0.375

    payload = write_session_status_payload(
        state,
        schema_failure={"required_fields": ["content"]},
        resume_action={"tool_name": "file_write"},
    )
    assert payload == {
        "write_session_id": "ws-1",
        "status": "open",
        "last_schema_failure": {"required_fields": ["content"]},
        "resume_action": {"tool_name": "file_write"},
        "contract": {"target_path": "src/app.py"},
    }

    state.scratchpad["_max_steps"] = "not-int"
    assert max_steps_progress(state) == (0, 0.0)


def test_fs_loop_guard_status_helpers_filter_paths_and_reads() -> None:
    root = {
        "paths": {
            "/tmp/inactive.py": {"pending_read_before_write": False, "escalation_level": 0, "recent_writes": []},
            "/tmp/active.py": {
                "pending_read_before_write": True,
                "escalation_level": 2,
                "blocked_attempts": 3,
                "last_score": 5,
                "last_section_name": "body",
                "last_next_section_name": "tests",
                "outline_required": True,
                "section_checkpoints": ["imports"],
            },
        }
    }
    assert active_loop_guard_paths(root) == [
        {
            "path": "/tmp/active.py",
            "escalation_level": 2,
            "pending_read_before_write": True,
            "blocked_attempts": 3,
            "last_score": 5,
            "last_section_name": "body",
            "last_next_section_name": "tests",
            "outline_required": True,
            "section_checkpoints": ["imports"],
        }
    ]

    state = SimpleNamespace(
        scratchpad={
            "_progress_read_history": [
                {"tool_name": "file_read", "path": "a.py", "complete_file": True, "total_lines": 10},
                {"tool_name": "file_read", "path": "b.py", "complete_file": True, "file_content_truncated": True},
                {"tool_name": "artifact_read", "artifact_id": "A1", "complete_file": True, "line_start": 1},
                {"tool_name": "file_read", "path": "a.py", "complete_file": True, "line_end": 10},
            ]
        }
    )
    reads = recent_complete_reads(state)
    assert [item["path"] or item["artifact_id"] for item in reads] == ["a.py", "A1"]
    assert reads[0]["line_end"] == 10


def test_client_transport_helpers_shape_retry_audit_and_diagnostics() -> None:
    response = SimpleNamespace(headers={"Retry-After": "2.5"})
    assert parse_retry_after_seconds(response) == 2.5
    assert parse_retry_after_seconds(SimpleNamespace(headers={"Retry-After": "-1"})) == 0.0
    assert parse_retry_after_seconds(SimpleNamespace(headers={})) is None

    client = SimpleNamespace(
        first_token_timeout_sec=10.0,
        provider_profile="lmstudio",
        is_small_model=False,
        model="gemma-3",
    )
    assert request_first_token_timeout_sec(client, [{"function": {"name": "x"}}]) == 60.0
    client.model = "other"
    assert request_first_token_timeout_sec(client, [{"function": {"name": str(i)}} for i in range(12)]) == 60.0

    assert provider_root("https://example.test/v1") == "https://example.test"
    assert extract_available_tool_names([{"function": {"name": "shell_exec"}}, {"bad": True}]) == {"shell_exec"}
    assert tool_name({"function": {"name": "file_read"}}) == "file_read"

    details = llamacpp_model_unloaded_details(
        SimpleNamespace(provider_profile="llamacpp", model="local"),
        {"error": "unloaded"},
        attempt=2,
        recovery={"action": "retry"},
    )
    assert details["reason"] == "model_unloaded"
    assert details["attempt"] == 2
    assert details["recovery"] == {"action": "retry"}

    audit = latest_user_message_audit([{"role": "user", "content": "hello secret=abc"}])
    assert audit["latest_user_present"] is True
    assert audit["latest_user_chars"] == len("hello secret=abc")
    assert len(audit["latest_user_sha256"]) == 12

    synthetic_audit = latest_user_message_audit(
        [
            {"role": "user", "content": "fix issue #3"},
            {"role": "user", "content": "<retrieved-knowledge-base>\nNormalized observations: ..."},
        ]
    )
    assert synthetic_audit["latest_user_is_synthetic_context"] is True
    assert synthetic_audit["latest_human_user_preview"] == "fix issue #3"
    assert synthetic_audit["latest_synthetic_user_present"] is True

    diagnostics = context_pressure_diagnostics({"messages": ["x"], "tools": []}, context_limit=1)
    assert diagnostics["known_context_limit"] == 1
    assert "estimated_payload_tokens" in diagnostics


def test_remote_mutation_readback_content_satisfies_requirement() -> None:
    requirement = {
        "verification_patterns": {
            "old_absent": ["NO_STYLE"],
            "new_present": ["HAS_LINK"],
        }
    }
    assert readback_content_satisfies_requirement(requirement, "page HAS_LINK ok") is True
    assert readback_content_satisfies_requirement(requirement, "page HAS_LINK NO_STYLE") is False
    assert readback_content_satisfies_requirement(requirement, "page without marker") is False
    assert readback_content_satisfies_requirement({}, "page HAS_LINK ok") is False


def test_remote_mutation_path_host_helpers_match_requirements() -> None:
    result = SimpleNamespace(metadata={"path": "/var/www/app.html", "host": "EXAMPLE.com"}, error="No such file")
    assert tool_result_path_host(result, {"path": "/fallback", "host": "fallback"}) == (
        "/var/www/app.html",
        "example.com",
    )
    assert tool_result_path_host(SimpleNamespace(metadata={}), {"path": "/tmp/x", "target": "HOST"}) == (
        "/tmp/x",
        "host",
    )

    requirement = {"host": "example.com", "guessed_paths": ["/var/www/app.html", ""]}
    assert remote_mutation_target_matches(requirement, path="/var/www/app.html", host="example.com") is True
    assert remote_mutation_target_matches(requirement, path="/var/www/app.html", host="other") is False
    assert remote_mutation_target_matches(requirement, path="/etc/other", host="example.com") is False
    assert remote_mutation_guessed_paths(requirement) == ["/var/www/app.html"]
    assert "no such file" in remote_missing_file_markers(result)


def test_remote_mutation_nudge_predicates() -> None:
    assert bounded_region_not_found(SimpleNamespace(metadata={"error_kind": "bounded_region_not_found"}, error="")) is True
    assert bounded_region_not_found(SimpleNamespace(metadata={}, error="Remote bounded region was not found")) is True
    assert bounded_region_not_found(SimpleNamespace(metadata={}, error="different")) is False

    assert should_emit_small_file_rewrite_nudge(
        path="/tmp/app.py",
        recent_read_size=100,
        replacement_text="x" * 60,
    ) is True
    assert should_emit_small_file_rewrite_nudge(
        path="/tmp/app.py",
        recent_read_size=100,
        replacement_text="x" * 50,
    ) is False
    assert should_emit_small_file_rewrite_nudge(
        path="/tmp/app.py",
        recent_read_size=1024,
        replacement_text="x" * 900,
    ) is False


def test_task_boundary_followup_helpers_detect_continue_and_approval() -> None:
    from smallctl.harness.task_boundary_followups import (
        has_plan_execution_approval_context,
        is_continue_like_followup,
    )

    assert is_continue_like_followup("continue") is True
    assert is_continue_like_followup("cntinue") is True
    assert is_continue_like_followup("please continue") is True
    assert is_continue_like_followup("stop") is False
    assert is_continue_like_followup("resume") is True

    state = SimpleNamespace(
        pending_interrupt={"kind": "plan_execute_approval"},
        planner_interrupt=None,
        active_plan=None,
        draft_plan=None,
    )
    assert has_plan_execution_approval_context(state) is True

    state.pending_interrupt = None
    state.draft_plan = SimpleNamespace(status="awaiting_approval")
    assert has_plan_execution_approval_context(state) is True

    state.draft_plan = SimpleNamespace(status="active")
    assert has_plan_execution_approval_context(state) is False


def test_task_boundary_summary_helpers_clip_extract_and_duration() -> None:
    assert clip_task_summary_text("  hello  ") == "hello"
    assert clip_task_summary_text("abcdef", limit=3).endswith("[truncated]")
    assert extract_task_terminal_message({"message": {"question": "Need input?"}}) == "Need input?"
    assert extract_task_terminal_message({"reason": "guard tripped"}) == "guard tripped"
    assert extract_task_terminal_message({"error": {"message": "boom"}}) == "boom"
    assert task_duration_seconds("2024-01-01T00:00:00+00:00", "2024-01-01T00:00:01.234+00:00") == 1.234
    assert task_duration_seconds("bad", "2024-01-01T00:00:00+00:00") == 0.0


def test_dispatcher_remote_detection_helpers_match_ssh_context() -> None:
    from smallctl.tools.dispatcher_remote_detection import (
        task_clearly_targets_remote_ssh_host,
        task_requests_ssh_connection_probe,
    )

    state = SimpleNamespace(
        run_brief=SimpleNamespace(original_task="SSH into example.com and fix nginx"),
        working_memory=SimpleNamespace(current_goal=""),
    )
    assert task_clearly_targets_remote_ssh_host(state) is True
    assert task_requests_ssh_connection_probe(state) is True

    state.run_brief.original_task = "connect to 192.168.1.1 and restart service"
    assert task_clearly_targets_remote_ssh_host(state) is True

    state.run_brief.original_task = "just continue"
    assert task_clearly_targets_remote_ssh_host(state) is False
    assert task_requests_ssh_connection_probe(state) is False

    assert task_clearly_targets_remote_ssh_host(None) is False


def test_control_task_complete_gates_shape_blockers() -> None:
    from smallctl.tools.control_task_complete_gates import (
        task_complete_gate_staged_execution,
        task_complete_gate_missing_input,
        task_complete_gate_mutation_expectation,
        task_complete_gate_plan_subtasks,
    )

    state = LoopState(cwd="/tmp")
    state.plan_execution_mode = True
    state.active_step_id = "S1"
    block = task_complete_gate_staged_execution(state)
    assert block is not None
    assert block["metadata"]["reason"] == "task_complete_blocked_in_staged_execution"

    state = LoopState(cwd="/tmp")
    state.scratchpad["_unresolved_missing_input_file"] = {"path": "missing.txt"}
    block = task_complete_gate_missing_input(state)
    assert block is not None
    assert "missing.txt" in str(block["error"])

    state = LoopState(cwd="/tmp")
    state.challenge_progress.task_category = "coding"
    block = task_complete_gate_mutation_expectation(state, message="Implement phase 2 parser")
    assert block is not None
    assert block["metadata"]["reason"] == "mutation_expected_but_no_code_changes"

    state = LoopState(cwd="/tmp")
    from types import SimpleNamespace
    state.subtask_ledger = SimpleNamespace(
        subtasks=[SimpleNamespace(subtask_id="S1", title="A", status="pending")]
    )
    state.active_plan = SimpleNamespace()
    state.active_plan.iter_steps = lambda: [SimpleNamespace(step_id="S1")]
    block = task_complete_gate_plan_subtasks(state)
    assert block is not None
    assert block["metadata"]["reason"] == "plan_subtasks_incomplete"


def test_stderr_signature_helpers_extract_line_and_key() -> None:
    from smallctl.harness.tool_result_stderr_signatures import (
        stderr_signature_key,
        stderr_signature_line,
        stderr_text,
    )

    result = SimpleNamespace(output={"stderr": "  Error:\tfoo\n  "}, error="")
    assert stderr_text(result) == "  Error:\tfoo\n  "
    assert stderr_signature_line(result) == "Error: foo"
    assert len(stderr_signature_key(result)) == 12

    result = SimpleNamespace(output={}, error="fallback")
    assert stderr_text(result) == "fallback"
    assert stderr_signature_line(result) == "fallback"

    result = SimpleNamespace(output={}, error="")
    assert stderr_signature_line(result) is None
    assert stderr_signature_key(result) is None


def test_stderr_signature_ignores_curl_and_apt_noise() -> None:
    from smallctl.harness.tool_result_stderr_signatures import stderr_signature_line

    # Curl progress meter header and stats should be ignored.
    curl_stderr = (
        "  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n"
        "                                 Dload  Upload   Total   Spent    Left  Speed\n"
        "\r  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0\r\n"
        "\r100  1320  100  1320    0     0   5743      0 --:--:-- --:--:-- --:--:--  5764\r\n"
        "\n"
        "Error: The repository is not signed.\n"
    )
    result = SimpleNamespace(output={"stderr": curl_stderr}, error="")
    assert stderr_signature_line(result) == "Error: The repository is not signed."

    # Apt CLI stability warning should be ignored in favor of the real error.
    apt_stderr = (
        "\nWARNING: apt does not have a stable CLI interface. Use with caution in scripts.\n"
        "\n"
        "Warning: OpenPGP signature verification failed: ...\n"
        "Error: The repository 'http://example.com sarge Release' is not signed.\n"
    )
    result = SimpleNamespace(output={"stderr": apt_stderr}, error="")
    assert stderr_signature_line(result).startswith("Error: The repository")

    # Pure noise should yield no signature.
    noise_only = (
        "  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n"
        "                                 Dload  Upload   Total   Spent    Left  Speed\n"
    )
    result = SimpleNamespace(output={"stderr": noise_only}, error="")
    assert stderr_signature_line(result) is None


def test_ssh_files_patch_utils_apply_exact_and_bounded() -> None:
    from smallctl.tools.ssh_files_patch_utils import (
        apply_exact_patch_content,
        apply_replace_between_content,
        find_bounded_regions,
        normalize_whitespace_with_spans,
    )

    ok, result, meta = apply_exact_patch_content(
        "hello world", target_text="world", replacement_text="universe"
    )
    assert ok and result == "hello universe"
    assert find_bounded_regions("start middle end", start_text="start", end_text="end") == [(0, 16)]
    norm, spans = normalize_whitespace_with_spans("a  b\tc")
    assert norm == "a b c"
    assert len(spans) == 5

    ok, result, meta = apply_replace_between_content(
        "start middle end", start_text="start", end_text="end", replacement_text="replaced"
    )
    assert ok and result == "startreplacedend"


def test_tool_result_artifact_lifecycle_helpers() -> None:
    from smallctl.harness.tool_result_artifact_lifecycle import (
        _mark_prior_read_artifacts_stale,
        _maybe_emit_artifact_read_eof_overread_nudge,
        _supersede_prior_read_artifacts,
    )

    state = SimpleNamespace(
        artifacts={
            "a1": SimpleNamespace(
                tool_name="file_read",
                metadata={"path": "/foo/bar.py"},
            ),
            "a2": SimpleNamespace(
                tool_name="file_read",
                metadata={"path": "/foo/bar.py"},
            ),
        },
        scratchpad={},
        messages=[],
    )
    service = SimpleNamespace(harness=SimpleNamespace(state=state))

    _supersede_prior_read_artifacts(service, new_artifact_id="a2", tool_name="file_read", path="/foo/bar.py")
    assert state.artifacts["a1"].metadata.get("superseded_by") == "a2"

    _mark_prior_read_artifacts_stale(service, path="/foo/bar.py")
    # a1 is skipped because it has superseded_by
    assert state.artifacts["a1"].metadata.get("stale") is None
    # a2 is marked stale because it matches the path and has no superseded_by
    assert state.artifacts["a2"].metadata.get("stale") is True

    # EOF overread nudge
    result = SimpleNamespace(metadata={"eof_overread": True, "artifact_id": "a1", "requested_start_line": 5, "artifact_total_lines": 3})
    artifact = SimpleNamespace(artifact_id="a1")
    state2 = SimpleNamespace(
        artifacts={},
        scratchpad={},
        messages=[],
        append_message=lambda msg: None,
        _runlog=lambda *a, **k: None,
    )
    service2 = SimpleNamespace(harness=SimpleNamespace(state=state2, log=lambda *a, **k: None, _runlog=lambda *a, **k: None))
    _maybe_emit_artifact_read_eof_overread_nudge(service2, result=result, artifact=artifact)
    assert "a1:5:3" in state2.scratchpad.get("_artifact_read_eof_overread_nudges", [])


def test_tool_result_verifier_staleness_helper() -> None:
    from smallctl.harness.tool_result_verifier_staleness import _mark_verifier_stale_after_file_change

    state = SimpleNamespace(
        scratchpad={},
        current_verifier_verdict=None,
        last_verifier_verdict={"verdict": "pass"},
    )
    service = SimpleNamespace(harness=SimpleNamespace(state=state))
    _mark_verifier_stale_after_file_change(service, tool_name="file_write", paths=["/foo.py"])
    assert "_last_verifier_stale_after_mutation" in state.scratchpad


def test_tool_result_context_invalidation_helper() -> None:
    from smallctl.harness.tool_result_context_invalidation import _emit_context_invalidation

    state = SimpleNamespace(
        scratchpad={},
        invalidate_context=lambda **kw: {"reason": kw.get("reason"), "paths": kw.get("paths") or []},
    )
    service = SimpleNamespace(harness=SimpleNamespace(state=state, _runlog=lambda *a, **k: None))
    _emit_context_invalidation(service, reason="test", paths=["/foo.py"])


def test_tool_result_ssh_memory_helper() -> None:
    from smallctl.harness.tool_result_ssh_memory import _remember_session_ssh_target

    state = SimpleNamespace(scratchpad={})
    result = SimpleNamespace(success=True, metadata={}, output={})
    service = SimpleNamespace(harness=SimpleNamespace(state=state))
    _remember_session_ssh_target(service, tool_name="ssh_exec", result=result, arguments={"host": "example.com", "user": "root"})
    assert state.scratchpad["_session_ssh_targets"]["example.com"]["user"] == "root"


def test_tool_result_web_memory_helper() -> None:
    from smallctl.harness.tool_result_web_memory import _remember_web_search_results

    state = SimpleNamespace(scratchpad={})
    result = SimpleNamespace(output={"results": [{"result_id": "r1", "fetch_id": "f1"}]})
    artifact = SimpleNamespace(artifact_id="a1", metadata={})
    service = SimpleNamespace(harness=SimpleNamespace(state=state))
    _remember_web_search_results(service, result=result, artifact=artifact)
    assert state.scratchpad["_web_last_search_result_ids"] == ["r1"]


def test_tool_result_critical_errors_helper() -> None:
    from smallctl.harness.tool_result_critical_errors import _extract_and_pin_critical_errors

    wm = SimpleNamespace(known_facts=[], known_fact_meta=[])
    state = SimpleNamespace(working_memory=wm, step_count=1, current_phase="test")
    result = SimpleNamespace(success=False, error="Error: something broke")
    service = SimpleNamespace(harness=SimpleNamespace(state=state))
    _extract_and_pin_critical_errors(service, tool_name="shell_exec", result=result, artifact=None)
    assert any("CRITICAL: Error: something broke" in f for f in wm.known_facts)


def test_tool_result_stderr_circuit_breaker_helper() -> None:
    from smallctl.harness.tool_result_stderr_circuit_breaker import _record_stderr_signature_circuit_breaker

    state = SimpleNamespace(scratchpad={}, recent_errors=[])
    result = SimpleNamespace(output={"stderr": "Error: foo"}, error="")
    service = SimpleNamespace(harness=SimpleNamespace(state=state))
    _record_stderr_signature_circuit_breaker(service, tool_name="shell_exec", result=result)
    # First call should not trigger circuit breaker
    assert "_stderr_signature_circuit_breaker" not in state.scratchpad
    _record_stderr_signature_circuit_breaker(service, tool_name="shell_exec", result=result)
    assert "_stderr_signature_circuit_breaker" in state.scratchpad


def test_tool_result_subtask_ledger_helper() -> None:
    from smallctl.harness.tool_result_subtask_ledger import _update_subtask_ledger_from_verifier

    class FakeLedger:
        def import_plan_if_needed(self): pass
        def infer_or_create_active_subtask(self):
            return SimpleNamespace(subtask_id="S1", failure_classes=[])
        def attach_evidence(self, sid, text): pass
        def mark_done_if_verified(self, sid, verdict): pass

    config = SimpleNamespace(subtask_ledger_enabled=True)
    state = SimpleNamespace()
    harness = SimpleNamespace(state=state, config=config, subtask_ledger=FakeLedger())
    service = SimpleNamespace(harness=harness)
    _update_subtask_ledger_from_verifier(service, {"verdict": "pass", "command": "test"})


def test_tool_result_touched_symbols_helper() -> None:
    from smallctl.harness.tool_result_touched_symbols import _record_touched_symbols_from_mutation

    state = SimpleNamespace(scratchpad={}, cwd="/tmp")
    result = SimpleNamespace(metadata={"touched_symbols": ["foo"]})
    service = SimpleNamespace(harness=SimpleNamespace(state=state, _runlog=lambda *a, **k: None))
    _record_touched_symbols_from_mutation(
        service, tool_name="file_write", result=result, arguments={"content": "def bar(): pass"},
        artifact=None, mutated_path="/tmp/test.py",
    )
    assert "_touched_symbols" in state.scratchpad


def test_tool_result_remote_mutation_helpers_import() -> None:
    from smallctl.harness.tool_result_remote_mutation import (
        _clear_remote_mutation_requirement_from_tool,
        _emit_remote_mutation_nudge,
        _handle_remote_mutation_verifier_result,
        _maybe_emit_bounded_region_trap_nudge,
        _maybe_emit_small_file_rewrite_nudge,
        _observe_remote_installer_preflight_check,
        _recent_ssh_file_read_size,
        _record_failed_verification_attempt,
        _record_remote_mutation_requirement,
    )

    # Smoke test imports only — these functions are heavily tested in test_ssh_file_tools
    assert callable(_record_remote_mutation_requirement)
    assert callable(_emit_remote_mutation_nudge)
    assert callable(_clear_remote_mutation_requirement_from_tool)
    assert callable(_handle_remote_mutation_verifier_result)
    assert callable(_record_failed_verification_attempt)
    assert callable(_maybe_emit_bounded_region_trap_nudge)
    assert callable(_maybe_emit_small_file_rewrite_nudge)
    assert callable(_recent_ssh_file_read_size)
    assert callable(_observe_remote_installer_preflight_check)


def test_ssh_files_preconditions_resolve_expected_sha() -> None:
    from smallctl.tools.ssh_files_preconditions import _resolve_expected_sha_precondition

    state = SimpleNamespace(artifacts={})
    sha, meta = _resolve_expected_sha_precondition(
        path="/foo.py", expected_sha256="abc", source_artifact_id=None, state=state
    )
    assert sha == "abc"
    assert meta is None


def test_ssh_files_mutation_tracking_clear_requirement() -> None:
    from smallctl.tools.ssh_files_mutation_tracking import _clear_remote_mutation_requirement

    state = SimpleNamespace(scratchpad={"_remote_mutation_requires_verification": {"guessed_paths": ["/foo.py"], "host": "example.com"}})
    _clear_remote_mutation_requirement(state, path="/foo.py", host="example.com")
    assert "_remote_mutation_requires_verification" not in state.scratchpad


def test_network_interactive_sessions_snapshot() -> None:
    from smallctl.tools.network_interactive_sessions import _interactive_session_snapshot

    session = {
        "proc": SimpleNamespace(returncode=0),
        "stdout": ["hello"],
        "stderr": ["world"],
        "host": "example.com",
        "user": "root",
        "command": "ls",
    }
    snapshot = _interactive_session_snapshot("s1", session)
    assert snapshot["status"] == "exited"
    assert snapshot["exit_code"] == 0


def test_network_installer_preflight_import() -> None:
    from smallctl.tools.network_installer_preflight import _run_remote_installer_preflight_probes

    assert callable(_run_remote_installer_preflight_probes)


def test_dispatcher_policy_guards_fama_block_none_on_missing_state() -> None:
    from smallctl.tools.dispatcher_policy_guards import _fama_dispatch_block

    assert _fama_dispatch_block("shell_exec", {}, state=None, phase="explore") is None


def test_dispatcher_policy_guards_staged_allowlist_none_when_not_staged() -> None:
    from smallctl.tools.dispatcher_policy_guards import _staged_tool_allowlist_error

    state = SimpleNamespace(plan_execution_mode=False)
    assert _staged_tool_allowlist_error(state, "shell_exec") is None


def test_dispatcher_tool_guards_remote_file_guard_none_for_local_path() -> None:
    from smallctl.tools.dispatcher_tool_guards import _guard_remote_file_tool_request

    result = _guard_remote_file_tool_request("file_read", {"path": "/tmp/foo.txt"}, state=None, ssh_available=True)
    assert result is None


def test_dispatcher_tool_guards_remote_shell_guard_none_for_empty_command() -> None:
    from smallctl.tools.dispatcher_tool_guards import _guard_remote_shell_tool_request

    assert _guard_remote_shell_tool_request("", state=None, ssh_available=True) is None


def test_dispatcher_tool_predicates_ssh_exec_available_with_missing_registry() -> None:
    from smallctl.tools.dispatcher_tool_predicates import _ssh_exec_available

    assert _ssh_exec_available(None, phase=None, state=None) is False


def test_dispatcher_tool_predicates_recent_ssh_auth_failure_none() -> None:
    from smallctl.tools.dispatcher_tool_predicates import _recent_ssh_auth_failure

    assert _recent_ssh_auth_failure(None) is False


def test_dispatcher_tool_predicates_escalation_recommends_local_shell_none() -> None:
    from smallctl.tools.dispatcher_tool_predicates import _escalation_recommends_local_shell

    assert _escalation_recommends_local_shell(None) is False


def test_dispatcher_ssh_recovery_infer_host_empty_on_none() -> None:
    from smallctl.tools.dispatcher_ssh_recovery import _infer_ssh_host_from_context

    assert _infer_ssh_host_from_context(None) == ""


def test_dispatcher_ssh_recovery_recover_args_passthrough() -> None:
    from smallctl.tools.dispatcher_ssh_recovery import _recover_ssh_arguments_from_task_context

    args, meta = _recover_ssh_arguments_from_task_context({"host": "example.com"}, state=None)
    assert args["host"] == "example.com"
    assert isinstance(meta, dict)


def test_dispatcher_ssh_recovery_pin_guard_none_on_none() -> None:
    from smallctl.tools.dispatcher_ssh_recovery import _pin_and_guard_ssh_credentials

    args, block, meta = _pin_and_guard_ssh_credentials({}, state=None, normalization_metadata={})
    assert block is None
    assert isinstance(meta, dict)


def test_dispatcher_scope_predicates_remote_execute_false() -> None:
    from smallctl.tools.dispatcher_scope_predicates import _task_is_remote_execute

    assert _task_is_remote_execute(None) is False
    assert _task_is_remote_execute(SimpleNamespace(task_mode="local")) is False


def test_dispatcher_scope_predicates_remote_scope_active_false() -> None:
    from smallctl.tools.dispatcher_scope_predicates import _remote_scope_is_active

    assert _remote_scope_is_active(None) is False


def test_dispatcher_scope_predicates_has_single_confirmed_target_false() -> None:
    from smallctl.tools.dispatcher_scope_predicates import _has_single_confirmed_ssh_target

    assert _has_single_confirmed_ssh_target(None) is False


def test_client_transport_client_lifecycle_key() -> None:
    from smallctl.client.client_transport_client_lifecycle import _client_key

    client = SimpleNamespace(base_url="http://localhost", api_key="secret")
    assert _client_key(client) == ("http://localhost", "secret")


def test_client_transport_llamacpp_repair_passthrough_for_non_llamacpp() -> None:
    from smallctl.client.client_transport_llamacpp_repair import _repair_llamacpp_system_messages_for_transport

    client = SimpleNamespace(provider_profile="openai", log=None, run_logger=None)
    messages = [{"role": "system", "content": "test"}]
    assert _repair_llamacpp_system_messages_for_transport(client, messages) == messages


def test_client_transport_context_limits_remember_none() -> None:
    from smallctl.client.client_transport_context_limits import _remember_context_limit

    client = SimpleNamespace()
    assert _remember_context_limit(client, None) is None


def test_client_transport_model_metadata_import() -> None:
    from smallctl.client.client_transport_model_metadata import _remember_model_metadata

    assert callable(_remember_model_metadata)


def test_client_transport_openrouter_preflight_import() -> None:
    from smallctl.client.client_transport_openrouter_preflight import _preflight_openrouter_auth

    assert callable(_preflight_openrouter_auth)


def test_client_transport_audit_import() -> None:
    from smallctl.client.client_transport_audit import _log_request_audit

    assert callable(_log_request_audit)


def test_tool_result_verification_assess_import() -> None:
    from smallctl.harness.tool_result_verification_assess import assess_remote_mutation_verification

    assert callable(assess_remote_mutation_verification)


def test_tool_result_verification_ssh_recovery_import() -> None:
    from smallctl.harness.tool_result_verification_ssh_recovery import _update_ssh_auth_recovery_state

    assert callable(_update_ssh_auth_recovery_state)


def test_tool_result_verification_artifact_import() -> None:
    from smallctl.harness.tool_result_verification_artifact import _annotate_verifier_artifact

    assert callable(_annotate_verifier_artifact)


def test_tool_result_verification_removal_import() -> None:
    from smallctl.harness.tool_result_verification_removal import _classify_removal_absence_probe

    assert callable(_classify_removal_absence_probe)


def test_tool_result_verification_timeout_import() -> None:
    from smallctl.harness.tool_result_verification_timeout import _is_long_running_remote_command_timeout

    assert callable(_is_long_running_remote_command_timeout)


def test_tool_result_verification_blocker_import() -> None:
    from smallctl.harness.tool_result_verification_blocker import _extract_latest_execution_blocker, _store_latest_execution_blocker

    assert callable(_extract_latest_execution_blocker)
    assert callable(_store_latest_execution_blocker)


def test_tool_result_verification_semantic_import() -> None:
    from smallctl.harness.tool_result_verification_semantic import _semantic_verifier_failure

    assert callable(_semantic_verifier_failure)


def test_tool_result_verification_readback_import() -> None:
    from smallctl.harness.tool_result_verification_readback import _simple_remote_readback_path

    assert callable(_simple_remote_readback_path)


def test_tool_result_verification_repair_import() -> None:
    from smallctl.harness.tool_result_verification_repair import _update_repair_cycle_state, _record_docker_retry_state, _update_acceptance_ledger

    assert callable(_update_repair_cycle_state)
    assert callable(_record_docker_retry_state)
    assert callable(_update_acceptance_ledger)


def test_tool_result_verification_audit_import() -> None:
    from smallctl.harness.tool_result_verification_audit import _is_audit_task

    assert callable(_is_audit_task)


def test_tool_result_verification_store_import() -> None:
    from smallctl.harness.tool_result_verification_store import _store_verifier_verdict

    assert callable(_store_verifier_verdict)
