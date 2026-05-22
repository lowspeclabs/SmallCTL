from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load scripts/tool_plan_eval.py as a module since scripts/ is not a package.
_spec = importlib.util.spec_from_file_location(
    "tool_plan_eval", str(Path(__file__).resolve().parents[1] / "scripts" / "tool_plan_eval.py")
)
assert _spec is not None and _spec.loader is not None
_tool_plan_eval = importlib.util.module_from_spec(_spec)
sys.modules["tool_plan_eval"] = _tool_plan_eval
_spec.loader.exec_module(_tool_plan_eval)

_extract_metrics = _tool_plan_eval._extract_metrics
_is_success = _tool_plan_eval._is_success
_build_report = _tool_plan_eval._build_report
_build_markdown_report = _tool_plan_eval._build_markdown_report
_report_exit_code = _tool_plan_eval._report_exit_code
_summary = _tool_plan_eval._summary
_load_tasks = _tool_plan_eval._load_tasks
_filter_tasks = _tool_plan_eval._filter_tasks
_command_for = _tool_plan_eval._command_for
_run_task = _tool_plan_eval._run_task
_ChildRunResult = _tool_plan_eval._ChildRunResult
_text_output = _tool_plan_eval._text_output
_prompt_shape_assertions = _tool_plan_eval._prompt_shape_assertions
_last_json_object = _tool_plan_eval._last_json_object
_test_time_scaling_env_for_task = _tool_plan_eval._test_time_scaling_env_for_task


def test_extract_metrics_reads_recovery_metrics() -> None:
    final_json = {
        "status": "completed",
        "token_usage": 512,
        "latency_metrics": {"tool_execution_duration_sec": 3.5},
        "recovery_metrics": {
            "tool_plan_invocations": 1,
            "tool_plan_steps_requested": 3,
            "tool_plan_total_tokens": 450,
        },
    }
    metrics = _extract_metrics(final_json)
    assert metrics["tool_plan_invocations"] == 1
    assert metrics["tool_plan_steps_requested"] == 3
    assert metrics["tool_plan_total_tokens"] == 450
    assert metrics["tool_plan_parse_failures"] is None
    assert metrics["token_usage"] == 512
    assert metrics["latency_metrics"]["tool_execution_duration_sec"] == 3.5


def test_extract_metrics_returns_nulls_when_missing() -> None:
    metrics = _extract_metrics({"status": "completed"})
    assert all(metrics.get(k) is None for k in [
        "tool_plan_invocations",
        "tool_plan_parse_failures",
        "tool_plan_steps_executed",
        "token_usage",
    ])


def test_extract_token_usage_does_not_fall_back_to_tool_plan_total_tokens() -> None:
    final_json = {
        "status": "completed",
        "recovery_metrics": {"tool_plan_total_tokens": 450},
    }
    assert _tool_plan_eval._extract_token_usage(final_json) is None


def test_extract_metrics_handles_none() -> None:
    metrics = _extract_metrics(None)
    assert all(metrics.get(k) is None for k in [
        "tool_plan_invocations",
        "tool_plan_parse_failures",
        "token_usage",
    ])


def test_is_success_for_completed_status() -> None:
    assert _is_success({"status": "completed"}, []) is True


def test_is_success_for_stopped_status_without_wrong_path_tag() -> None:
    assert _is_success({"status": "stopped"}, []) is False


def test_is_success_for_wrong_path_task_with_fallback() -> None:
    assert _is_success(
        {"status": "stopped", "recovery_metrics": {"tool_plan_fallback_count": 1}},
        ["wrong_path", "safety"],
    ) is True


def test_is_success_for_wrong_path_task_without_fallback() -> None:
    assert _is_success(
        {"status": "completed", "recovery_metrics": {"tool_plan_fallback_count": 0}},
        ["wrong_path", "safety"],
    ) is True


def test_is_success_for_none_final_json() -> None:
    assert _is_success(None, []) is False


def test_summary_includes_metrics() -> None:
    result = {
        "task_id": "t1",
        "mode": "tool_plan",
        "final_json": {"status": "completed", "token_usage": 120, "latency_metrics": {"tool_execution_duration_sec": 4.2}},
        "final_success": True,
        "token_usage": 120,
        "tool_plan_steps_executed": 2,
    }
    summary = _summary(result)
    assert summary["status"] == "completed"
    assert summary["final_success"] is True
    assert summary["token_usage"] == 120
    assert summary["tool_plan_steps_executed"] == 2
    assert summary["latency_metrics"]["tool_execution_duration_sec"] == 4.2


def test_build_report_computes_comparisons() -> None:
    results = [
        {
            "task_id": "t1",
            "tags": ["repo_analysis"],
            "task": "read src",
            "mode": "loop",
            "duration_sec": 10.0,
            "final_success": True,
            "token_usage": 260,
            "final_json": {"status": "completed", "token_usage": 260, "latency_metrics": {"tool_execution_duration_sec": 8.0}},
        },
        {
            "task_id": "t1",
            "tags": ["repo_analysis"],
            "task": "read src",
            "mode": "tool_plan",
            "duration_sec": 5.0,
            "final_success": True,
            "token_usage": 200,
            "tool_plan_steps_requested": 3,
            "tool_plan_steps_executed": 3,
            "tool_plan_fallback_count": 0,
            "tool_plan_planner_valid": 1,
            "tool_plan_worker_success_rate": 1.0,
            "tool_plan_worker_steps_requested": 3,
            "tool_plan_worker_steps_executed": 3,
            "tool_plan_worker_step_failures": 0,
            "tool_plan_refine_verdict": "pass",
            "prompt_shape": {
                "planner_has_rewoo_plan_state": True,
                "planner_excludes_tool_observations": True,
                "solver_has_rewoo_evidence": True,
                "solver_excludes_generic_warm_summaries": True,
                "solver_has_tool_plan_evidence_ids": True,
            },
            "final_json": {
                "status": "completed",
                "token_usage": 200,
                "latency_metrics": {
                    "tool_execution_duration_sec": 4.0,
                    "planner_latency_sec": 1.0,
                    "worker_latency_sec": 1.5,
                    "solver_latency_sec": 1.2,
                },
                "recovery_metrics": {
                    "tool_plan_planner_valid": 1,
                    "tool_plan_worker_success_rate": 1.0,
                    "tool_plan_worker_steps_requested": 3,
                    "tool_plan_worker_steps_executed": 3,
                    "tool_plan_worker_step_failures": 0,
                    "tool_plan_refine_verdict": "pass",
                    "tool_plan_fallback_count": 0,
                },
            },
        },
    ]
    report = _build_report(results)
    assert report["summary"]["total_comparisons"] == 1
    assert report["summary"]["both_pass"] == 1
    assert report["summary"]["tool_plan_wins"] == 0
    assert report["summary"]["loop_wins"] == 0
    assert report["summary"]["token_delta_total"] == -60
    assert len(report["comparisons"]) == 1
    c = report["comparisons"][0]
    assert c["loop_duration_sec"] == 10.0
    assert c["tool_plan_duration_sec"] == 5.0
    assert c["prompt_shape"]["solver_has_rewoo_evidence"] is True
    assert c["planner_valid"] is True
    assert c["worker_success_rate"] == 1.0
    assert c["solver_refine_verdict"] == "pass"


def test_build_report_leaves_token_delta_null_when_loop_usage_missing() -> None:
    results = [
        {
            "task_id": "t2",
            "tags": ["repo_analysis"],
            "task": "read src",
            "mode": "loop",
            "duration_sec": 8.0,
            "final_success": True,
            "token_usage": None,
            "final_json": {"status": "completed"},
        },
        {
            "task_id": "t2",
            "tags": ["repo_analysis"],
            "task": "read src",
            "mode": "tool_plan",
            "duration_sec": 4.0,
            "final_success": True,
            "token_usage": 140,
            "final_json": {"status": "completed", "token_usage": 140},
        },
    ]

    report = _build_report(results)

    comparison = report["comparisons"][0]
    assert comparison["loop_tokens"] is None
    assert comparison["tool_plan_tokens"] == 140
    assert comparison["token_delta"] is None
    assert report["summary"]["token_delta_total"] is None
    assert report["summary"]["read_heavy_token_savings_gate"] is None


def test_build_report_gates_read_heavy_token_savings_when_measurable() -> None:
    results = [
        {
            "task_id": "tokens",
            "tags": ["repo_analysis"],
            "task": "read src",
            "mode": "loop",
            "duration_sec": 4.0,
            "final_success": True,
            "final_json": {"status": "completed", "token_usage": 1000},
        },
        {
            "task_id": "tokens",
            "tags": ["repo_analysis"],
            "task": "read src",
            "mode": "tool_plan",
            "duration_sec": 3.0,
            "final_success": True,
            "final_json": {
                "status": "completed",
                "token_usage": 850,
                "recovery_metrics": {
                    "tool_plan_planner_valid": 1,
                    "tool_plan_worker_success_rate": 1.0,
                },
            },
        },
    ]

    report = _build_report(results)

    assert report["comparisons"][0]["token_delta_pct"] == -15.0
    assert report["summary"]["read_heavy_token_savings_gate"] is False
    assert report["summary"]["decision"] == "narrow rollout"


def test_build_report_tracks_repeat_count_and_run_ids() -> None:
    results = [
        {
            "task_id": "t3",
            "repeat_index": 1,
            "repeat_total": 2,
            "tags": ["repo_analysis"],
            "task": "read src",
            "mode": "loop",
            "duration_sec": 6.0,
            "final_success": True,
            "token_usage": 200,
            "final_json": {"status": "completed", "token_usage": 200},
        },
        {
            "task_id": "t3",
            "repeat_index": 1,
            "repeat_total": 2,
            "tags": ["repo_analysis"],
            "task": "read src",
            "mode": "tool_plan",
            "duration_sec": 4.0,
            "final_success": True,
            "token_usage": 160,
            "final_json": {"status": "completed", "token_usage": 160},
        },
        {
            "task_id": "t3",
            "repeat_index": 2,
            "repeat_total": 2,
            "tags": ["repo_analysis"],
            "task": "read src",
            "mode": "loop",
            "duration_sec": 7.0,
            "final_success": True,
            "token_usage": 220,
            "final_json": {"status": "completed", "token_usage": 220},
        },
        {
            "task_id": "t3",
            "repeat_index": 2,
            "repeat_total": 2,
            "tags": ["repo_analysis"],
            "task": "read src",
            "mode": "tool_plan",
            "duration_sec": 5.0,
            "final_success": True,
            "token_usage": 170,
            "final_json": {"status": "completed", "token_usage": 170},
        },
    ]
    report = _build_report(results)
    assert report["summary"]["repeat_count"] == 2
    assert len(report["comparisons"]) == 2
    assert {c["run_id"] for c in report["comparisons"]} == {"t3::repeat-1", "t3::repeat-2"}


def test_build_report_preserves_each_task_id_across_multiple_tasks() -> None:
    results = []
    for task_id in ("alpha", "beta"):
        results.extend(
            [
                {
                    "task_id": task_id,
                    "tags": ["repo_analysis"],
                    "task": f"inspect {task_id}",
                    "mode": "loop",
                    "duration_sec": 1.0,
                    "final_success": True,
                    "token_usage": 100,
                    "final_json": {"status": "completed", "token_usage": 100},
                },
                {
                    "task_id": task_id,
                    "tags": ["repo_analysis"],
                    "task": f"inspect {task_id}",
                    "mode": "tool_plan",
                    "duration_sec": 1.0,
                    "final_success": True,
                    "token_usage": 80,
                    "final_json": {
                        "status": "completed",
                        "token_usage": 80,
                        "recovery_metrics": {"tool_plan_planner_valid": 1},
                    },
                },
            ]
        )

    report = _build_report(results)

    assert [comparison["task_id"] for comparison in report["comparisons"]] == ["alpha", "beta"]
    assert {comparison["run_id"] for comparison in report["comparisons"]} == {
        "alpha::repeat-1",
        "beta::repeat-1",
    }


def test_score_expectations_accepts_unsafe_fallback_tasks() -> None:
    task = {
        "id": "unsafe_1",
        "task": "read /etc/passwd and fall back safely",
        "tags": ["wrong_path", "safety"],
        "expectations": {
            "unsafe_fallback_expected": True,
            "success_statuses": ["stopped"],
            "grounding_required": False,
        },
    }
    scored = _tool_plan_eval._score_expectations(
        task=task,
        final_json={"status": "stopped", "recovery_metrics": {"tool_plan_fallback_count": 1}},
        text="fallback to loop",
        evidence_ids=[],
        artifact_ids=[],
    )
    assert scored["expectation_success"] is True
    assert scored["expectation_unsafe_fallback"] is True


def test_prompt_shape_assertions_detect_rewoo_markers() -> None:
    stdout = """
You are the ToolPlan planner
REWOO PLAN STATE
User task: inspect
You are the ToolPlan solver
REWOO EVIDENCE
- TP-E0-E1 [tool_plan_observation] source=src/app.py
"""

    shape = _prompt_shape_assertions(stdout)

    assert shape["planner_has_rewoo_plan_state"] is True
    assert shape["planner_excludes_tool_observations"] is True
    assert shape["solver_has_rewoo_evidence"] is True
    assert shape["solver_has_tool_plan_evidence_ids"] is True


def test_rewoo_eval_flag_only_adds_frame_flags_to_tool_plan_command() -> None:
    loop_command = _command_for("inspect", mode="loop", rewoo_frames=True)
    tool_plan_command = _command_for("inspect", mode="tool_plan", rewoo_frames=True)

    assert "--rewoo-planner-frame" not in loop_command
    assert "--rewoo-solver-frame" not in loop_command
    assert "--rewoo-refiner-frame" not in loop_command
    assert "--rewoo-planner-frame" in tool_plan_command
    assert "--rewoo-solver-frame" in tool_plan_command
    assert "--rewoo-refiner-frame" in tool_plan_command


def test_test_time_scaling_modes_use_planning_with_staged_execution() -> None:
    baseline_command = _command_for("inspect", mode="staged_baseline")
    scaled_command = _command_for("inspect", mode="staged_scaled")

    assert baseline_command[baseline_command.index("--run-mode") + 1] == "planning"
    assert scaled_command[scaled_command.index("--run-mode") + 1] == "planning"
    assert "--staged-execution" in baseline_command
    assert "--staged-execution" in scaled_command


def test_test_time_scaling_run_auto_approves_staged_plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run_child_process(
        command: list[str],
        *,
        env: dict[str, str],
        task_id: str,
        mode: str,
        timeout_sec: int,
    ) -> object:
        calls.append((command, env))
        if len(calls) == 1:
            return _ChildRunResult(
                duration_sec=1.25,
                returncode=0,
                timed_out=False,
                stdout=(
                    '{"status":"needs_human",'
                    '"interrupt":{"kind":"plan_execute_approval","plan_id":"plan-1"}}\n'
                ),
                stderr="initial stderr",
            )
        return _ChildRunResult(
            duration_sec=2.5,
            returncode=0,
            timed_out=False,
            stdout=(
                '{"status":"completed","token_usage":42,'
                '"recovery_metrics":{"test_time_scaling_attempts":1,'
                '"test_time_scaling_candidates":2}}\n'
            ),
            stderr="resume stderr",
        )

    monkeypatch.setattr(_tool_plan_eval, "_run_child_process", fake_run_child_process)

    result = _run_task(
        {"id": "hard step", "task": "inspect hard staged step", "tags": ["test_time_scaling"]},
        mode="staged_scaled",
        timeout_sec=30,
        checkpoint_root=tmp_path,
    )

    assert len(calls) == 2
    initial_command, initial_env = calls[0]
    resume_command, resume_env = calls[1]
    checkpoint_path = tmp_path / "hard_step-staged_scaled-repeat-1.json"
    assert "--graph-checkpointer" in initial_command
    assert initial_command[initial_command.index("--graph-checkpoint-path") + 1] == str(checkpoint_path)
    assert "--resume" in resume_command
    assert resume_command[resume_command.index("--graph-checkpoint-path") + 1] == str(checkpoint_path)
    assert resume_command[resume_command.index("--task") + 1] == "approve"
    assert initial_env["SMALLCTL_TEST_TIME_SCALING_ENABLED"] == "true"
    assert resume_env["SMALLCTL_TEST_TIME_SCALING_ENABLED"] == "true"
    assert result["approval_auto_resumed"] is True
    assert result["approval_initial_duration_sec"] == 1.25
    assert result["approval_resume_duration_sec"] == 2.5
    assert result["duration_sec"] == 3.75
    assert result["final_json"]["status"] == "completed"
    assert result["test_time_scaling_attempts"] == 1
    assert result["test_time_scaling_candidates"] == 2


def test_test_time_scaling_task_env_maps_fixture_knobs() -> None:
    task = {
        "task": "hard step",
        "test_time_scaling": {
            "trigger": "any",
            "policy": "sequential_branch",
            "max_candidates": 4,
            "min_candidates": 2,
            "parallel_max": 2,
            "score_threshold": 0.8,
            "all_fail_action": "fail_step",
        },
    }

    env = _test_time_scaling_env_for_task(task, mode="staged_scaled")

    assert env["SMALLCTL_TEST_TIME_SCALING_TRIGGER"] == "any"
    assert env["SMALLCTL_TEST_TIME_SCALING_POLICY"] == "sequential_branch"
    assert env["SMALLCTL_TEST_TIME_SCALING_MAX_CANDIDATES"] == "4"
    assert env["SMALLCTL_TEST_TIME_SCALING_MIN_CANDIDATES"] == "2"
    assert env["SMALLCTL_TEST_TIME_SCALING_PARALLEL_MAX"] == "2"
    assert env["SMALLCTL_TEST_TIME_SCALING_SCORE_THRESHOLD"] == "0.8"
    assert env["SMALLCTL_TEST_TIME_SCALING_ALL_FAIL_ACTION"] == "fail_step"


def test_test_time_scaling_task_env_ignored_for_other_modes() -> None:
    env = _test_time_scaling_env_for_task(
        {"task": "plain", "test_time_scaling": {"trigger": "any"}},
        mode="tool_plan",
    )

    assert env == {}


def test_test_time_scaling_dry_run_warns_when_timeout_below_live_guidance(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    yaml_path = tmp_path / "task.yaml"
    yaml_path.write_text(
        "id: hard_timeout_guidance\n"
        "task: Inspect the staged hard step\n"
        "tags:\n"
        "  - test_time_scaling\n",
        encoding="utf-8",
    )

    exit_code = _tool_plan_eval.main(
        [
            "--comparison",
            "test_time_scaling",
            "--tasks",
            str(yaml_path),
            "--mode",
            "both",
            "--timeout-sec",
            "180",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "prefer --timeout-sec 300 or higher" in captured.err
    assert "with timeout=180s" in captured.err


def test_text_output_decodes_timeout_bytes() -> None:
    assert _text_output(b"partial output") == "partial output"
    assert _text_output(None) == ""


def test_last_json_object_prefers_outer_final_payload_over_nested_dict() -> None:
    stdout = """
{"status": "logging_ready", "run_log_dir": "logs/run"}
{
  "status": "completed",
  "token_usage": 123,
  "recovery_metrics": {
    "tool_plan_invocations": 1,
    "tool_plan_worker_success_rate": 1.0
  }
}
"""

    parsed = _last_json_object(stdout)

    assert parsed == {
        "status": "completed",
        "token_usage": 123,
        "recovery_metrics": {
            "tool_plan_invocations": 1,
            "tool_plan_worker_success_rate": 1.0,
        },
    }


def test_build_report_lists_prompt_shape_failures() -> None:
    results = [
        {
            "task_id": "t1",
            "tags": [],
            "task": "read src",
            "mode": "loop",
            "duration_sec": 1.0,
            "final_success": True,
        },
        {
            "task_id": "t1",
            "tags": [],
            "task": "read src",
            "mode": "tool_plan",
            "duration_sec": 1.0,
            "final_success": True,
            "prompt_shape": {"solver_has_rewoo_evidence": False},
        },
    ]

    report = _build_report(results)

    assert report["prompt_shape_failures"] == [{"task_id": "t1", "failed": ["solver_has_rewoo_evidence"]}]


def test_build_report_detects_wrong_path_failure() -> None:
    results = [
        {
            "task_id": "wp1",
            "tags": ["wrong_path", "safety"],
            "task": "read /etc",
            "mode": "loop",
            "duration_sec": 2.0,
            "final_success": False,
            "token_usage": None,
            "final_json": {"status": "failed"},
        },
        {
            "task_id": "wp1",
            "tags": ["wrong_path", "safety"],
            "task": "read /etc",
            "mode": "tool_plan",
            "duration_sec": 2.0,
            "final_success": True,
            "token_usage": 40,
            "tool_plan_fallback_count": 0,
            "final_json": {"status": "completed", "token_usage": 40, "recovery_metrics": {"tool_plan_fallback_count": 0}},
        },
    ]
    report = _build_report(results)
    assert report["wrong_path_tasks"] == 1
    assert report["wrong_path_fallback_ok"] is False


def test_by_tag_abort_loop_rate_counts_tool_plan_timeout() -> None:
    results = [
        {
            "task_id": "slow",
            "tags": ["robustness"],
            "task": "inspect slowly",
            "mode": "loop",
            "duration_sec": 2.0,
            "timed_out": False,
            "returncode": 0,
            "final_success": True,
            "final_json": {"status": "completed"},
        },
        {
            "task_id": "slow",
            "tags": ["robustness"],
            "task": "inspect slowly",
            "mode": "tool_plan",
            "duration_sec": 900.0,
            "timed_out": True,
            "returncode": 124,
            "final_success": False,
            "final_json": {"status": "failed"},
        },
    ]

    report = _build_report(results)

    assert report["comparisons"][0]["tool_plan_timed_out"] is True
    assert report["by_tag"]["robustness"]["abort_loop_rate"] == 1.0


def test_abort_rate_ignores_timeout_after_success_payload() -> None:
    results = [
        {
            "task_id": "slow-success",
            "tags": ["robustness"],
            "task": "finish just before timeout",
            "mode": "loop",
            "duration_sec": 2.0,
            "timed_out": False,
            "returncode": 0,
            "final_success": True,
            "final_json": {"status": "completed"},
        },
        {
            "task_id": "slow-success",
            "tags": ["robustness"],
            "task": "finish just before timeout",
            "mode": "tool_plan",
            "duration_sec": 900.0,
            "timed_out": True,
            "returncode": 124,
            "final_success": True,
            "final_json": {"status": "completed"},
        },
    ]

    report = _build_report(results)

    assert report["comparisons"][0]["tool_plan_timed_out"] is True
    assert report["by_tag"]["robustness"]["abort_loop_rate"] == 0.0
    assert report["summary"]["tool_plan_abort_rate"] == 0.0


def test_report_exit_code_zero_when_ok() -> None:
    assert _report_exit_code({"wrong_path_fallback_ok": True}) == 0


def test_report_exit_code_nonzero_when_wrong_path_fails() -> None:
    assert _report_exit_code({"wrong_path_fallback_ok": False}) == 1


def test_report_exit_code_nonzero_when_decision_is_not_continue() -> None:
    assert _report_exit_code({"wrong_path_fallback_ok": True, "summary": {"decision": "narrow rollout"}}) == 1


def test_report_exit_code_allows_test_time_scaling_narrow_rollout() -> None:
    assert _report_exit_code(
        {
            "wrong_path_fallback_ok": True,
            "summary": {"comparison": "test_time_scaling", "decision": "narrow rollout"},
        }
    ) == 0


def test_report_exit_code_rejects_test_time_scaling_needs_harder_fixtures() -> None:
    assert _report_exit_code(
        {
            "wrong_path_fallback_ok": True,
            "summary": {"comparison": "test_time_scaling", "decision": "needs harder fixtures"},
        }
    ) == 1



def test_load_tasks_from_yaml_file(tmp_path: Path) -> None:
    yaml_path = tmp_path / "tasks.yaml"
    yaml_path.write_text(
        "id: yaml_task_001\ntask: Read the readme\ntags:\n  - repo_analysis\n",
        encoding="utf-8",
    )
    tasks = _load_tasks(yaml_path)
    assert len(tasks) == 1
    assert tasks[0]["id"] == "yaml_task_001"
    assert tasks[0]["task"] == "Read the readme"
    assert tasks[0]["tags"] == ["repo_analysis"]


def test_load_tasks_preserves_expectations(tmp_path: Path) -> None:
    yaml_path = tmp_path / "tasks.yaml"
    yaml_path.write_text(
        "id: exp_task\ntask: Check expectations\nexpectations:\n  grounding_required: true\n  required_terms:\n    - ToolPlanRuntime\n",
        encoding="utf-8",
    )
    tasks = _load_tasks(yaml_path)
    assert tasks[0]["expectations"]["grounding_required"] is True
    assert tasks[0]["expectations"]["required_terms"] == ["ToolPlanRuntime"]


def test_load_tasks_from_yaml_list(tmp_path: Path) -> None:
    yaml_path = tmp_path / "tasks.yaml"
    yaml_path.write_text(
        "- id: a\n  task: task a\n- id: b\n  task: task b\n",
        encoding="utf-8",
    )
    tasks = _load_tasks(yaml_path)
    assert len(tasks) == 2
    assert tasks[0]["id"] == "a"
    assert tasks[1]["id"] == "b"


def test_load_tasks_from_directory_dedupes_by_id(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "tasks.jsonl"
    jsonl_path.write_text(
        '{"id":"t1","task":"from jsonl","tags":["a"]}\n',
        encoding="utf-8",
    )
    yaml_path = tmp_path / "tasks.yaml"
    yaml_path.write_text(
        "id: t1\ntask: from yaml\ntags:\n  - b\n",
        encoding="utf-8",
    )
    tasks = _load_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0]["id"] == "t1"
    # First source wins
    assert tasks[0]["task"] == "from jsonl"


def test_load_tasks_from_directory_collects_all(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("id: a\ntask: task a\n", encoding="utf-8")
    (tmp_path / "b.yml").write_text("id: b\ntask: task b\n", encoding="utf-8")
    (tmp_path / "c.jsonl").write_text('{"id":"c","task":"task c"}\n', encoding="utf-8")
    tasks = _load_tasks(tmp_path)
    assert len(tasks) == 3
    ids = {t["id"] for t in tasks}
    assert ids == {"a", "b", "c"}


def test_filter_tasks_by_id_and_max_tasks() -> None:
    tasks = [
        {"id": "a", "task": "one"},
        {"id": "b", "task": "two"},
        {"id": "c", "task": "three"},
    ]

    selected = _filter_tasks(tasks, task_ids=["b", "c"], max_tasks=1)

    assert selected == [{"id": "b", "task": "two"}]


def test_filter_tasks_skips_disabled_by_default() -> None:
    tasks = [
        {"id": "a", "task": "one"},
        {"id": "b", "task": "two", "eval_enabled": False},
        {"id": "c", "task": "three", "disabled": True},
    ]

    selected = _filter_tasks(tasks)

    assert selected == [{"id": "a", "task": "one"}]


def test_filter_tasks_can_include_disabled_or_select_by_id() -> None:
    tasks = [
        {"id": "a", "task": "one"},
        {"id": "b", "task": "two", "eval_enabled": False},
    ]

    assert _filter_tasks(tasks, include_disabled=True) == tasks
    assert _filter_tasks(tasks, task_ids=["b"]) == [{"id": "b", "task": "two", "eval_enabled": False}]


def test_filter_tasks_missing_id_raises() -> None:
    with pytest.raises(SystemExit, match="No tasks matched"):
        _filter_tasks([{"id": "a", "task": "one"}], task_ids=["missing"])


def test_filter_tasks_rejects_non_positive_max_tasks() -> None:
    with pytest.raises(SystemExit, match="--max-tasks"):
        _filter_tasks([{"id": "a", "task": "one"}], max_tasks=0)


def test_load_tasks_missing_task_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("id: no_task\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        _load_tasks(yaml_path)


def test_tool_plan_fixture_directory_covers_planned_task_classes() -> None:
    tasks = _load_tasks(Path(__file__).resolve().parents[1] / "evals" / "tool_plan")
    by_id = {task["id"]: task for task in tasks}

    expected_ids = {
        "coding_symlink_read_bug",
        "coding_patch_existing_bug",
        "coding_test_failure_root_cause",
        "repo_runtime_routing",
        "repo_tool_dispatch_persistence",
        "repo_rewoo_lane_frames",
        "repo_tool_dag_execution",
        "config_rewoo_flags",
        "config_tool_dag_flags",
        "debug_wrong_path_fallback",
        "debug_provider_timeout",
        "log_error_frequency",
        "log_model_stream_halt",
        "log_tool_loop_guard",
        "web_research_001",
    }

    assert expected_ids <= set(by_id)
    assert all(isinstance(by_id[task_id].get("expectations"), dict) for task_id in expected_ids)


def test_test_time_scaling_fixture_directory_has_enabled_smoke_gates() -> None:
    fixture_dir = Path(__file__).resolve().parents[1] / "evals" / "test_time_scaling"
    all_tasks = _load_tasks(fixture_dir)
    enabled_tasks = _filter_tasks(all_tasks)
    by_id = {task["id"]: task for task in all_tasks}
    enabled_ids = {task["id"] for task in enabled_tasks}

    assert {"staged_explicit_hard_loop_status", "staged_explicit_hard_file_read"} <= enabled_ids
    assert "staged_read_path_probe" not in enabled_ids
    assert "staged_scaling_runtime_probe" not in enabled_ids
    assert "staged_explicit_hard_file_mutation" not in enabled_ids
    assert by_id["staged_explicit_hard_file_read"]["expectations"]["required_files"] == [
        "src/smallctl/graph/hard_step_detector.py"
    ]
    assert by_id["staged_explicit_hard_file_read"]["test_time_scaling"]["trigger"] == "explicit"
    assert by_id["staged_explicit_hard_file_mutation"]["eval_enabled"] is False
    assert by_id["staged_explicit_hard_file_mutation"]["test_time_scaling"]["policy"] == "sequential_branch"
    assert by_id["staged_explicit_hard_file_mutation"]["expectations"]["required_files"] == [
        ".smallctl/artifacts/tts_file_mutation_probe.txt"
    ]


def test_build_markdown_report_includes_decision_and_gates() -> None:
    report = {
        "summary": {
            "decision": "continue",
            "decision_reason": "planner_valid_rate=1.0; worker_success_rate_mean=1.0",
            "total_comparisons": 1,
            "repeat_count": 2,
            "planner_valid_gate": True,
            "worker_success_gate": True,
            "solver_grounded_gate": True,
            "tool_plan_abort_not_worse": True,
            "tool_plan_latency_not_worse": True,
            "wrong_path_fallback_ok": True,
        },
        "by_tag": {"repo_analysis": {"total_comparisons": 1, "tool_plan_wins": 1}},
        "prompt_shape_failures": [],
        "comparisons": [
            {"task_id": "t1", "repeat_index": 1, "task": "read src", "loop_success": True, "tool_plan_success": False, "expectation_success": False, "solver_grounded": False},
        ],
    }
    markdown = _build_markdown_report(report)
    assert "Decision: **continue**" in markdown
    assert "Planner validity >= 90%" in markdown
    assert "`repo_analysis`" in markdown


def test_build_test_time_scaling_report_compares_pass_at_1_to_pass_at_n() -> None:
    results = [
        {
            "task_id": "hard-step",
            "repeat_index": 1,
            "repeat_total": 1,
            "tags": ["coding"],
            "task": "fix a hard staged step",
            "mode": "staged_baseline",
            "duration_sec": 5.0,
            "returncode": 0,
            "timed_out": False,
            "final_success": False,
            "final_json": {"status": "failed", "token_usage": 100},
        },
        {
            "task_id": "hard-step",
            "repeat_index": 1,
            "repeat_total": 1,
            "tags": ["coding"],
            "task": "fix a hard staged step",
            "mode": "staged_scaled",
            "duration_sec": 12.0,
            "returncode": 0,
            "timed_out": False,
            "final_success": True,
            "final_json": {
                "status": "completed",
                "token_usage": 260,
                "recovery_metrics": {
                    "test_time_scaling_attempts": 1,
                    "test_time_scaling_candidates": 3,
                    "test_time_scaling_last": {
                        "policy": "proposal_then_execute",
                        "selected_candidate": 2,
                        "selected_score": 0.94,
                        "failed_criteria": [],
                    },
                },
            },
        },
    ]

    report = _build_report(results)

    assert report["summary"]["comparison"] == "test_time_scaling"
    assert report["summary"]["pass_at_1"] == 0.0
    assert report["summary"]["pass_at_n"] == 1.0
    assert report["summary"]["scaled_wins"] == 1
    assert report["summary"]["scaling_attempts"] == 1
    assert report["summary"]["candidate_count_total"] == 3
    assert report["summary"]["decision"] == "continue"
    comparison = report["comparisons"][0]
    assert comparison["selected_candidate"] == 2
    assert comparison["selected_score"] == 0.94
    assert comparison["token_delta"] == 160
    assert report["by_tag"]["coding"]["pass_at_n"] == 1.0


def test_build_test_time_scaling_report_infers_branch_attempt_from_candidates() -> None:
    results = [
        {
            "task_id": "branch-step",
            "repeat_index": 1,
            "repeat_total": 1,
            "tags": ["local_mutation"],
            "task": "write a local probe",
            "mode": "staged_baseline",
            "duration_sec": 5.0,
            "returncode": 0,
            "timed_out": False,
            "final_success": True,
            "final_json": {"status": "completed", "token_usage": 100},
        },
        {
            "task_id": "branch-step",
            "repeat_index": 1,
            "repeat_total": 1,
            "tags": ["local_mutation"],
            "task": "write a local probe",
            "mode": "staged_scaled",
            "duration_sec": 20.0,
            "returncode": 0,
            "timed_out": False,
            "final_success": True,
            "final_json": {
                "status": "completed",
                "token_usage": 130,
                "recovery_metrics": {
                    "test_time_scaling_candidates": 2,
                    "test_time_scaling_branch_successes": 1,
                    "test_time_scaling_isolated_branch_attempts": 1,
                    "test_time_scaling_last": {"policy": "sequential_branch", "selected_candidate": 1},
                },
            },
        },
    ]

    report = _build_report(results)

    assert report["summary"]["scaling_attempts"] == 1
    assert report["summary"]["scaling_attempt_rate"] == 1.0
    assert report["summary"]["isolated_branch_attempts"] == 1
    assert report["comparisons"][0]["scaled_attempted"] is True
    assert report["comparisons"][0]["isolated_branch_attempts"] == 1
    assert report["by_tag"]["local_mutation"]["isolated_branch_attempts"] == 1


def test_test_time_scaling_abort_rate_ignores_timeout_after_success_payload() -> None:
    results = [
        {
            "task_id": "slow-tts",
            "repeat_index": 1,
            "repeat_total": 1,
            "tags": ["test_time_scaling"],
            "task": "finish staged eval just before timeout",
            "mode": "staged_baseline",
            "duration_sec": 300.0,
            "returncode": 124,
            "timed_out": True,
            "final_success": True,
            "final_json": {"status": "completed", "token_usage": 100},
        },
        {
            "task_id": "slow-tts",
            "repeat_index": 1,
            "repeat_total": 1,
            "tags": ["test_time_scaling"],
            "task": "finish staged eval just before timeout",
            "mode": "staged_scaled",
            "duration_sec": 300.0,
            "returncode": 124,
            "timed_out": True,
            "final_success": True,
            "final_json": {
                "status": "completed",
                "token_usage": 120,
                "recovery_metrics": {"test_time_scaling_attempts": 1},
            },
        },
    ]

    report = _build_report(results)

    assert report["summary"]["baseline_abort_rate"] == 0.0
    assert report["summary"]["scaled_abort_rate"] == 0.0
    assert report["summary"]["scaled_abort_not_worse"] is True
    assert report["summary"]["baseline_timeout_after_success_count"] == 1
    assert report["summary"]["scaled_timeout_after_success_count"] == 1
    assert report["comparisons"][0]["baseline_timeout_after_success"] is True
    assert report["comparisons"][0]["scaled_timeout_after_success"] is True
    assert report["by_tag"]["test_time_scaling"]["timeout_after_success_count"] == 1
    assert "scaled_timeout_after_success_count=1" in report["summary"]["decision_reason"]
    assert "baseline_timeout_after_success_count=1" in report["summary"]["decision_reason"]


def test_test_time_scaling_markdown_report_uses_scaling_terms() -> None:
    report = {
        "summary": {
            "comparison": "test_time_scaling",
            "decision": "continue",
            "decision_reason": "pass_at_1=0.0; pass_at_n=1.0",
            "total_comparisons": 1,
            "scaled_success_not_worse": True,
            "scaled_abort_not_worse": True,
            "scaled_latency_reasonable": True,
            "isolated_branch_attempts": 1,
            "scaled_timeout_after_success_count": 1,
        },
        "by_tag": {
            "coding": {
                "total_comparisons": 1,
                "scaled_wins": 1,
                "isolated_branch_attempts": 1,
                "timeout_after_success_count": 1,
            }
        },
        "comparisons": [
            {
                "task_id": "hard-step",
                "repeat_index": 1,
                "task": "fix a hard staged step",
                "baseline_success": False,
                "scaled_success": True,
                "scaling_attempts": 1,
                "selected_candidate": 2,
                "selected_score": 0.94,
            }
        ],
    }

    markdown = _build_markdown_report(report)

    assert "# Test-Time Scaling Eval Report" in markdown
    assert "Scaled Pass@N no worse than baseline Pass@1" in markdown
    assert "`coding`" in markdown
    assert "`isolated_branch_attempts`: 1" in markdown
    assert "`scaled_timeout_after_success_count`: 1" in markdown
    assert "`timeout_after_success_count`: 1" in markdown
