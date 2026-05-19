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
_text_output = _tool_plan_eval._text_output
_prompt_shape_assertions = _tool_plan_eval._prompt_shape_assertions
_last_json_object = _tool_plan_eval._last_json_object


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


def test_report_exit_code_zero_when_ok() -> None:
    assert _report_exit_code({"wrong_path_fallback_ok": True}) == 0


def test_report_exit_code_nonzero_when_wrong_path_fails() -> None:
    assert _report_exit_code({"wrong_path_fallback_ok": False}) == 1


def test_report_exit_code_nonzero_when_decision_is_not_continue() -> None:
    assert _report_exit_code({"wrong_path_fallback_ok": True, "summary": {"decision": "narrow rollout"}}) == 1



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
