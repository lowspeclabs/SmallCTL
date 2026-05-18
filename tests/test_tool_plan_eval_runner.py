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
_report_exit_code = _tool_plan_eval._report_exit_code
_summary = _tool_plan_eval._summary
_load_tasks = _tool_plan_eval._load_tasks
_command_for = _tool_plan_eval._command_for
_text_output = _tool_plan_eval._text_output
_prompt_shape_assertions = _tool_plan_eval._prompt_shape_assertions


def test_extract_metrics_reads_recovery_metrics() -> None:
    final_json = {
        "status": "completed",
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
    assert metrics["tool_plan_parse_failures"] == 0


def test_extract_metrics_returns_zeros_when_missing() -> None:
    metrics = _extract_metrics({"status": "completed"})
    assert all(metrics.get(k, -1) == 0 for k in [
        "tool_plan_invocations",
        "tool_plan_parse_failures",
        "tool_plan_steps_executed",
    ])


def test_extract_metrics_handles_none() -> None:
    metrics = _extract_metrics(None)
    assert all(metrics.get(k, -1) == 0 for k in [
        "tool_plan_invocations",
        "tool_plan_parse_failures",
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
        "final_json": {"status": "completed"},
        "final_success": True,
        "tool_plan_total_tokens": 120,
        "tool_plan_steps_executed": 2,
    }
    summary = _summary(result)
    assert summary["status"] == "completed"
    assert summary["final_success"] is True
    assert summary["tool_plan_total_tokens"] == 120
    assert summary["tool_plan_steps_executed"] == 2


def test_build_report_computes_comparisons() -> None:
    results = [
        {
            "task_id": "t1",
            "tags": ["repo_analysis"],
            "task": "read src",
            "mode": "loop",
            "duration_sec": 10.0,
            "final_success": True,
            "tool_plan_total_tokens": 0,
        },
        {
            "task_id": "t1",
            "tags": ["repo_analysis"],
            "task": "read src",
            "mode": "tool_plan",
            "duration_sec": 5.0,
            "final_success": True,
            "tool_plan_total_tokens": 200,
            "tool_plan_steps_requested": 3,
            "tool_plan_steps_executed": 3,
            "tool_plan_fallback_count": 0,
            "prompt_shape": {
                "planner_has_rewoo_plan_state": True,
                "planner_excludes_tool_observations": True,
                "solver_has_rewoo_evidence": True,
                "solver_excludes_generic_warm_summaries": True,
                "solver_has_tool_plan_evidence_ids": True,
            },
        },
    ]
    report = _build_report(results)
    assert report["total_comparisons"] == 1
    assert report["both_pass"] == 1
    assert report["tool_plan_wins"] == 0
    assert report["loop_wins"] == 0
    assert report["token_savings"] == -200
    assert len(report["comparisons"]) == 1
    c = report["comparisons"][0]
    assert c["loop_duration_sec"] == 10.0
    assert c["tool_plan_duration_sec"] == 5.0
    assert c["prompt_shape"]["solver_has_rewoo_evidence"] is True


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
            "tool_plan_total_tokens": 0,
        },
        {
            "task_id": "wp1",
            "tags": ["wrong_path", "safety"],
            "task": "read /etc",
            "mode": "tool_plan",
            "duration_sec": 2.0,
            "final_success": True,
            "tool_plan_fallback_count": 0,
        },
    ]
    report = _build_report(results)
    assert report["wrong_path_tasks"] == 1
    assert report["wrong_path_fallback_ok"] is False


def test_report_exit_code_zero_when_ok() -> None:
    assert _report_exit_code({"wrong_path_fallback_ok": True}) == 0


def test_report_exit_code_nonzero_when_wrong_path_fails() -> None:
    assert _report_exit_code({"wrong_path_fallback_ok": False}) == 1



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


def test_load_tasks_missing_task_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("id: no_task\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        _load_tasks(yaml_path)
