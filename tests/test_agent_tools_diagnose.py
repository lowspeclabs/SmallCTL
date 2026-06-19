from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Agent-Tools"))

from run_diagnose import _classify_failure, _file_patch_target_loop_count
from runscan import _classify_run


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
