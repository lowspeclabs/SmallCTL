from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from smallctl.plans import resolve_plan_export_target, write_plan_file
from smallctl.state import ExecutionPlan, LoopState
from smallctl.tools.planning import plan_export, plan_set


def test_resolve_plan_export_target_accepts_workspace_relative_path(tmp_path) -> None:
    output_path, fmt = resolve_plan_export_target(
        "plan.md", format="markdown", cwd=str(tmp_path)
    )
    assert output_path == tmp_path / "plan.md"
    assert fmt == "markdown"


def test_resolve_plan_export_target_rejects_absolute_path(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be relative to the active workspace"):
        resolve_plan_export_target(
            "/plan/file.md", format="markdown", cwd=str(tmp_path)
        )


def test_resolve_plan_export_target_rejects_path_traversal(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be relative to the active workspace"):
        resolve_plan_export_target(
            "../outside.md", format="markdown", cwd=str(tmp_path)
        )


def test_write_plan_file_creates_relative_export(tmp_path) -> None:
    plan = ExecutionPlan(
        plan_id="plan-test",
        goal="test goal",
        outputs=["output"],
        acceptance_criteria=["criterion"],
        implementation_plan=["step"],
    )
    content = write_plan_file(
        plan, "subdir/plan.md", format="markdown", cwd=str(tmp_path)
    )
    written = tmp_path / "subdir" / "plan.md"
    assert written.exists()
    assert content in written.read_text(encoding="utf-8")


def test_write_plan_file_rejects_absolute_export(tmp_path) -> None:
    plan = ExecutionPlan(
        plan_id="plan-test",
        goal="test goal",
        outputs=["output"],
        acceptance_criteria=["criterion"],
        implementation_plan=["step"],
    )
    with pytest.raises(ValueError, match="must be relative to the active workspace"):
        write_plan_file(plan, "/plan/file.md", format="markdown", cwd=str(tmp_path))


def test_plan_set_normalizes_absolute_output_path_to_none(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    harness = SimpleNamespace(
        state=state,
        subtask_ledger=None,
        log=SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None),
    )

    result = asyncio.run(
        plan_set(
            goal="diagnose permissions",
            summary="summary",
            outputs=["report"],
            acceptance_criteria=["find root cause"],
            implementation_plan=["run diagnostics"],
            steps=[{"title": "step", "description": "desc", "task": "task"}],
            plan_output_path="/plan/file_permissions_diagnostics.md",
            plan_output_format="markdown",
            state=state,
            harness=harness,
        )
    )

    assert result["success"] is True
    assert state.draft_plan is not None
    assert state.draft_plan.requested_output_path is None
    assert (
        result["metadata"].get("rejected_output_path")
        == "/plan/file_permissions_diagnostics.md"
    )
    assert "export_warning" in result["metadata"]


def test_plan_set_keeps_workspace_relative_output_path(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    harness = SimpleNamespace(
        state=state,
        subtask_ledger=None,
        log=SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None),
    )

    result = asyncio.run(
        plan_set(
            goal="diagnose permissions",
            summary="summary",
            outputs=["report"],
            acceptance_criteria=["find root cause"],
            implementation_plan=["run diagnostics"],
            steps=[{"title": "step", "description": "desc", "task": "task"}],
            plan_output_path="plans/file_permissions_diagnostics.md",
            plan_output_format="markdown",
            state=state,
            harness=harness,
        )
    )

    assert result["success"] is True
    assert state.draft_plan is not None
    assert (
        state.draft_plan.requested_output_path
        == "plans/file_permissions_diagnostics.md"
    )
    assert "export_warning" not in result["metadata"]
    # plan_set does not write the file itself; downstream nodes do. Verify the
    # export path is valid by writing it explicitly.
    written = write_plan_file(
        state.draft_plan,
        state.draft_plan.requested_output_path,
        format=state.draft_plan.requested_output_format,
        cwd=str(tmp_path),
    )
    assert (tmp_path / "plans" / "file_permissions_diagnostics.md").exists()
    assert written in (
        tmp_path / "plans" / "file_permissions_diagnostics.md"
    ).read_text(encoding="utf-8")


def test_plan_export_survives_permission_denied(tmp_path, monkeypatch) -> None:
    """plan_export must not crash when the workspace parent is unwritable."""
    state = LoopState(cwd=str(tmp_path))
    state.draft_plan = ExecutionPlan(
        plan_id="plan-test",
        goal="test goal",
        outputs=["output"],
        acceptance_criteria=["criterion"],
        implementation_plan=["step"],
        requested_output_path="plans/plan.md",
        requested_output_format="markdown",
    )
    harness = SimpleNamespace(
        state=state,
        subtask_ledger=None,
        artifact_store=SimpleNamespace(
            persist_generated_text=lambda **kwargs: SimpleNamespace(
                artifact_id="A1", summary="playbook"
            )
        ),
        log=SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None),
    )

    # Make Path.mkdir raise PermissionError to simulate an unwritable parent.
    def _raising_mkdir(self, *args, **kwargs):
        raise PermissionError("read-only filesystem")

    monkeypatch.setattr(Path, "mkdir", _raising_mkdir)

    result = asyncio.run(
        plan_export(
            path="plans/plan.md",
            format="markdown",
            state=state,
            harness=harness,
        )
    )

    assert result["success"] is False
    assert "Failed to export plan" in result["error"]

