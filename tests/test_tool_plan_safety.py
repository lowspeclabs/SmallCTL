from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.tool_plan_safety import validate_tool_plan
from smallctl.graph.tool_plan_schema import ToolPlan, ToolPlanStep


def _harness(tmp_path, names: list[str] | None = None):
    return SimpleNamespace(
        state=SimpleNamespace(cwd=str(tmp_path)),
        registry=SimpleNamespace(
            names=lambda: names
            or ["file_read", "dir_list", "grep", "find_files", "web_fetch", "ssh_file_read", "git_status", "git_diff", "read_log"]
        ),
    )


def test_validate_tool_plan_allows_workspace_read_tools(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect",
        steps=[
            ToolPlanStep("E1", "file_read", {"path": "src/app.py"}),
            ToolPlanStep("E2", "grep", {"path": "src", "pattern": "dispatch"}),
        ],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path))

    assert safe is plan
    assert errors == []


def test_validate_tool_plan_blocks_absolute_and_parent_paths(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect",
        steps=[
            ToolPlanStep("E1", "file_read", {"path": "/etc/passwd"}),
            ToolPlanStep("E2", "grep", {"path": "../outside", "pattern": "x"}),
        ],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path))

    assert safe is None
    assert len(errors) == 2


def test_validate_tool_plan_blocks_web_when_disabled(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="fetch",
        steps=[ToolPlanStep("E1", "web_fetch", {"url": "https://example.com"})],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path), allow_web=False)

    assert safe is None
    assert "web tools are disabled" in errors[0]


def test_validate_tool_plan_allows_ssh_file_read_remote_absolute_path(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect remote config",
        steps=[
            ToolPlanStep(
                "E1",
                "ssh_file_read",
                {"target": "root@example.test", "path": "/etc/nginx/nginx.conf"},
            )
        ],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path))

    assert safe is plan
    assert errors == []


def test_validate_tool_plan_does_not_apply_local_path_rules_to_ssh_file_read(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect remote relative path",
        steps=[ToolPlanStep("E1", "ssh_file_read", {"target": "root@example.test", "path": "../remote/app.log"})],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path))

    assert safe is plan
    assert errors == []


def test_validate_tool_plan_requires_ssh_file_read_path(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect remote config",
        steps=[ToolPlanStep("E1", "ssh_file_read", {"target": "root@example.test"})],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path))

    assert safe is None
    assert any("ssh_file_read requires a non-empty remote path" in error for error in errors)


def test_validate_tool_plan_allows_read_log_in_workspace(tmp_path) -> None:
    log = tmp_path / "app.log"
    log.write_text("hello")
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect logs",
        steps=[ToolPlanStep("E1", "read_log", {"path": "app.log", "lines": 50})],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path))

    assert safe is plan
    assert errors == []


def test_validate_tool_plan_blocks_read_log_absolute_path(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect logs",
        steps=[ToolPlanStep("E1", "read_log", {"path": "/var/log/syslog"})],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path))

    assert safe is None
    assert any("path must be a relative path inside the workspace" in error for error in errors)


def test_validate_tool_plan_allows_git_tools_when_enabled(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect git state",
        steps=[
            ToolPlanStep("E1", "git_status", {"path": "."}),
            ToolPlanStep("E2", "git_diff", {"path": ".", "cached": True}),
        ],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path), allow_git=True)

    assert safe is plan
    assert errors == []


def test_validate_tool_plan_blocks_git_status_absolute_path(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect git state",
        steps=[ToolPlanStep("E1", "git_status", {"path": "/etc"})],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path), allow_git=True)

    assert safe is None
    assert any("path must be a relative path inside the workspace" in error for error in errors)


def test_validate_tool_plan_blocks_git_diff_target_absolute_path(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect git diff",
        steps=[ToolPlanStep("E1", "git_diff", {"path": ".", "target": "/etc/passwd"})],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path), allow_git=True)

    assert safe is None
    assert any("target must be a relative path inside the workspace" in error for error in errors)


def test_validate_tool_plan_blocks_git_tools_when_disabled(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect git state",
        steps=[
            ToolPlanStep("E1", "git_status", {"path": "."}),
            ToolPlanStep("E2", "git_diff", {"path": "."}),
        ],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path), allow_git=False)

    assert safe is None
    assert any("git tools are disabled for ToolPlan" in error for error in errors)
