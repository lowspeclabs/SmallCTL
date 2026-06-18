from __future__ import annotations

from typing import Any

from .tool_plan_schema import ToolPlan, ToolPlanStep

_MUTATING_SOLVER_TOOLS = {
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
    "file_delete",
    "shell_exec",
    "ssh_exec",
    "ssh_file_write",
    "ssh_file_patch",
    "ansible",
}
_TERMINAL_SOLVER_TOOLS = {"task_complete", "task_fail"}


def _tool_plan_config(deps: Any, name: str, default: Any) -> Any:
    config = getattr(deps.harness, "config", None)
    return getattr(config, name, default)


def _rewoo_role_enabled(deps: Any, role_flag: str) -> bool:
    return bool(_tool_plan_config(deps, "rewoo_lane_frames_enabled", False)) or bool(
        _tool_plan_config(deps, role_flag, False)
    )


def _rewoo_frame_budget(deps: Any) -> int:
    return max(1, int(_tool_plan_config(deps, "rewoo_frame_token_budget", 1200) or 1200))


def _tool_plan_observation_budget(deps: Any) -> tuple[int, int]:
    token_limit = int(_tool_plan_config(deps, "tool_plan_observation_token_limit", 900) or 900)
    max_chars = int(_tool_plan_config(deps, "tool_plan_max_observation_chars_per_step", 600) or 600)
    from ..recovery_metrics import recovery_metrics
    metrics = recovery_metrics(deps.harness.state)
    try:
        max_batch = int(metrics.get("tool_plan_dag_max_batch_size", 1) or 1)
    except (TypeError, ValueError):
        max_batch = 1
    if max_batch >= 3:
        token_limit = int(token_limit * 0.85)
        max_chars = int(max_chars * 0.80)
    return max(1, token_limit), max(1, max_chars)


def _select_no_tools(graph_state: Any, deps: Any) -> list[dict[str, Any]]:
    del graph_state, deps
    return []


def _select_solver_tools(graph_state: Any, deps: Any) -> list[dict[str, Any]]:
    from .lifecycle_nodes import select_loop_tools
    tools = select_loop_tools(graph_state, deps)
    selected: list[dict[str, Any]] = []
    for schema in tools:
        function = schema.get("function") if isinstance(schema, dict) else None
        name = str(function.get("name") or "").strip() if isinstance(function, dict) else ""
        if name in _TERMINAL_SOLVER_TOOLS:
            selected.append(schema)
    return selected


def _coerce_tool_plan(value: Any) -> ToolPlan | None:
    if isinstance(value, ToolPlan):
        return value
    if not isinstance(value, dict):
        return None
    raw_steps = value.get("steps")
    if not isinstance(raw_steps, list):
        return None
    steps: list[ToolPlanStep] = []
    for raw in raw_steps:
        if not isinstance(raw, dict):
            return None
        args = raw.get("args")
        if not isinstance(args, dict):
            args = {}
        depends_on = raw.get("depends_on")
        if not isinstance(depends_on, list):
            depends_on = []
        steps.append(
            ToolPlanStep(
                id=str(raw.get("id") or ""),
                tool=str(raw.get("tool") or ""),
                args=dict(args),
                reason=str(raw.get("reason") or ""),
                depends_on=[str(item) for item in depends_on],
                optional=bool(raw.get("optional", False)),
            )
        )
    return ToolPlan(
        mode="tool_plan",
        objective=str(value.get("objective") or ""),
        steps=steps,
        max_steps=int(value.get("max_steps") or 6),
    )


def _compact_evidence_text(text: str, *, limit: int = 240) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    compact = " | ".join(lines)
    return compact[:limit]


def _usage_token_count(usage: dict[str, Any]) -> int:
    for key in ("total_tokens", "tokens", "total"):
        try:
            value = int(usage.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    total = 0
    for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"):
        try:
            total += int(usage.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
    return max(0, total)
