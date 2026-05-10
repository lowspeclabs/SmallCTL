from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .request_budget import RequestBudget, RequestEstimator, RequestFootprint

_CONTROL_TOOLS = {"ask_human", "log_note", "loop_status", "memory_update"}
_REMOTE_READ_TOOLS = {"ssh_exec", "ssh_file_read"}
_SSH_MUTATION_TOOLS = {"ssh_file_patch", "ssh_file_write", "ssh_file_replace_between"}
_LOCAL_MUTATION_TOOLS = {"file_patch", "file_write", "ast_patch"}
_COMPLETION_TOOLS = {"task_complete", "task_fail", "step_complete", "step_fail"}


@dataclass(frozen=True)
class ToolBudgetResult:
    payload: dict[str, Any]
    action: str
    footprint: RequestFootprint
    tool_count_before: int
    tool_count_after: int
    dropped_tool_names: tuple[str, ...]
    kept_tool_names: tuple[str, ...]

    @property
    def fits(self) -> bool:
        return self.action != "exceeded"


def tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()


def _is_index_or_plan_tool(name: str) -> bool:
    return name.startswith("index_") or name.startswith("plan_")


def _protected_tool_names(
    available_names: set[str],
    *,
    requested_tool_name: str,
    mode: str,
) -> set[str]:
    del mode
    protected = set(_CONTROL_TOOLS & available_names)
    protected.update(_REMOTE_READ_TOOLS & available_names)
    protected.update(_COMPLETION_TOOLS & available_names)

    requested = str(requested_tool_name or "").strip()
    if requested in available_names:
        protected.add(requested)
    if requested in _SSH_MUTATION_TOOLS:
        protected.update({"ssh_file_read"} & available_names)
    if requested == "ssh_file_read":
        protected.update({"ssh_exec", "ssh_file_read"} & available_names)
    if requested in _LOCAL_MUTATION_TOOLS:
        protected.update({requested} & available_names)
        protected.update({"file_read"} & available_names)
    return protected


def _drop_priority(name: str, protected_names: set[str]) -> int:
    if name in protected_names:
        return 1000
    if _is_index_or_plan_tool(name):
        return 15
    drop_first = {
        "artifact_read",
        "artifact_grep",
        "artifact_print",
        "web_search",
        "web_fetch",
        "http_get",
        "http_post",
        "dir_tree",
        "dir_list",
        "file_download",
        "file_read",
        "file_write",
        "file_patch",
        "find_files",
        "grep",
        "ast_patch",
        "ssh_file_write",
        "ssh_file_patch",
        "ssh_file_replace_between",
        "shell_exec",
        "process_kill",
    }
    if name in drop_first:
        return 10
    return 50


def _with_tools(payload: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    fitted = dict(payload)
    if tools:
        fitted["tools"] = tools
    else:
        fitted.pop("tools", None)
    fitted.pop("stream_options", None)
    return fitted


def _slim_schema_descriptions(value: Any) -> Any:
    if isinstance(value, list):
        return [_slim_schema_descriptions(item) for item in value]
    if not isinstance(value, dict):
        return value
    slimmed: dict[str, Any] = {}
    for key, item in value.items():
        if key == "description" and isinstance(item, str) and len(item) > 80:
            slimmed[key] = item[:80]
            continue
        slimmed[key] = _slim_schema_descriptions(item)
    return slimmed


def _with_slimmed_tool_descriptions(payload: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    return _with_tools(payload, [_slim_schema_descriptions(tool) for tool in tools])


def _result(
    *,
    payload: dict[str, Any],
    action: str,
    budget: RequestBudget,
    estimator: RequestEstimator,
    original_names: list[str],
) -> ToolBudgetResult:
    kept_names = [tool_name(tool) for tool in payload.get("tools", []) if tool_name(tool)]
    kept_set = set(kept_names)
    dropped_names = [name for name in original_names if name not in kept_set]
    return ToolBudgetResult(
        payload=payload,
        action=action,
        footprint=estimator.footprint(payload, budget),
        tool_count_before=len(original_names),
        tool_count_after=len(kept_names),
        dropped_tool_names=tuple(dropped_names),
        kept_tool_names=tuple(kept_names),
    )


def fit_tools_to_context_budget(
    *,
    payload: dict[str, Any],
    tools: list[dict[str, Any]],
    budget: RequestBudget,
    requested_tool_name: str = "",
    mode: str = "",
    estimator: RequestEstimator,
) -> ToolBudgetResult:
    working_tools = [dict(tool) for tool in tools if isinstance(tool, dict)]
    original_names = [tool_name(tool) for tool in working_tools if tool_name(tool)]
    footprint = estimator.footprint(payload, budget)
    if footprint.over_budget_tokens <= 0:
        return ToolBudgetResult(
            payload=payload,
            action="unchanged",
            footprint=footprint,
            tool_count_before=len(original_names),
            tool_count_after=len(original_names),
            dropped_tool_names=(),
            kept_tool_names=tuple(original_names),
        )

    available_names = set(original_names)
    protected_names = _protected_tool_names(
        available_names,
        requested_tool_name=requested_tool_name,
        mode=mode,
    )
    drop_order = sorted(
        range(len(working_tools)),
        key=lambda index: (_drop_priority(tool_name(working_tools[index]), protected_names), -index),
    )
    kept_indexes = set(range(len(working_tools)))
    current_payload = dict(payload)
    current_tools = list(working_tools)

    for index in drop_order:
        name = tool_name(working_tools[index])
        if name in protected_names:
            continue
        kept_indexes.discard(index)
        current_tools = [tool for i, tool in enumerate(working_tools) if i in kept_indexes]
        current_payload = _with_tools(payload, current_tools)
        current_footprint = estimator.footprint(current_payload, budget)
        if current_footprint.over_budget_tokens <= 0:
            return _result(
                payload=current_payload,
                action="reduced_tools",
                budget=budget,
                estimator=estimator,
                original_names=original_names,
            )

    required_tools = [tool for tool in working_tools if tool_name(tool) in protected_names]
    current_payload = _with_tools(payload, required_tools)
    action = "reduced_tools"
    if estimator.footprint(current_payload, budget).over_budget_tokens > 0:
        slimmed_payload = _with_slimmed_tool_descriptions(payload, required_tools)
        if estimator.footprint(slimmed_payload, budget).over_budget_tokens <= 0:
            current_payload = slimmed_payload
        else:
            current_payload = slimmed_payload
            action = "exceeded"
    return _result(
        payload=current_payload,
        action=action,
        budget=budget,
        estimator=estimator,
        original_names=original_names,
    )
