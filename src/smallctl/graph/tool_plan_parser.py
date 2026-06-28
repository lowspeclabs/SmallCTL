from __future__ import annotations

import json
import re
from typing import Any

from .tool_plan_schema import (
    MUTATING_TOOL_PLAN_BLOCKLIST,
    READONLY_TOOL_PLAN_TOOLS,
    TOOL_PLAN_ALIASES,
    ToolPlan,
    ToolPlanStep,
)


def _extract_fenced_json(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else None


def _extract_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _load_plan_payload(text: str) -> dict[str, Any] | None:
    candidate = _extract_fenced_json(text) or _extract_balanced_json(text)
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def summarize_tool_plan_json(text: str, *, max_steps: int = 12) -> str:
    """Return a human-readable summary for ToolPlan-shaped JSON, even if invalid.

    This is intentionally looser than parse_tool_plan: it is for UI diagnostics
    and fallback chat, not for execution.
    """
    payload = _load_plan_payload(text)
    if not payload or payload.get("mode") != "tool_plan":
        return ""
    objective = str(payload.get("objective") or "").strip()
    raw_steps = payload.get("steps")
    lines: list[str] = []
    if objective:
        lines.append(f"ToolPlan objective: {objective}")
    if isinstance(raw_steps, list) and raw_steps:
        lines.append("Proposed steps:")
        for index, raw_step in enumerate(raw_steps[:max_steps], start=1):
            if not isinstance(raw_step, dict):
                lines.append(f"{index}. Invalid step payload")
                continue
            tool = str(raw_step.get("tool") or "unknown_tool").strip() or "unknown_tool"
            args = raw_step.get("args")
            target = ""
            if isinstance(args, dict):
                for key in ("path", "target_path", "target", "url", "query", "pattern"):
                    value = args.get(key)
                    if isinstance(value, str) and value.strip():
                        target = value.strip()
                        break
            reason = str(raw_step.get("reason") or "").strip()
            detail = f"{tool}"
            if target:
                detail += f" on {target}"
            if reason:
                detail += f" - {reason}"
            lines.append(f"{index}. {detail}")
        if len(raw_steps) > max_steps:
            lines.append(f"... {len(raw_steps) - max_steps} more step(s) omitted")
    elif isinstance(raw_steps, list):
        lines.append("Proposed steps: none")
    return "\n".join(lines).strip()


def _normalize_step_id(raw: Any, index: int) -> str:
    text = str(raw or "").strip().upper()
    if re.fullmatch(r"E[1-9][0-9]*", text):
        return text
    return f"E{index}"


def _normalize_tool(raw: Any) -> str:
    tool = str(raw or "").strip()
    return TOOL_PLAN_ALIASES.get(tool, tool)


def _normalize_depends_on(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    normalized: list[str] = []
    for item in raw:
        text = str(item or "").strip().upper()
        if text:
            normalized.append(text)
    return normalized


def parse_tool_plan(text: str, *, max_steps: int = 6) -> ToolPlan | None:
    payload = _load_plan_payload(text)
    if not payload or payload.get("mode") != "tool_plan":
        return None
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not (0 <= len(raw_steps) <= max_steps):
        return None

    steps: list[ToolPlanStep] = []
    seen_ids: set[str] = set()
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            return None
        step_id = _normalize_step_id(raw_step.get("id"), index)
        if step_id in seen_ids:
            return None
        seen_ids.add(step_id)
        tool = _normalize_tool(raw_step.get("tool"))
        if tool in MUTATING_TOOL_PLAN_BLOCKLIST:
            return None
        if tool not in READONLY_TOOL_PLAN_TOOLS:
            return None
        args = raw_step.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None
        steps.append(
            ToolPlanStep(
                id=step_id,
                tool=tool,
                args=dict(args),
                reason=str(raw_step.get("reason") or "").strip(),
                depends_on=_normalize_depends_on(raw_step.get("depends_on")),
                optional=bool(raw_step.get("optional", False)),
            )
        )

    valid_ids = {step.id for step in steps}
    for step in steps:
        if any(dep not in valid_ids for dep in step.depends_on):
            return None
    return ToolPlan(
        mode="tool_plan",
        objective=str(payload.get("objective") or "").strip(),
        steps=steps,
        max_steps=max_steps,
    )
