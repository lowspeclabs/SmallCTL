from __future__ import annotations

import json
import logging
from typing import Any

from ..models.events import UIEvent, UIEventType
from ..state import clip_text_value

_CHECKLIST_LOGGER = logging.getLogger("smallctl.subtask_checklist")


_CHECKLIST_DIGEST_KEY = "_subtask_checklist_digest"
_GENERIC_TITLES = {"Complete user task", "Continue user task", "Follow latest user direction"}


def render_subtask_checklist(state: Any, *, max_items: int = 12) -> str:
    ledger = getattr(state, "subtask_ledger", None)
    subtasks = list(getattr(ledger, "subtasks", []) or [])
    if not subtasks:
        return ""

    lines: list[str] = []
    goal = _goal_text(state)
    if goal:
        lines.append(f"Task Objective: {goal}")

    # Separate synthetic roots from real subtasks.
    # When real subtasks exist we skip synthetic roots so the checklist
    # shows meaningful plan steps. When there are *only* synthetic roots
    # (e.g. non-plan execution) we render them so the widget isn't empty.
    synthetic_roots = [task for task in subtasks[-max_items:] if _is_synthetic_root(task)]
    real_tasks = [task for task in subtasks[-max_items:] if not _is_synthetic_root(task)]
    tasks_to_render = real_tasks if real_tasks else synthetic_roots

    rendered_tasks = 0
    skipped_tasks = len(synthetic_roots) if real_tasks else 0
    goal_key = _normalize_title(goal)
    seen_titles: set[str] = set()
    for task in tasks_to_render:
        display_title = _display_title(task)
        title_key = _normalize_title(display_title)
        if goal_key and title_key == goal_key:
            skipped_tasks += 1
            continue
        if title_key and title_key in seen_titles:
            skipped_tasks += 1
            continue
        if title_key:
            seen_titles.add(title_key)
        symbol = _status_symbol(str(getattr(task, "status", "") or "pending"))
        title = _clip_line(display_title)
        lines.append(f"  {symbol} {title}")
        rendered_tasks += 1

    result = "\n".join(lines)
    _CHECKLIST_LOGGER.debug(
        "render_subtask_checklist: goal=%r tasks=%d skipped=%d len=%d",
        goal, rendered_tasks, skipped_tasks, len(result),
    )
    return result


def build_subtask_checklist_update(state: Any) -> str:
    text = render_subtask_checklist(state)
    if not text:
        return ""
    digest = _subtask_checklist_digest(state)
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return text
    if scratchpad.get(_CHECKLIST_DIGEST_KEY) == digest:
        return ""
    scratchpad[_CHECKLIST_DIGEST_KEY] = digest
    return text


async def emit_subtask_checklist_if_changed(harness: Any, event_handler: Any) -> None:
    if harness is None or event_handler is None:
        return
    text = build_subtask_checklist_update(getattr(harness, "state", None))
    if not text:
        return
    lines = text.splitlines()
    title = ""
    if lines and lines[0].startswith("Task Objective: "):
        title = _clip_line(lines[0][16:].strip(), limit=40)
    emit = getattr(harness, "_emit", None)
    if not callable(emit):
        _CHECKLIST_LOGGER.warning("emit_subtask_checklist_if_changed: no _emit on harness")
        return
    _CHECKLIST_LOGGER.debug(
        "emit_subtask_checklist: title=%r content_len=%d",
        title, len(text),
    )
    await emit(
        event_handler,
        UIEvent(
            event_type=UIEventType.ALERT,
            content=text,
            data={
                "ui_kind": "subtask_checklist",
                "checklist_title": title,
            },
        ),
    )


def _goal_text(state: Any) -> str:
    plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
    if plan is not None:
        text = str(getattr(plan, "goal", "") or "").strip()
        if text:
            return _clip_line(text, limit=80)
    run_brief = getattr(state, "run_brief", None)
    text = str(getattr(run_brief, "current_phase_objective", "") or "").strip()
    if not text:
        text = str(getattr(run_brief, "original_task", "") or "").strip()
    return _clip_line(text, limit=80) if text else ""


def _subtask_checklist_digest(state: Any) -> str:
    ledger = getattr(state, "subtask_ledger", None)
    subtasks = []
    for task in list(getattr(ledger, "subtasks", []) or []):
        subtasks.append(
            {
                "id": str(getattr(task, "subtask_id", "") or ""),
                "title": str(getattr(task, "title", "") or ""),
                "display_title": _display_title(task),
                "goal": str(getattr(task, "goal", "") or ""),
                "status": str(getattr(task, "status", "") or ""),
                "next_action": str(getattr(task, "next_action", "") or ""),
                "attempts": int(getattr(task, "attempts", 0) or 0),
            }
        )
    payload = {
        "active_subtask_id": str(getattr(ledger, "active_subtask_id", "") or ""),
        "goal": _goal_text(state),
        "subtasks": subtasks,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _status_symbol(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"done", "completed", "pass", "passed"}:
        return "✓"
    if normalized in {"active", "in_progress", "running"}:
        return "○"
    if normalized in {"blocked"}:
        return "⚠"
    if normalized in {"failed", "fail"}:
        return "✗"
    if normalized in {"abandoned", "skipped"}:
        return "⊘"
    return "○"

def _display_title(task: Any) -> str:
    """Return the best human-readable title for a subtask.

    When the title is a generic placeholder (e.g. "Complete user task"),
    fall back to the goal field which contains the actual task description.
    """
    title = str(getattr(task, "title", "") or "").strip()
    if title and title not in _GENERIC_TITLES:
        return title
    goal = str(getattr(task, "goal", "") or "").strip()
    return goal or title or "Untitled task"


def _is_synthetic_root(task: Any) -> bool:
    return (
        str(getattr(task, "subtask_id", "") or "") == "S1"
        and str(getattr(task, "title", "") or "") == "Complete user task"
        and str(getattr(task, "goal", "") or "").strip()
        and list(getattr(task, "acceptance", []) or [])
        == ["User request satisfied with tool-backed evidence when needed."]
    )


def _clip_line(text: str, *, limit: int = 80) -> str:
    clipped, was_clipped = clip_text_value(" ".join(str(text or "").split()), limit=limit)
    return f"{clipped}..." if was_clipped else clipped


def _normalize_title(text: str) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())
    for prefix in ("execute:", "planning:", "repair:", "verify:", "verification:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    return normalized
