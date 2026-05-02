from __future__ import annotations

from pathlib import Path
from typing import Any

from .state import ToolExecutionRecord


def _is_plan_export_validation_error(error: str | None) -> bool:
    normalized = str(error or "").strip().lower()
    if not normalized:
        return False
    markers = (
        "refusing to write a plan",
        "plan export targets must use .md, .txt, or .text",
        "plan export path ending in .md requires markdown format",
        "plan export path ending in .txt/.text requires text format",
        "unsupported plan export format",
    )
    return any(marker in normalized for marker in markers)


def _build_plan_export_recovery_message(record: ToolExecutionRecord) -> str:
    requested_path = str(
        record.args.get("path")
        or record.args.get("output_path")
        or record.args.get("plan_output_path")
        or ""
    ).strip()
    if requested_path:
        suggested_path = str(Path(requested_path).with_suffix(".md"))
        return (
            "Plan export paths are only for plan documents (.md, .txt, .text). "
            f"Keep implementation targets like `{requested_path}` out of `plan_set` and `plan_export`; "
            f"continue planning without that export, or use `{suggested_path}` for the plan file instead."
        )
    return (
        "Plan exports only support markdown or text plan documents. "
        "Continue planning, and if you still want a plan file, use a `.md`, `.txt`, or `.text` path."
    )


def _auto_update_active_plan_step(harness: Any, *, status: str, note: str = "") -> None:
    plan = getattr(harness.state, "active_plan", None) or getattr(harness.state, "draft_plan", None)
    if plan is None:
        return
    active_step = plan.active_step()
    if active_step is None:
        return
    active_step.status = status
    if note.strip():
        active_step.notes.append(note.strip())
    plan.touch()
    harness.state.sync_plan_mirror()
    harness.state.touch()
