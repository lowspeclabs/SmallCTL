from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from ..state import LoopState


def _normalize_section_name(section_name: str | None, section_id: str | None) -> str:
    return str(section_name or section_id or "unnamed").strip() or "unnamed"


def _normalize_replace_strategy(replace_strategy: str | None) -> str:
    strategy = str(replace_strategy or "auto").strip().lower()
    if strategy in {"overwrite", "replace", "rewrite"}:
        return "overwrite"
    if strategy == "append":
        return "append"
    return "auto"


def _write_session_can_finalize(session: Any) -> bool:
    mode = str(getattr(session, "write_session_mode", "") or "").strip().lower()
    return mode in {"chunked_author", "local_repair", "stub_and_fill"}


def _append_unique_section(completed_sections: list[str], section_name: str) -> bool:
    normalized = str(section_name or "").strip()
    if not normalized or normalized in completed_sections:
        return False
    completed_sections.append(normalized)
    return True


def _clone_section_ranges(value: dict[str, dict[str, int]] | None) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        return {}
    cloned: dict[str, dict[str, int]] = {}
    for key, item in value.items():
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
        except (TypeError, ValueError):
            continue
        if start < 0 or end < start:
            continue
        cloned[str(key)] = {"start": start, "end": end}
    return cloned


def new_write_session_id() -> str:
    return f"ws_{uuid.uuid4().hex[:6]}"


def infer_write_session_intent(path: str, cwd: str | None = None) -> str:
    from .fs_write_sessions import _resolve

    try:
        return "patch_existing" if _resolve(path, cwd).exists() else "replace_file"
    except Exception:
        return "replace_file"


def _same_target_path(left: str, right: str, cwd: str | None = None) -> bool:
    from .fs_write_sessions import _resolve

    try:
        return _resolve(left, cwd) == _resolve(right, cwd)
    except Exception:
        return str(left) == str(right)


def _repair_cycle_reads(state: LoopState | None) -> list[str]:
    if state is None:
        return []
    reads = state.scratchpad.setdefault("_repair_cycle_reads", [])
    if not isinstance(reads, list):
        reads = []
        state.scratchpad["_repair_cycle_reads"] = reads
    return [str(item) for item in reads if str(item).strip()]


def _record_repair_cycle_read(state: LoopState | None, path: Path) -> None:
    if state is None or not state.repair_cycle_id:
        return
    reads = _repair_cycle_reads(state)
    normalized = str(path.resolve()) if hasattr(path, "resolve") else str(path)
    if normalized not in reads:
        reads.append(normalized)
        state.scratchpad["_repair_cycle_reads"] = reads


def _repair_cycle_allows_patch(state: LoopState | None, path: Path) -> bool:
    if state is None or not state.repair_cycle_id:
        return True
    reads = set(_repair_cycle_reads(state))
    normalized = str(path.resolve()) if hasattr(path, "resolve") else str(path)
    return normalized in reads


def _mark_repeat_patch(state: LoopState | None) -> None:
    if state is None:
        return
    counters = state.stagnation_counters if isinstance(state.stagnation_counters, dict) else {}
    counters["repeat_patch"] = int(counters.get("repeat_patch", 0)) + 1
    state.stagnation_counters = counters


def _mark_repeat_command(state: LoopState | None) -> None:
    if state is None:
        return
    counters = state.stagnation_counters if isinstance(state.stagnation_counters, dict) else {}
    counters["repeat_command"] = int(counters.get("repeat_command", 0)) + 1
    state.stagnation_counters = counters


def _record_file_change(state: LoopState | None, path: Path) -> None:
    if state is None:
        return
    normalized = str(path.resolve()) if hasattr(path, "resolve") else str(path)
    if normalized in state.files_changed_this_cycle:
        _mark_repeat_patch(state)
    changed = [item for item in state.files_changed_this_cycle if item != normalized]
    changed.append(normalized)
    state.files_changed_this_cycle = changed[-12:]
    try:
        state.invalidate_context(
            reason="file_changed",
            paths=[normalized],
            details={"state_change": f"File changed: {normalized}"},
        )
    except Exception:
        pass
    state.touch()


def _repair_cycle_session_id_failure(
    *,
    supplied_id: str,
    path: str,
    state: LoopState | None,
) -> dict[str, Any]:
    from .fs_write_session_policy import _write_session_resume_metadata
    from .common import fail

    session = getattr(state, "write_session", None) if state is not None else None
    if session is not None and str(getattr(session, "status", "")).strip().lower() == "complete":
        session = None

    metadata: dict[str, Any] = {
        "path": path,
        "error_kind": "repair_cycle_used_as_write_session_id",
        "supplied_write_session_id": supplied_id,
        "system_repair_cycle_id": str(getattr(state, "repair_cycle_id", "") or "").strip(),
    }
    if session is not None:
        metadata["active_write_session_id"] = str(getattr(session, "write_session_id", "") or "").strip()
        metadata["next_required_tool"] = _write_session_resume_metadata(session, path=path)
        return fail(
            f"`{supplied_id}` looks like a system repair cycle ID, not a `write_session_id`. "
            f"Resume the active Write Session with `write_session_id='{session.write_session_id}'` for `{session.write_target_path}`.",
            metadata=metadata,
        )

    return fail(
        f"`{supplied_id}` looks like a system repair cycle ID, not a `write_session_id`. "
        "There is no active write session to resume; omit `write_session_id` for a direct write or inspect `loop_status` for the current blocker.",
        metadata=metadata,
    )
