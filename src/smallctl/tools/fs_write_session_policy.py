from __future__ import annotations

from typing import Any

from ..state import LoopState
from .common import fail
from .fs_write_sessions import _ensure_write_session_files, _resolve
from .fs_sessions import _same_target_path


def _looks_like_system_repair_cycle_id(value: str | None) -> bool:
    return str(value or "").strip().lower().startswith("repair-")


def _suspicious_temp_root_details(path: str) -> dict[str, str] | None:
    raw = str(path or "").strip()
    normalized = raw.replace("\\", "/")
    if not raw or (normalized != "/temp" and not normalized.startswith("/temp/")):
        return None

    suffix = normalized[len("/temp"):]
    relative_suffix = suffix.lstrip("/")
    return {
        "path": raw,
        "suggested_tmp_path": f"/tmp{suffix}",
        "suggested_relative_path": "./temp" if not relative_suffix else f"./temp/{relative_suffix}",
    }


def _guard_suspicious_temp_root_path(path: str) -> dict[str, Any] | None:
    details = _suspicious_temp_root_details(path)
    if details is None:
        return None
    return fail(
        f"Path `{details['path']}` points at the root-level `/temp` directory, which is usually a typo in this harness. "
        f"Use `{details['suggested_tmp_path']}` for a system temp file or `{details['suggested_relative_path']}` for a workspace-local temp path instead.",
        metadata={
            "path": details["path"],
            "error_kind": "suspicious_temp_root_path",
            "suggested_tmp_path": details["suggested_tmp_path"],
            "suggested_relative_path": details["suggested_relative_path"],
        },
    )


def _write_session_resume_metadata(session: Any, *, path: str) -> dict[str, Any]:
    section_name = str(
        getattr(session, "write_next_section", "")
        or getattr(session, "write_current_section", "")
        or "imports"
    ).strip() or "imports"
    return {
        "tool_name": "file_write",
        "required_fields": ["path", "content", "write_session_id", "section_name"],
        "required_arguments": {
            "path": str(getattr(session, "write_target_path", "") or path or "").strip(),
            "write_session_id": str(getattr(session, "write_session_id", "") or "").strip(),
            "section_name": section_name,
        },
        "optional_fields": ["next_section_name"],
    }


def _write_session_staging_mutation_failure(
    *,
    tool_name: str,
    path: str,
    session: Any,
    cwd: str | None = None,
    section_name: str | None = None,
) -> dict[str, Any]:
    target_path = str(getattr(session, "write_target_path", "") or "").strip()
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    write_session_id = str(getattr(session, "write_session_id", "") or "").strip()

    if tool_name in {"file_write", "file_append"}:
        recommended_section = (
            str(section_name or "").strip()
            or str(getattr(session, "write_next_section", "") or getattr(session, "write_current_section", "") or "imports").strip()
            or "imports"
        )
        next_required_tool = {
            "tool_name": tool_name,
            "required_fields": ["path", "content", "write_session_id", "section_name"],
            "required_arguments": {
                "path": target_path,
                "write_session_id": write_session_id,
                "section_name": recommended_section,
            },
            "optional_fields": ["next_section_name"],
            "notes": [
                "Reuse the same content payload; only correct the path back to the target file.",
                "The staging path is for read/verify only and must not be used as the write target.",
            ],
        }
    elif tool_name == "file_patch":
        next_required_tool = {
            "tool_name": "file_patch",
            "required_fields": ["path", "target_text", "replacement_text"],
            "required_arguments": {
                "path": target_path,
                "write_session_id": write_session_id,
            },
            "optional_fields": ["expected_occurrences"],
            "notes": [
                "Reuse the same patch arguments; only correct the path back to the target file.",
                "The staging path is for read/verify only and must not be patched directly.",
            ],
        }
    elif tool_name == "ast_patch":
        next_required_tool = {
            "tool_name": "ast_patch",
            "required_fields": ["path", "language", "operation", "target"],
            "required_arguments": {
                "path": target_path,
                "write_session_id": write_session_id,
            },
            "optional_fields": ["payload", "dry_run", "expected_followup_verifier"],
            "notes": [
                "Reuse the same structural patch arguments; only correct the path back to the target file.",
                "The staging path is for read/verify only and must not be structurally patched directly.",
            ],
        }
    elif tool_name == "file_delete":
        next_required_tool = {
            "tool_name": "file_delete",
            "required_fields": ["path"],
            "required_arguments": {
                "path": target_path,
            },
            "optional_fields": [],
            "notes": [
                "Delete the target path only if you still intend to remove the generated file.",
                "The staging path is for read/verify only and must not be deleted directly.",
            ],
        }
    else:
        next_required_tool = {
            "tool_name": tool_name,
            "required_fields": ["path"],
            "required_arguments": {
                "path": target_path,
            },
            "optional_fields": [],
            "notes": [
                "Retry the same tool with the target path instead of the staging path.",
            ],
        }

    return fail(
        f"Write Session `{write_session_id}` targets `{target_path}`. "
        f"You passed the active staged copy `{path}` to `{tool_name}`, but the staging path "
        f"`{staging_path}` is for read/verify only. Retry the same `{tool_name}` call with "
        f"`path='{target_path}'` instead.",
        metadata={
            "path": str(_resolve(target_path or path, cwd)),
            "requested_path": path,
            "error_kind": "write_session_staging_path_used_as_target",
            "tool_name": tool_name,
            "target_path": target_path,
            "staging_path": staging_path,
            "write_session_id": write_session_id,
            "staged_only": True,
            "next_required_tool": next_required_tool,
        },
    )


def _guard_write_session_staging_mutation(
    *,
    tool_name: str,
    path: str,
    state: LoopState | None,
    cwd: str | None = None,
    session: Any | None = None,
    write_session_id: str | None = None,
    encoding: str = "utf-8",
    section_name: str | None = None,
) -> dict[str, Any] | None:
    active_session = session
    if active_session is None and state is not None:
        candidate = getattr(state, "write_session", None)
        if candidate is not None and str(getattr(candidate, "status", "") or "").strip().lower() != "complete":
            active_session = candidate

    if active_session is None:
        return None

    active_session_id = str(getattr(active_session, "write_session_id", "") or "").strip()
    if write_session_id and active_session_id and write_session_id != active_session_id:
        return None

    target_path = str(getattr(active_session, "write_target_path", "") or "").strip()
    if not target_path:
        return None

    try:
        session_target = _resolve(target_path, cwd)
    except Exception:
        return None

    staging_path = str(getattr(active_session, "write_staging_path", "") or "").strip()
    if not staging_path:
        try:
            staging = _ensure_write_session_files(
                active_session,
                session_target,
                cwd=cwd,
                encoding=encoding,
            )
        except Exception:
            return None
        staging_path = str(staging)

    if not staging_path:
        return None
    if not _same_target_path(staging_path, path, cwd):
        return None

    return _write_session_staging_mutation_failure(
        tool_name=tool_name,
        path=path,
        session=active_session,
        cwd=cwd,
        section_name=section_name,
    )
