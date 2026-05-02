from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ..guards import is_four_b_or_under_model_name
from ..harness.run_mode import should_enable_complex_write_chat_draft
from ..state import clip_text_value
from ..task_targets import primary_task_target_path
from .recovery_context import build_goal_recap
from .tool_call_parser import _ensure_chunk_write_session
from .model_stream_fallback_support import _format_partial_tool_calls


def _build_tool_specific_recovery_hint(
    *,
    harness: Any,
    partial_tool_calls: list[dict[str, Any]],
) -> str:
    registry = getattr(harness, "registry", None)
    for tool_call in partial_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        tool_name = str(function.get("name") or "").strip()
        if not tool_name:
            continue

        if tool_name in {"file_patch", "ast_patch"}:
            target_path = primary_task_target_path(harness)
            path_hint = f" Target path for this task: `{target_path}`." if target_path else ""
            session = _active_text_write_fallback_session(harness)
            if session is not None and str(getattr(session, "write_session_intent", "") or "").strip().lower() in {
                "replace_file",
                "patch_existing",
            }:
                return (
                    f"For `{tool_name}`, continue Write Session `{session.write_session_id}` for "
                    f"`{session.write_target_path}`. The target path is still the canonical destination; use the "
                    "staged copy for read/verify context. Include required fields `path`, `target_text`, and "
                    f"`replacement_text`, plus `write_session_id='{session.write_session_id}'` when the harness "
                    "supplies an active session. Use exact text including whitespace when you choose `file_patch`. For a narrow repair inside the staged copy, "
                    "prefer `file_patch` for exact text or `ast_patch` for structural edits; for section continuation, use `file_write`. Regenerate the complete "
                    "JSON tool call from scratch."
                )
            return (
                f"For `{tool_name}`, include the required fields `path`, `target_text`, and `replacement_text`."
                f"{path_hint} "
                "Use exact text including whitespace. If the target appears more than once, read the smallest "
                "relevant slice first and regenerate the call with a more specific `target_text`."
            )

        if tool_name in {"file_write", "file_append"}:
            target_path = primary_task_target_path(harness)
            session = _ensure_chunk_write_session(harness, target_path) if target_path else None
            if session is None:
                session = getattr(getattr(harness, "state", None), "write_session", None)
                if session is not None and str(getattr(session, "status", "")).strip().lower() == "complete":
                    session = None
            if session is not None:
                section_name = session.write_next_section or session.write_current_section or "imports"
                return (
                    f"For `{tool_name}`, continue Write Session `{session.write_session_id}` for "
                    f"`{session.write_target_path}`. Include required fields `path` and `content`, plus "
                    f"`write_session_id='{session.write_session_id}'` and `section_name='{section_name}'`. "
                    "If additional chunks remain, include `next_section_name='...'`; omit it on the final chunk. "
                    "When resuming an active write session, prefer `file_write` for chunk continuation. "
                    "If you need a narrow repair inside the staged copy, switch to `file_patch` for exact text or `ast_patch` for structural edits. "
                    "Regenerate the complete JSON tool call from scratch."
            )
            path_hint = f" Target path for this task: `{target_path}`." if target_path else ""
            return (
                f"For `{tool_name}`, include both required fields: `path` and `content`."
                f"{path_hint} "
                "If the full implementation is too large for one write, start with a small valid scaffold "
                "such as imports, a module docstring, entrypoints, or TODO stubs, then extend it in later writes. "
                "If this is a localized edit to an existing file, switch to `file_patch` or `ast_patch` instead of retrying `file_write`. "
                "Empty file writes are currently allowed if the user explicitly asked for that. "
                "If you are making a narrow repair inside a staged copy, use `file_patch` for exact text or `ast_patch` for structural edits instead of `file_write`. "
                "Regenerate the complete JSON tool call from scratch."
            )

        if tool_name == "ssh_file_write":
            return (
                "For `ssh_file_write`, include the required fields `path` and `content`. "
                "If this is a narrow change to an existing remote file, prefer `ssh_file_patch` for exact text "
                "or `ssh_file_replace_between` for bounded multiline regions so the payload stays smaller and the provider is less likely to stall. "
                "Reserve `ssh_file_write` for new remote files or intentional full overwrites."
            )

        if tool_name == "ssh_file_patch":
            return (
                "For `ssh_file_patch`, include the required fields `path`, `target_text`, and `replacement_text`. "
                "Use exact text including whitespace. Prefer this over `ssh_file_write` when only a small remote edit is needed."
            )

        if tool_name == "ssh_file_replace_between":
            return (
                "For `ssh_file_replace_between`, include the required fields `path`, `start_text`, `end_text`, and `replacement_text`. "
                "Prefer this for bounded remote HTML/CSS blocks because it keeps the payload smaller than a full `ssh_file_write`."
            )

        if registry is not None:
            tool_spec = registry.get(tool_name)
            if tool_spec is not None:
                required = tool_spec.schema.get("required", [])
                if required:
                    required_text = ", ".join(str(field) for field in required)
                    return (
                        f"For `{tool_name}`, include the required fields: {required_text}. "
                        "Regenerate the complete JSON tool call from scratch."
                    )

        return f"Regenerate the complete `{tool_name}` JSON tool call from scratch with all required arguments."
    return ""


def _with_speaker(harness: Any, data: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(data or {})
    if getattr(harness.state, "planning_mode_enabled", False):
        payload.setdefault("speaker", "planner")
    return payload


def _build_incomplete_tool_call_recovery_message(
    *,
    harness: Any,
    assistant_text: str,
    partial_tool_calls: list[dict[str, Any]],
) -> str:
    lines = [
        "The previous assistant response was interrupted before a tool call finished streaming.",
        "Please regenerate the full tool call from scratch with complete JSON arguments.",
    ]
    goal_recap = build_goal_recap(harness)
    if goal_recap:
        lines.append(goal_recap)
    clipped_assistant_text, assistant_was_clipped = clip_text_value(assistant_text.strip(), limit=600)
    if clipped_assistant_text:
        prefix = "Assistant text before interruption"
        if assistant_was_clipped:
            prefix += " [truncated]"
        lines.insert(1, f"{prefix}: {clipped_assistant_text}")
    tool_hint = _build_tool_specific_recovery_hint(harness=harness, partial_tool_calls=partial_tool_calls)
    if tool_hint:
        lines.append(tool_hint)
    if partial_tool_calls:
        lines.append("Partial tool calls observed:")
        for item in _format_partial_tool_calls(partial_tool_calls)[:3]:
            lines.append(f"- {item}")
        if len(partial_tool_calls) > 3:
            lines.append(f"- ... and {len(partial_tool_calls) - 3} more")
    return "\n".join(lines)


def _active_text_write_fallback_session(harness: Any) -> Any | None:
    session = getattr(getattr(harness, "state", None), "write_session", None)
    if session is None:
        return None
    if str(getattr(session, "status", "") or "").strip().lower() == "complete":
        return None
    return session


def _is_sub4b_write_timeout(
    harness: Any,
    *,
    error_text: str,
    error_details: dict[str, Any] | None = None,
) -> bool:
    model_name = getattr(getattr(harness, "client", None), "model", None)
    if not is_four_b_or_under_model_name(model_name):
        return False
    if not should_enable_complex_write_chat_draft(
        getattr(getattr(harness, "state", None), "run_brief", SimpleNamespace(original_task="")).original_task,
        model_name=model_name,
        cwd=getattr(getattr(harness, "state", None), "cwd", None),
    ):
        return False
    text = " ".join(
        str(part)
        for part in (
            error_text,
            getattr(error_details, "get", lambda *_: None)("message") if isinstance(error_details, dict) else None,
            getattr(error_details, "get", lambda *_: None)("error") if isinstance(error_details, dict) else None,
            getattr(error_details, "get", lambda *_: None)("detail") if isinstance(error_details, dict) else None,
        )
        if part
    ).lower()
    return any(token in text for token in ("timeout", "timed out", "deadline"))


def _fallback_section_name(session: Any) -> str:
    return str(
        getattr(session, "write_current_section", "")
        or getattr(session, "write_next_section", "")
        or "imports"
    ).strip()


def _fallback_next_section_name(session: Any, current_section: str) -> str:
    next_section = str(getattr(session, "write_next_section", "") or "").strip()
    if next_section and next_section != current_section:
        return next_section
    remaining = [str(section).strip() for section in getattr(session, "suggested_sections", []) or [] if str(section).strip()]
    if current_section in remaining:
        index = remaining.index(current_section)
        if index + 1 < len(remaining):
            return remaining[index + 1]
    return ""


def _build_text_write_fallback_prompt(
    *,
    session: Any,
    current_section: str,
    remaining_sections: list[str],
    task_text: str,
) -> str:
    target_path = str(getattr(session, "write_target_path", "") or "").strip()
    session_intent = str(getattr(session, "write_session_intent", "") or "").strip()
    has_session_id = bool(str(getattr(session, "write_session_id", "") or "").strip())
    if not has_session_id:
        current_section = current_section or "complete_file"
    lines = [
        "You are continuing a code-writing task after a tool-call stream stalled.",
        f"Target path: `{target_path}`." if target_path else "Target path: (unknown).",
        f"Write session id: `{getattr(session, 'write_session_id', '')}`." if has_session_id else "Write session id: (none).",
        f"Current section: `{current_section or 'imports'}`.",
        f"Session intent: `{session_intent or 'general_write'}`.",
        "",
        "User task:",
        task_text.strip() or "(empty)",
        "",
        (
            "Return the complete updated file content for the target path, with no prose."
            if not has_session_id
            else "Return only the code needed for the current section."
        ),
        "Use fenced code blocks when appropriate.",
    ]
    if remaining_sections:
        lines.extend(["", "Suggested remaining sections:"])
        lines.extend(f"- {section}" for section in remaining_sections)
    return "\n".join(lines)
