from __future__ import annotations

import json
from typing import Any

from ..state import json_safe_value
from ..task_targets import extract_task_target_paths, primary_task_target_path
from .state import PendingToolCall
from .write_recovery import infer_write_target_path, normalize_write_argument_aliases
from .tool_write_session_policy import (
    _active_write_session_for_target,
    _ensure_chunk_write_session,
    _should_enter_chunk_mode,
    _suggested_chunk_sections,
)


def _declared_read_before_write_reason(assistant_text: str) -> dict[str, Any] | None:
    text = str(assistant_text or "").strip().lower()
    if not text:
        return None

    explicit_tool = "file_read(" in text or "artifact_read(" in text
    read_phrases = (
        "let me read",
        "i'll read",
        "i will read",
        "need to read",
        "going to read",
        "read exactly what we have so far",
        "read what we have so far",
        "read the current staged",
        "read the staged copy",
        "read the current file",
        "recover the current staged content",
        "recover the staged copy",
        "inspect the staged content",
        "inspect the staged copy",
        "check what we have so far",
    )
    matched_phrase = next((phrase for phrase in read_phrases if phrase in text), "")
    read_intent = explicit_tool or bool(matched_phrase)
    if not read_intent:
        return None

    context_hints = (
        "what we have so far",
        "current staged",
        "staged copy",
        "staged content",
        "current file",
        "current content",
        "exactly what we have",
        "recover",
        "read first",
    )
    matched_hint = next((hint for hint in context_hints if hint in text), "")
    if not explicit_tool and not matched_hint:
        return None

    excerpt = str(assistant_text or "").strip()
    if len(excerpt) > 220:
        excerpt = excerpt[:217].rstrip() + "..."
    return {
        "reason_kind": "declared_read_before_write",
        "explicit_tool": explicit_tool,
        "matched_phrase": matched_phrase,
        "matched_hint": matched_hint,
        "assistant_excerpt": excerpt,
    }


def _assistant_declares_read_before_write(assistant_text: str) -> bool:
    return _declared_read_before_write_reason(assistant_text) is not None


def _recover_declared_read_before_write(
    harness: Any,
    pending: PendingToolCall,
    *,
    assistant_text: str = "",
) -> tuple[PendingToolCall, dict[str, Any]] | None:
    if pending.tool_name not in {"file_write", "file_append", "file_patch", "ast_patch"}:
        return None
    reason = _declared_read_before_write_reason(assistant_text)
    if reason is None:
        return None

    target_path = str(pending.args.get("path") or primary_task_target_path(harness) or "").strip()
    session = _active_write_session_for_target(harness, target_path)
    if session is None:
        session = getattr(getattr(harness, "state", None), "write_session", None)
        if session is not None and str(getattr(session, "status", "")).strip().lower() == "complete":
            session = None
    if session is None:
        return None

    target_path = str(target_path or getattr(session, "write_target_path", "") or "").strip()
    if not target_path:
        return None

    args = {"path": target_path}
    return (
        PendingToolCall(
            tool_name="file_read",
            args=args,
            tool_call_id=pending.tool_call_id,
            raw_arguments=json.dumps(args, ensure_ascii=True, sort_keys=True),
            source="system",
        ),
        reason,
    )


def _assistant_text_target_paths(harness: Any, assistant_text: str = "") -> list[str]:
    candidates: list[str] = []
    if assistant_text.strip():
        candidates.append(assistant_text)

    recent_messages = getattr(getattr(harness, "state", None), "recent_messages", [])
    for message in reversed(recent_messages):
        if getattr(message, "role", "") != "assistant":
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if content:
            candidates.append(content)
        if len(candidates) >= 4:
            break

    ordered: list[str] = []
    seen: set[str] = set()
    for text in candidates:
        for path in extract_task_target_paths(text):
            if path in seen:
                continue
            seen.add(path)
            ordered.append(path)
    return ordered


def _infer_write_tool_path(harness: Any, pending: PendingToolCall, *, assistant_text: str = "") -> str:
    if pending.tool_name not in {"file_write", "file_append"}:
        return ""

    explicit_path = str(pending.args.get("path") or "").strip()
    if explicit_path:
        return explicit_path

    session = getattr(getattr(harness, "state", None), "write_session", None)
    if session is not None and str(getattr(session, "status", "")).strip().lower() != "complete":
        session_target = str(getattr(session, "write_target_path", "") or "").strip()
        if session_target:
            return session_target

    task_target = str(primary_task_target_path(harness) or "").strip()
    if task_target:
        return task_target

    assistant_paths = _assistant_text_target_paths(harness, assistant_text)
    if assistant_paths:
        return assistant_paths[0]

    return ""


def _repair_active_write_session_args(
    harness: Any,
    pending: PendingToolCall,
    *,
    assistant_text: str = "",
) -> bool:
    if pending.tool_name not in {"file_write", "file_append"}:
        return False

    raw_args = dict(getattr(pending, "args", {}) or {})
    args = normalize_write_argument_aliases(raw_args)
    repaired = args != raw_args

    def _is_blank(value: Any) -> bool:
        return value is None or (isinstance(value, str) and not value.strip())

    session = getattr(getattr(harness, "state", None), "write_session", None)
    if session is not None and str(getattr(session, "status", "")).strip().lower() != "complete":
        session_id = str(args.get("write_session_id") or "").strip()
        inferred_path = str(args.get("path") or "").strip()
        if not inferred_path:
            inferred_path, _confidence, _evidence = infer_write_target_path(
                harness=harness,
                pending=pending,
                assistant_text=assistant_text,
                partial_tool_calls=None,
            )
        session_matches_target = not inferred_path
        if inferred_path:
            session_matches_target = _active_write_session_for_target(harness, inferred_path) is session
        if session_id and session_id == session.write_session_id:
            if _is_blank(args.get("path")) and str(session.write_target_path or "").strip():
                args["path"] = session.write_target_path
                repaired = True
            if _is_blank(args.get("section_name")) and _is_blank(args.get("section_id")):
                section_name = str(
                    session.write_next_section
                    or session.write_current_section
                    or ""
                ).strip()
                if section_name:
                    args["section_name"] = section_name
                    repaired = True
        elif session_matches_target:
            if session_id != session.write_session_id:
                args["write_session_id"] = session.write_session_id
                repaired = True
            if _is_blank(args.get("path")) and str(session.write_target_path or "").strip():
                args["path"] = session.write_target_path
                repaired = True
            if _is_blank(args.get("section_name")) and _is_blank(args.get("section_id")):
                section_name = str(
                    session.write_next_section
                    or session.write_current_section
                    or ""
                ).strip()
                if section_name:
                    args["section_name"] = section_name
                    repaired = True

    if _is_blank(args.get("path")):
        inferred_path, _confidence, _evidence = infer_write_target_path(
            harness=harness,
            pending=pending,
            assistant_text=assistant_text,
            partial_tool_calls=None,
        )
        if inferred_path:
            args["path"] = inferred_path
            repaired = True

    if not repaired:
        return False

    pending.args = args
    pending.raw_arguments = json.dumps(args, ensure_ascii=True, sort_keys=True)
    return True




def _salvage_active_write_session_append(
    harness: Any,
    pending: PendingToolCall,
) -> PendingToolCall | None:
    if pending.tool_name != "file_append":
        return None

    content = pending.args.get("content")
    if content is not None and str(content).strip():
        return None

    active_session = getattr(getattr(harness, "state", None), "write_session", None)
    fallback_target = ""
    if active_session is not None:
        fallback_target = str(getattr(active_session, "write_target_path", "") or "").strip()
    target_path = str(
        pending.args.get("path")
        or primary_task_target_path(harness)
        or fallback_target
    ).strip()
    session = _active_write_session_for_target(harness, target_path) if target_path else None
    if session is None:
        session = getattr(getattr(harness, "state", None), "write_session", None)
        if session is not None and str(getattr(session, "status", "")).strip().lower() == "complete":
            session = None
    if session is None:
        return None

    payload = getattr(getattr(harness, "state", None), "scratchpad", {}).get("_last_incomplete_tool_call")
    if not isinstance(payload, dict):
        return None
    raw_calls = payload.get("partial_tool_calls_raw")
    if not isinstance(raw_calls, list) or not raw_calls:
        return None

    from ..tools.fs import _same_target_path

    for item in reversed(raw_calls):
        candidate = PendingToolCall.from_payload(item)
        if candidate is None or candidate.tool_name not in {"file_write", "file_append", "file_patch", "ast_patch"}:
            continue
        candidate_content = candidate.args.get("content")
        if candidate_content is None or not str(candidate_content).strip():
            continue
        candidate_path = str(candidate.args.get("path") or "").strip()
        if candidate_path and not _same_target_path(session.write_target_path, candidate_path, getattr(harness.state, "cwd", None)):
            continue

        repaired_args: dict[str, Any] = {
            "path": candidate_path or session.write_target_path,
            "content": str(candidate_content),
            "write_session_id": session.write_session_id,
            "section_name": str(
                candidate.args.get("section_name")
                or candidate.args.get("section_id")
                or session.write_next_section
                or session.write_current_section
                or "imports"
            ).strip(),
        }
        for key in (
            "section_id",
            "section_role",
            "next_section_name",
            "replace_strategy",
            "expected_followup_verifier",
        ):
            value = candidate.args.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            repaired_args[key] = value

        return PendingToolCall(
            tool_name="file_write",
            args=repaired_args,
            tool_call_id=pending.tool_call_id,
            raw_arguments=json.dumps(repaired_args, ensure_ascii=True, sort_keys=True),
            source=str(getattr(pending, "source", "model") or "model"),
        )
    return None


def _detect_oversize_write_payload(
    harness: Any,
    pending: PendingToolCall,
) -> tuple[str, dict[str, Any]] | None:
    if pending.tool_name != "file_write":
        return None

    model_name = getattr(getattr(harness, "client", None), "model", None)
    from ..guards import is_small_model_name
    is_small = is_small_model_name(model_name)

    if not is_small:
        return None

    content = str(pending.args.get("content", ""))
    payload_size = len(content)
    write_session = getattr(harness.state, "write_session", None)

    if write_session and str(getattr(write_session, "status", "")).strip().lower() != "complete":
        from ..tools.fs import _resolve
        try:
            target_path = _resolve(str(pending.args.get("path") or ""), getattr(harness.state, "cwd", None))
            session_path = _resolve(write_session.write_target_path, getattr(harness.state, "cwd", None))
        except Exception:
            target_path = None
            session_path = None
        if target_path == session_path and (
            not pending.args.get("write_session_id")
            or pending.args.get("write_session_id") != write_session.write_session_id
        ):
            message = (
                f"A write session `{write_session.write_session_id}` is already active for `{write_session.write_target_path}`. "
                "You must include the correct `write_session_id` and section metadata to continue. "
                "Do not attempt to overwrite the file directly during an active session."
            )
            return message, {
                "tool_name": pending.tool_name,
                "tool_call_id": pending.tool_call_id,
                "reason": "session_context_missing",
                "active_session_id": write_session.write_session_id,
            }

    threshold = _write_policy_value(harness, "small_model_hard_write_chars", 4000)
    if payload_size > threshold and not (write_session and pending.args.get("write_session_id")):
        message = (
            f"Write payload for `{pending.tool_name}` exceeds the hard limit of {threshold} characters ({payload_size} chars). "
            "Please use chunked write mode or break your edit into smaller pieces."
        )
        return message, {
            "tool_name": pending.tool_name,
            "tool_call_id": pending.tool_call_id,
            "size": payload_size,
            "threshold": threshold,
            "reason": "payload_too_large",
        }

    return None


def _build_schema_repair_message(
    harness: Any,
    pending: PendingToolCall,
    required_fields: list[Any],
) -> str:
    field_names = [str(field) for field in required_fields if str(field).strip()]
    required_text = ", ".join(field_names) or "path, content"
    if pending.tool_name in {"file_patch", "ast_patch"}:
        target_path = str(pending.args.get("path") or primary_task_target_path(harness) or "").strip()
        target_hint = f" Target path for this task: `{target_path}`." if target_path else ""
        session = _active_write_session_for_target(harness, target_path)
        if session is None:
            session = getattr(getattr(harness, "state", None), "write_session", None)
            if session is not None and str(getattr(session, "status", "")).strip().lower() == "complete":
                session = None
        structural_hint = (
            "Use structural locator fields like `language`, `operation`, and `target`; add `payload` when the operation needs replacement or inserted code. "
            "If the target appears more than once, narrow the locator before retrying."
            if pending.tool_name == "ast_patch"
            else "Use exact target text and replacement text including whitespace. If the target appears more "
            "than once, read the smallest relevant slice first and make the target text more specific."
        )
        if session is not None:
            return (
                f"Tool call '{pending.tool_name}' was emitted without arguments. "
                f"Continue with Write Session `{session.write_session_id}` for `{session.write_target_path}` if this is the current target. "
                "The active staged copy is the read/verify source. "
                f"Resend `{pending.tool_name}` with these required fields: {required_text}."
                f"{target_hint} "
                f"{structural_hint} The target path remains the canonical destination while the staged copy is the read/verify source."
            )
        return (
            f"Tool call '{pending.tool_name}' was emitted without arguments. "
            f"Please resend the tool call with these required fields: {required_text}."
            f"{target_hint} "
            f"{structural_hint}"
        )
    if pending.tool_name in {"file_write", "file_append"}:
        target_path = str(pending.args.get("path") or primary_task_target_path(harness) or "").strip()
        target_hint = f" Target path for this task: `{target_path}`." if target_path else ""
        session = _active_write_session_for_target(harness, target_path)
        if session is None:
            session = getattr(getattr(harness, "state", None), "write_session", None)
            if session is not None and str(getattr(session, "status", "")).strip().lower() == "complete":
                session = None
        if session is not None:
            section_name = session.write_next_section or session.write_current_section or "imports"
            next_hint = (
                f" Resume with section `{section_name}` and include `next_section_name='...'` if more sections remain."
            )
            if session.write_sections_completed and not session.write_next_section:
                next_hint = " Omit `next_section_name` on the final chunk to finalize the session."
            return (
                f"Tool call '{pending.tool_name}' was emitted without arguments. "
                f"Continue Write Session `{session.write_session_id}` for `{session.write_target_path}`. "
                f"Resend `file_write` with these required fields: {required_text}, plus "
                f"`write_session_id='{session.write_session_id}'` and `section_name='{section_name}'`."
                f"{next_hint} Do not switch away from `file_write` or `file_read` unless you truly need local context for a repair."
                " For a narrow repair inside the staged copy, use `file_patch` for exact text or `ast_patch` for structural edits instead of `file_write`."
            )
        return (
            f"Tool call '{pending.tool_name}' was emitted without arguments. "
            f"Please resend the tool call with these required fields: {required_text}."
            f"{target_hint} "
            "If a full implementation is too large, break it down with a small valid scaffold first, "
            "then extend it with later writes. If this is a localized edit to an existing file, switch to "
            "`file_patch` or `ast_patch` instead of retrying a full `file_write`."
        )
    return (
        f"Tool call '{pending.tool_name}' was emitted without arguments. "
        f"Please resend the tool call with these required fields: {required_text}."
    )
