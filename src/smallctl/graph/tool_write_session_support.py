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
    _write_policy_value,
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
            # Path-based implicit resolution: do NOT inject write_session_id.
            # The tool layer will match the path to the active session.
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
        if target_path == session_path:
            has_section = bool(
                str(pending.args.get("section_name") or "").strip()
                or str(pending.args.get("section_id") or "").strip()
            )
            has_next_section = bool(str(pending.args.get("next_section_name") or "").strip())
            session_next = str(getattr(write_session, "write_next_section", "") or "").strip()
            if not has_section and not has_next_section and not session_next:
                message = (
                    f"A write session is already active for `{write_session.write_target_path}`. "
                    "This large payload needs a `section_name` so the harness can place it in the staged file. "
                    "Use chunked authoring: send one logical section at a time with `path` and `section_name`, "
                    "and omit `next_section_name` on the final section."
                )
                return message, {
                    "tool_name": pending.tool_name,
                    "tool_call_id": pending.tool_call_id,
                    "reason": "session_context_missing",
                    "active_session_path": write_session.write_target_path,
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


def _detect_oversize_patch_payload(
    harness: Any,
    pending: PendingToolCall,
) -> tuple[str, dict[str, Any]] | None:
    """Reject file_patch calls whose target_text + replacement_text exceed a safe limit.

    Large patch payloads make JSON generation unreliable (models drop escapes or
    truncate closing braces), which causes the downstream server to fail when it
    re-parses the tool-call arguments on the next turn.
    """
    if pending.tool_name != "file_patch":
        return None

    target_text = str(pending.args.get("target_text", "") or "")
    replacement_text = str(pending.args.get("replacement_text", "") or "")
    total_size = len(target_text) + len(replacement_text)
    max_chars = _write_policy_value(harness, "patch_hard_chars_limit", 4000)

    if total_size > max_chars:
        message = (
            f"Patch payload for `{pending.tool_name}` exceeds the hard limit of {max_chars} characters "
            f"({total_size} chars). Large patches are unreliable. "
            "Please use `file_write` with chunked authoring or break the edit into smaller `file_patch` calls."
        )
        return message, {
            "tool_name": pending.tool_name,
            "tool_call_id": pending.tool_call_id,
            "size": total_size,
            "threshold": max_chars,
            "reason": "patch_payload_too_large",
        }
    return None


def _build_schema_repair_message(
    harness: Any,
    pending: PendingToolCall,
    required_fields: list[Any],
) -> str:
    field_names = [str(field) for field in required_fields if str(field).strip()]
    required_text = ", ".join(field_names) or "path, content"
    schema_hint = compact_tool_schema_hint(harness, pending.tool_name)

    def _with_schema_hint(message: str) -> str:
        if schema_hint:
            return message.rstrip() + "\n\n" + schema_hint
        return message

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
            return _with_schema_hint(
                f"Tool call '{pending.tool_name}' was emitted without arguments. "
                f"Continue with Write Session `{session.write_session_id}` for `{session.write_target_path}` if this is the current target. "
                "The active staged copy is the read/verify source. "
                f"Resend `{pending.tool_name}` with these required fields: {required_text}."
                f"{target_hint} "
                f"{structural_hint} The target path remains the canonical destination while the staged copy is the read/verify source."
            )
        return _with_schema_hint(
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
            return _with_schema_hint(
                f"Tool call '{pending.tool_name}' was emitted without arguments. "
                f"Continue writing to `{session.write_target_path}`. "
                f"Resend `file_write` with these required fields: {required_text}, plus "
                f"`section_name='{section_name}'`."
                f"{next_hint} Do not switch away from `file_write` or `file_read` unless you truly need local context for a repair."
                " For a narrow repair inside the staged copy, use `file_patch` for exact text or `ast_patch` for structural edits instead of `file_write`."
            )
        return _with_schema_hint(
            f"Tool call '{pending.tool_name}' was emitted without arguments. "
            f"Please resend the tool call with these required fields: {required_text}."
            f"{target_hint} "
            "If a full implementation is too large, break it down with a small valid scaffold first, "
            "then extend it with later writes. If this is a localized edit to an existing file, switch to "
            "`file_patch` or `ast_patch` instead of retrying a full `file_write`."
        )
    return _with_schema_hint(
        f"Tool call '{pending.tool_name}' was emitted without arguments. "
        f"Please resend the tool call with these required fields: {required_text}."
    )


def compact_tool_schema_hint(harness: Any, tool_name: str, *, max_chars: int = 900) -> str:
    schema = _tool_schema_for_hint(harness, tool_name)
    function = schema.get("function") if isinstance(schema, dict) else None
    function = function if isinstance(function, dict) else {}
    parameters = function.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    properties = parameters.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = [str(item) for item in parameters.get("required", []) if str(item).strip()]
    if not properties and not required:
        return ""

    tool = str(function.get("name") or tool_name or "").strip()
    prop_bits: list[str] = []
    for name, spec in properties.items():
        if not str(name).strip():
            continue
        spec = spec if isinstance(spec, dict) else {}
        type_name = spec.get("type")
        if isinstance(type_name, list):
            type_text = "/".join(str(item) for item in type_name if str(item).strip())
        else:
            type_text = str(type_name or "any")
        bit = f"{name}:{type_text}"
        enum_values = spec.get("enum")
        if isinstance(enum_values, list) and 0 < len(enum_values) <= 8:
            enum_text = ", ".join(json.dumps(item) for item in enum_values)
            if len(enum_text) <= 120:
                bit += f" enum=[{enum_text}]"
        prop_bits.append(bit)

    minimal = _minimal_schema_example(required, properties)
    lines = [f"Compact schema for `{tool}`:"]
    if required:
        lines.append("Required fields: " + ", ".join(required))
    if prop_bits:
        lines.append("Allowed properties: " + "; ".join(prop_bits))
    if minimal:
        lines.append("Minimal valid call arguments: " + json.dumps(minimal, sort_keys=True))
    hint = "\n".join(lines)
    if len(hint) > max_chars:
        hint = hint[: max(0, max_chars - 3)].rstrip() + "..."
    _remember_schema_validation_hint(
        harness,
        tool_name=tool,
        required_fields=required,
        schema_excerpt=hint,
    )
    return hint


def _tool_schema_for_hint(harness: Any, tool_name: str) -> dict[str, Any]:
    name = str(tool_name or "").strip()
    registry = getattr(harness, "registry", None)
    get_fn = getattr(registry, "get", None) if registry is not None else None
    spec = get_fn(name) if callable(get_fn) and name else None
    openai_schema = getattr(spec, "openai_schema", None) if spec is not None else None
    if callable(openai_schema):
        try:
            schema = openai_schema()
            if isinstance(schema, dict):
                return schema
        except Exception:
            pass
    export_fn = getattr(registry, "export_openai_tools", None) if registry is not None else None
    if callable(export_fn):
        for kwargs in ({"mode": "loop"}, {}):
            try:
                schemas = export_fn(**kwargs)
            except TypeError:
                continue
            except Exception:
                break
            for entry in schemas or []:
                function = entry.get("function") if isinstance(entry, dict) else None
                if isinstance(function, dict) and str(function.get("name") or "").strip() == name:
                    return entry
    return {}


def _minimal_schema_example(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    example: dict[str, Any] = {}
    for field in required:
        spec = properties.get(field)
        spec = spec if isinstance(spec, dict) else {}
        value = _placeholder_for_schema(spec)
        if value is _NO_PLACEHOLDER:
            return {}
        example[field] = value
    return example


_NO_PLACEHOLDER = object()


def _placeholder_for_schema(spec: dict[str, Any]) -> Any:
    enum_values = spec.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]
    type_name = spec.get("type")
    if isinstance(type_name, list):
        type_name = next((item for item in type_name if item != "null"), type_name[0] if type_name else "")
    if type_name == "string":
        return "..."
    if type_name in {"integer", "number"}:
        return 1
    if type_name == "boolean":
        return False
    if type_name == "array":
        return []
    if type_name == "object":
        return {}
    return "..."


def _remember_schema_validation_hint(
    harness: Any,
    *,
    tool_name: str,
    required_fields: list[str],
    schema_excerpt: str,
) -> None:
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    step = int(getattr(state, "step_count", 0) or 0)
    scratchpad["_last_schema_validation_hint"] = {
        "tool_name": tool_name,
        "required_fields": required_fields,
        "schema_excerpt": schema_excerpt,
        "created_at_step": step,
    }
