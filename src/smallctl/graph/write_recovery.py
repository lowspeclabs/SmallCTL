from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..task_targets import extract_task_target_paths, task_target_paths_from_harness
from .state import PendingToolCall
from .write_recovery_parsing import (
    _active_open_write_session,
    _attach_session_metadata,
    _contains_write_tool_calls,
    _extract_first_fenced_block,
    _extract_inline_tool_paths,
    _extract_inline_tool_payload,
    _extract_partial_write_payload,
    _extract_unclosed_final_fenced_block,
    _has_conflicting_paths,
    _min_confidence,
    _ordered_unique,
    _same_path,
    maybe_finalize_recovered_assistant_write,
    recovered_content_looks_like_complete_file,
    _force_finalize_if_complete_file,
)

_WRITE_TOOLS = {"file_write", "file_append"}
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2, "certain": 3}


@dataclass
class RecoveredWriteIntent:
    tool_name: str = "file_write"
    path: str = ""
    content: str = ""
    write_session_id: str = ""
    section_name: str = ""
    next_section_name: str = ""
    next_section_name_supplied: bool = False
    next_section_name_origin: str = ""
    replace_strategy: str = ""
    confidence: str = "low"
    evidence: list[str] = field(default_factory=list)
    source: str = ""
    _is_append: bool = False


def normalize_write_argument_aliases(args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    normalized = dict(args)

    target_path = normalized.pop("target_path", None)
    if (not str(normalized.get("path") or "").strip()) and str(target_path or "").strip():
        normalized["path"] = target_path

    next_section = normalized.pop("next_section", None)
    if (not str(normalized.get("next_section_name") or "").strip()) and str(next_section or "").strip():
        normalized["next_section_name"] = next_section

    session_id = normalized.pop("session_id", None)
    if (not str(normalized.get("write_session_id") or "").strip()) and str(session_id or "").strip():
        normalized["write_session_id"] = session_id

    return normalized


def infer_write_target_path(
    *,
    harness: Any,
    pending: PendingToolCall | None = None,
    assistant_text: str = "",
    partial_tool_calls: list[dict[str, Any]] | None = None,
) -> tuple[str, str, list[str]]:
    explicit_path = ""
    if pending is not None:
        normalized_args = normalize_write_argument_aliases(dict(getattr(pending, "args", {}) or {}))
        explicit_path = str(normalized_args.get("path") or "").strip()
    if explicit_path:
        return explicit_path, "certain", ["explicit_tool_path"]

    session = _active_open_write_session(harness)
    if session is not None:
        session_target = str(getattr(session, "write_target_path", "") or "").strip()
        if session_target:
            return session_target, "high", ["active_write_session"]

    task_targets = _ordered_unique(task_target_paths_from_harness(harness))
    if len(task_targets) == 1:
        return task_targets[0], "high", ["task_target_path"]
    if len(task_targets) > 1:
        return "", "low", ["conflicting_task_targets"]

    inline_targets = _ordered_unique(_extract_inline_tool_paths(str(assistant_text or "")))
    if len(inline_targets) == 1:
        return inline_targets[0], "high", ["inline_tool_path"]

    assistant_targets = _ordered_unique(extract_task_target_paths(str(assistant_text or "")))
    if len(assistant_targets) == 1:
        return assistant_targets[0], "medium", ["assistant_target_path"]
    return "", "low", []


def recover_content_from_partial_calls(
    partial_tool_calls: list[dict[str, Any]] | None,
    *,
    target_path: str = "",
    cwd: str | None = None,
) -> str:
    if not isinstance(partial_tool_calls, list):
        return ""

    for raw in reversed(partial_tool_calls):
        candidate = PendingToolCall.from_payload(raw)
        if candidate is None or candidate.tool_name not in _WRITE_TOOLS:
            continue
        candidate_args = normalize_write_argument_aliases(candidate.args)
        content = str(candidate_args.get("content") or "")
        if not content.strip():
            continue
        candidate_path = str(candidate_args.get("path") or "").strip()
        if target_path and candidate_path and not _same_path(candidate_path, target_path, cwd):
            continue
        return content
    return ""


def recover_content_from_assistant_text(
    text: str,
    *,
    target_path: str,
    allow_raw_text_targets: bool,
    path_confidence: str = "low",
) -> tuple[str, str, list[str]]:
    inline_payload, inline_evidence = _extract_inline_tool_payload(text, target_path=target_path)
    inline_content = str(inline_payload.get("content") or "")
    if inline_content.strip():
        evidence = ["assistant_inline_tool_block"]
        evidence.extend(inline_evidence)
        return inline_content, "high", evidence

    fenced = _extract_first_fenced_block(text)
    if fenced:
        return fenced, "high", ["assistant_fenced_code"]

    unclosed_fence = _extract_unclosed_final_fenced_block(
        text,
        target_path=target_path,
        path_confidence=path_confidence,
    )
    if unclosed_fence:
        return unclosed_fence, "high", ["assistant_fenced_code", "assistant_unclosed_fenced_code"]

    if allow_raw_text_targets and str(target_path or "").endswith((".md", ".txt", ".text")):
        prose = str(text or "").strip()
        if prose and "<tool_call>" not in prose and "<function=" not in prose:
            return prose, "medium", ["assistant_raw_text"]
    return "", "low", []


def recover_write_intent(
    *,
    harness: Any,
    pending: PendingToolCall | None,
    assistant_text: str = "",
    partial_tool_calls: list[dict[str, Any]] | None = None,
) -> RecoveredWriteIntent | None:
    config = getattr(harness, "config", None)
    if config is not None and not bool(getattr(config, "enable_write_intent_recovery", True)):
        return None

    if pending is not None and pending.tool_name not in _WRITE_TOOLS:
        return None
    if pending is None and not _contains_write_tool_calls(partial_tool_calls):
        return None

    raw_args = dict(getattr(pending, "args", {}) or {}) if pending is not None else {}
    args = normalize_write_argument_aliases(raw_args) if pending is not None else {}
    intent = RecoveredWriteIntent()
    if pending is not None:
        intent._is_append = pending.tool_name == "file_append"
    elif partial_tool_calls:
        for partial in partial_tool_calls:
            if str(partial.get("function", {}).get("name", "")) == "file_append":
                intent._is_append = True
                break

    if pending is not None:
        intent.tool_name = "file_write" if pending.tool_name in _WRITE_TOOLS else pending.tool_name
    elif intent._is_append:
        intent.tool_name = "file_append"
    else:
        intent.tool_name = "file_write"

    path, path_confidence, path_evidence = infer_write_target_path(
        harness=harness,
        pending=pending,
        assistant_text=assistant_text,
        partial_tool_calls=partial_tool_calls,
    )
    intent.path = path
    intent.evidence.extend(path_evidence)

    explicit_content = str(args.get("content") or "")
    content_confidence = "low"
    if explicit_content.strip():
        intent.content = explicit_content
        intent.source = "tool_args_content"
        content_confidence = "certain"
        intent.evidence.append("explicit_tool_content")
    else:
        partial_content = recover_content_from_partial_calls(
            partial_tool_calls,
            target_path=path,
            cwd=getattr(getattr(harness, "state", None), "cwd", None),
        )
        if partial_content.strip():
            intent.content = partial_content
            intent.source = "partial_tool_arguments"
            content_confidence = "high"
            intent.evidence.append("partial_tool_content")
        else:
            allow_raw_text_targets = bool(
                getattr(config, "write_recovery_allow_raw_text_targets", True) if config is not None else True
            )
            text_content, recovered_conf, recovered_evidence = recover_content_from_assistant_text(
                assistant_text,
                target_path=path,
                allow_raw_text_targets=allow_raw_text_targets,
                path_confidence=path_confidence,
            )
            if text_content.strip():
                intent.content = text_content
                intent.source = "assistant_text"
                content_confidence = recovered_conf
                intent.evidence.extend(recovered_evidence)

    intent.write_session_id = str(args.get("write_session_id") or "").strip()
    intent.section_name = str(args.get("section_name") or args.get("section_id") or "").strip()
    intent.next_section_name = str(args.get("next_section_name") or "").strip()
    intent.next_section_name_supplied = any(key in raw_args for key in ("next_section_name", "next_section"))
    if intent.next_section_name:
        intent.next_section_name_origin = "tool_args"
    intent.replace_strategy = str(args.get("replace_strategy") or "").strip()

    assistant_payload, assistant_payload_evidence = _extract_inline_tool_payload(
        assistant_text,
        target_path=path,
    )
    if assistant_payload:
        intent.evidence.extend(["assistant_inline_tool_block", *assistant_payload_evidence])
        if not intent.write_session_id:
            intent.write_session_id = str(assistant_payload.get("write_session_id") or "").strip()
        if not intent.section_name:
            intent.section_name = str(
                assistant_payload.get("section_name")
                or assistant_payload.get("section_id")
                or ""
            ).strip()
        if not intent.next_section_name:
            intent.next_section_name = str(assistant_payload.get("next_section_name") or "").strip()
            if intent.next_section_name:
                intent.next_section_name_supplied = True
                intent.next_section_name_origin = "assistant_payload"
        if not intent.replace_strategy:
            intent.replace_strategy = str(assistant_payload.get("replace_strategy") or "").strip()

    partial_payload, partial_payload_evidence = _extract_partial_write_payload(
        partial_tool_calls,
        target_path=path,
        cwd=getattr(getattr(harness, "state", None), "cwd", None),
    )
    if partial_payload:
        intent.evidence.extend(partial_payload_evidence)
        if not intent.write_session_id:
            intent.write_session_id = str(partial_payload.get("write_session_id") or "").strip()
        if not intent.section_name:
            intent.section_name = str(
                partial_payload.get("section_name")
                or partial_payload.get("section_id")
                or ""
            ).strip()
        if not intent.next_section_name and not intent.next_section_name_supplied:
            intent.next_section_name = str(partial_payload.get("next_section_name") or "").strip()
            if intent.next_section_name:
                intent.next_section_name_supplied = True
                intent.next_section_name_origin = "partial_tool_calls"
        if not intent.replace_strategy:
            intent.replace_strategy = str(partial_payload.get("replace_strategy") or "").strip()

    _attach_session_metadata(intent, harness=harness)
    if intent.path:
        conflicts = _has_conflicting_paths(harness=harness, assistant_text=assistant_text, chosen_path=intent.path)
        if conflicts:
            intent.confidence = "low"
            intent.evidence.append("conflicting_target_paths")
            return intent

    intent.confidence = _min_confidence(path_confidence, content_confidence)
    return intent


def can_safely_synthesize(intent: RecoveredWriteIntent, *, harness: Any) -> bool:
    if not intent.path or not intent.content:
        return False
    config = getattr(harness, "config", None)
    min_confidence = str(getattr(config, "write_recovery_min_confidence", "high") if config is not None else "high")
    min_rank = _CONFIDENCE_RANK.get(min_confidence, _CONFIDENCE_RANK["high"])
    confidence_rank = _CONFIDENCE_RANK.get(intent.confidence, _CONFIDENCE_RANK["low"])
    if confidence_rank < min_rank:
        return False
    if (intent.source == "assistant_text") and ("assistant_raw_text" in intent.evidence):
        allow_raw = bool(getattr(config, "write_recovery_allow_raw_text_targets", True) if config is not None else True)
        if not allow_raw:
            return False
    if (intent.source == "assistant_text") and any(
        tag in intent.evidence for tag in ("assistant_fenced_code", "assistant_inline_tool_block", "assistant_raw_text")
    ):
        if config is not None and not bool(getattr(config, "enable_assistant_code_write_recovery", True)):
            return False
    return True


def build_synthetic_write_args(intent: RecoveredWriteIntent) -> dict[str, Any]:
    args: dict[str, Any] = {
        "path": intent.path,
        "content": intent.content,
    }
    if intent.write_session_id:
        args["write_session_id"] = intent.write_session_id
    if intent.section_name:
        args["section_name"] = intent.section_name
    if intent.next_section_name:
        args["next_section_name"] = intent.next_section_name
    if intent.replace_strategy:
        args["replace_strategy"] = intent.replace_strategy
    return args


def _maybe_prepend_existing_content(intent: RecoveredWriteIntent, *, harness: Any) -> None:
    if not intent._is_append or not intent.path or not intent.content:
        return

    cwd = getattr(getattr(harness, "state", None), "cwd", None)
    path = Path(intent.path)
    if not path.is_absolute():
        base = Path(cwd) if isinstance(cwd, str) and cwd else Path.cwd()
        try:
            path = (base / path).resolve()
        except Exception:
            path = base / path

    if path.is_file():
        try:
            existing = path.read_text(encoding="utf-8")
            if existing:
                intent.content = existing + intent.content
                harness._runlog(
                    "write_recovery_append_merge",
                    "prepended existing file content for write-recovery append synthesis",
                    path=str(path),
                    existing_chars=len(existing),
                    new_chars=len(intent.content),
                )
        except Exception as exc:
            harness.log.warning("failed to read existing file for write-recovery append merge: %s", exc)


def build_synthetic_file_write_call(intent: RecoveredWriteIntent, *, tool_call_id: str = "") -> dict[str, Any]:
    args = build_synthetic_write_args(intent)
    call_id = tool_call_id or f"write_recovery_{intent.write_session_id or 'no_session'}"
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "file_write",
            "arguments": json.dumps(args, ensure_ascii=True, sort_keys=True),
        },
    }


def write_recovery_metadata(intent: RecoveredWriteIntent, *, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "path": intent.path,
        "confidence": intent.confidence,
        "evidence": list(intent.evidence),
        "recovery_kind": write_recovery_kind(intent),
        "source": intent.source,
        "write_session_id": intent.write_session_id,
        "next_section_name": intent.next_section_name,
        "next_section_name_origin": intent.next_section_name_origin,
        "content_chars": len(intent.content or ""),
    }


def write_recovery_kind(intent: RecoveredWriteIntent) -> str:
    source = str(getattr(intent, "source", "") or "").strip()
    evidence = {
        str(item).strip()
        for item in getattr(intent, "evidence", []) or []
        if str(item).strip()
    }

    if source == "tool_args_content":
        return "tool_args_content"
    if source == "partial_tool_arguments":
        return "partial_tool_arguments"
    if source != "assistant_text":
        return "unknown"

    for tag in (
        "assistant_unclosed_fenced_code",
        "assistant_fenced_code",
        "assistant_xml_tool_block",
        "assistant_json_tool_payload",
        "assistant_bracket_tool_block",
        "assistant_inline_tool_block",
        "assistant_raw_text",
    ):
        if tag in evidence:
            return tag
    return "assistant_text"


def _same_path(a: str, b: str, cwd: str | None) -> bool:
    if str(a or "").strip() == str(b or "").strip():
        return True
    try:
        from ..tools.fs import _same_target_path

        return _same_target_path(a, b, cwd)
    except Exception:
        return False
