from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..task_targets import extract_task_target_paths, task_target_paths_from_harness
from .state import PendingToolCall

_WRITE_TOOLS = {"file_write", "file_append"}
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2, "certain": 3}
_RAW_TEXT_TARGET_SUFFIXES = (".md", ".txt", ".text")


@dataclass
class RecoveredWriteIntent:
    tool_name: str = "file_write"
    path: str = ""
    content: str = ""
    write_session_id: str = ""
    section_name: str = ""
    next_section_name: str = ""
    replace_strategy: str = ""
    confidence: str = "low"
    evidence: list[str] = field(default_factory=list)
    source: str = ""


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
    inline_content, inline_evidence = _extract_inline_tool_content(text, target_path=target_path)
    if inline_content:
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

    if allow_raw_text_targets and target_path.endswith(_RAW_TEXT_TARGET_SUFFIXES):
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

    args = normalize_write_argument_aliases(dict(getattr(pending, "args", {}) or {})) if pending is not None else {}
    intent = RecoveredWriteIntent()
    if pending is not None:
        intent.tool_name = "file_write" if pending.tool_name in _WRITE_TOOLS else pending.tool_name

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
    intent.replace_strategy = str(args.get("replace_strategy") or "").strip()

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


def _contains_write_tool_calls(partial_tool_calls: list[dict[str, Any]] | None) -> bool:
    if not isinstance(partial_tool_calls, list):
        return False
    for item in partial_tool_calls:
        candidate = PendingToolCall.from_payload(item)
        if candidate is not None and candidate.tool_name in _WRITE_TOOLS:
            return True
    return False


def _active_open_write_session(harness: Any) -> Any | None:
    session = getattr(getattr(harness, "state", None), "write_session", None)
    if session is None:
        return None
    if str(getattr(session, "status", "")).strip().lower() == "complete":
        return None
    return session


def _same_path(a: str, b: str, cwd: str | None) -> bool:
    if str(a or "").strip() == str(b or "").strip():
        return True
    try:
        from ..tools.fs import _same_target_path

        return _same_target_path(a, b, cwd)
    except Exception:
        return False


def _attach_session_metadata(intent: RecoveredWriteIntent, *, harness: Any) -> None:
    session = _active_open_write_session(harness)
    if session is None:
        return
    session_target = str(getattr(session, "write_target_path", "") or "").strip()
    if intent.path and session_target:
        if not _same_path(intent.path, session_target, getattr(getattr(harness, "state", None), "cwd", None)):
            return
    if not intent.path and session_target:
        intent.path = session_target
        intent.evidence.append("active_write_session")
    if not intent.write_session_id:
        intent.write_session_id = str(getattr(session, "write_session_id", "") or "").strip()
        if intent.write_session_id:
            intent.evidence.append("active_write_session_id")
    if not intent.section_name:
        section_name = str(
            getattr(session, "write_next_section", "") or getattr(session, "write_current_section", "") or "imports"
        ).strip() or "imports"
        intent.section_name = section_name
        intent.evidence.append("active_section_name")
    if not intent.next_section_name:
        sections = [
            str(item).strip()
            for item in (getattr(session, "suggested_sections", []) or [])
            if str(item).strip()
        ]
        if intent.section_name and sections:
            for idx, section in enumerate(sections):
                if section != intent.section_name:
                    continue
                if idx + 1 < len(sections):
                    intent.next_section_name = sections[idx + 1]
                    intent.evidence.append("active_next_section_name")
                break


def _min_confidence(path_confidence: str, content_confidence: str) -> str:
    path_rank = _CONFIDENCE_RANK.get(path_confidence, 0)
    content_rank = _CONFIDENCE_RANK.get(content_confidence, 0)
    min_rank = min(path_rank, content_rank)
    for label, rank in _CONFIDENCE_RANK.items():
        if rank == min_rank:
            return label
    return "low"


def _ordered_unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = str(value or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _extract_first_fenced_block(text: str) -> str:
    cleaned = str(text or "")
    if not cleaned.strip():
        return ""
    match = re.search(r"```[^\n`]*\n?(.*?)```", cleaned, re.DOTALL)
    if match is None:
        return ""
    return str(match.group(1) or "").strip()


def _extract_unclosed_final_fenced_block(
    text: str,
    *,
    target_path: str,
    path_confidence: str,
) -> str:
    cleaned = str(text or "")
    normalized_target = str(target_path or "").strip().lower()
    if not cleaned.strip():
        return ""
    if path_confidence not in {"high", "certain"}:
        return ""
    if not normalized_target or normalized_target.endswith(_RAW_TEXT_TARGET_SUFFIXES):
        return ""

    fence_matches = list(re.finditer(r"```[^\n`]*\n?", cleaned))
    if not fence_matches:
        return ""

    last_fence = fence_matches[-1]
    recovered = cleaned[last_fence.end():].strip()
    if not recovered:
        return ""
    if "```" in recovered:
        return ""
    if "<tool_call>" in recovered or "<function=" in recovered:
        return ""
    return recovered


def _extract_inline_tool_paths(text: str) -> list[str]:
    paths: list[str] = []
    for pending in _extract_inline_write_calls(text):
        payload = normalize_write_argument_aliases(dict(getattr(pending, "args", {}) or {}))
        path = str(payload.get("path") or "").strip()
        if path:
            paths.append(path)
    return paths


def _extract_inline_tool_content(text: str, *, target_path: str) -> tuple[str, list[str]]:
    for pending in _extract_inline_write_calls(text):
        payload = normalize_write_argument_aliases(dict(getattr(pending, "args", {}) or {}))
        candidate_path = str(payload.get("path") or "").strip()
        if target_path and candidate_path and candidate_path != target_path:
            continue
        content = str(payload.get("content") or "")
        if content.strip():
            evidence: list[str] = []
            block_source = str(getattr(pending, "_recovery_block_source", "") or "").strip()
            if block_source:
                evidence.append(block_source)
            block_format = str(getattr(pending, "_recovery_block_format", "") or "").strip()
            if block_format:
                evidence.append(block_format)
            return content, evidence
    return "", []


def _has_conflicting_paths(*, harness: Any, assistant_text: str, chosen_path: str) -> bool:
    candidates = _ordered_unique(
        task_target_paths_from_harness(harness) + _extract_inline_tool_paths(str(assistant_text or ""))
    )
    if not candidates:
        return False
    for candidate in candidates:
        if candidate == chosen_path:
            continue
        if _same_path(candidate, chosen_path, getattr(getattr(harness, "state", None), "cwd", None)):
            continue
        return True
    return False


def _extract_inline_write_calls(text: str) -> list[PendingToolCall]:
    cleaned = str(text or "")
    if not cleaned.strip():
        return []

    results: list[PendingToolCall] = []
    seen: set[tuple[str, str]] = set()

    def _remember(candidate: PendingToolCall | None, *, block_source: str = "", block_format: str = "") -> None:
        if candidate is None or candidate.tool_name not in _WRITE_TOOLS:
            return
        candidate.args = normalize_write_argument_aliases(dict(getattr(candidate, "args", {}) or {}))
        signature = (
            candidate.tool_name,
            json.dumps(candidate.args, ensure_ascii=True, sort_keys=True),
        )
        if signature in seen:
            return
        seen.add(signature)
        if block_source:
            setattr(candidate, "_recovery_block_source", block_source)
        if block_format:
            setattr(candidate, "_recovery_block_format", block_format)
        results.append(candidate)

    _remember(_pending_from_embedded_xml_text(cleaned), block_source="assistant_xml_tool_block", block_format="assistant_inline_tool_payload")
    _remember(_pending_from_embedded_json_text(cleaned), block_source="assistant_json_tool_payload", block_format="assistant_inline_tool_payload")

    for pattern in (r"<tool_code>(.*?)</tool_code>", r"<tool_call>(.*?)</tool_call>", r"<call>(.*?)</call>"):
        for match in re.finditer(pattern, cleaned, re.DOTALL):
            block = str(match.group(1) or "").strip()
            _remember(
                _pending_from_embedded_xml_text(block),
                block_source="assistant_xml_tool_block",
                block_format="assistant_inline_tool_payload",
            )
            _remember(
                _pending_from_embedded_json_text(block),
                block_source="assistant_json_tool_payload",
                block_format="assistant_inline_tool_payload",
            )

    for match in re.finditer(r"\[(file_write|file_append)\]\s*(\{.*?\})", cleaned, re.DOTALL):
        _remember(
            PendingToolCall.from_payload(
                {
                    "function": {
                        "name": str(match.group(1) or "").strip(),
                        "arguments": str(match.group(2) or "").strip(),
                    }
                }
            ),
            block_source="assistant_bracket_tool_block",
            block_format="assistant_inline_tool_payload",
        )

    fenced = _extract_first_fenced_block(cleaned)
    if fenced:
        _remember(
            _pending_from_embedded_xml_text(fenced),
            block_source="assistant_xml_tool_block",
            block_format="assistant_fenced_tool_payload",
        )
        _remember(
            _pending_from_embedded_json_text(fenced),
            block_source="assistant_json_tool_payload",
            block_format="assistant_fenced_tool_payload",
        )

    return results


def _pending_from_embedded_json_text(text: str) -> PendingToolCall | None:
    cleaned = str(text or "").strip()
    if not cleaned or not cleaned.startswith("{"):
        return None
    try:
        data = json.loads(cleaned)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    if isinstance(data.get("function"), dict):
        return PendingToolCall.from_payload(data)

    tool_name = str(data.get("name", data.get("tool_name", data.get("tool", data.get("action", ""))))).strip()
    if not tool_name:
        return None
    args = data.get("arguments", data.get("args", data.get("params", data.get("parameters", {}))))
    if isinstance(args, dict):
        arguments = json.dumps(args, ensure_ascii=True, sort_keys=True)
    elif isinstance(args, str):
        arguments = args
    else:
        arguments = "{}"
    return PendingToolCall.from_payload(
        {
            "function": {
                "name": tool_name,
                "arguments": arguments,
            }
        }
    )


def _pending_from_embedded_xml_text(text: str) -> PendingToolCall | None:
    cleaned = str(text or "").strip()
    if not cleaned or "<function" not in cleaned:
        return None

    compact_match = re.search(r"<function=([\w_-]+)>\s*(\{.*\})\s*$", cleaned, re.DOTALL)
    if compact_match:
        return PendingToolCall.from_payload(
            {
                "function": {
                    "name": str(compact_match.group(1) or "").strip(),
                    "arguments": str(compact_match.group(2) or "").strip(),
                }
            }
        )

    for pattern in (
        r"<function=([\w_-]+)>(.*?)</function>",
        r"<function\s+name=['\"]?([\w_-]+)['\"]?\s*>(.*?)</function>",
    ):
        match = re.search(pattern, cleaned, re.DOTALL)
        if match is None:
            continue
        tool_name = str(match.group(1) or "").strip()
        inner = str(match.group(2) or "").strip()
        if not tool_name:
            continue
        if inner.startswith("{"):
            return PendingToolCall.from_payload(
                {
                    "function": {
                        "name": tool_name,
                        "arguments": inner,
                    }
                }
            )

        params: dict[str, Any] = {}
        for param_match in re.finditer(r"<parameter=([\w_-]+)>(.*?)</parameter>", inner, re.DOTALL):
            params[str(param_match.group(1) or "").strip()] = str(param_match.group(2) or "").strip()
        if params:
            return PendingToolCall(
                tool_name=tool_name,
                args=params,
                raw_arguments=json.dumps(params, ensure_ascii=True, sort_keys=True),
            )
    return None
