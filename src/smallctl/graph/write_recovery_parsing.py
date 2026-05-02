from __future__ import annotations

import json
import re
from typing import Any

from ..task_targets import extract_task_target_paths, task_target_paths_from_harness
from .state import PendingToolCall

_WRITE_TOOLS = {"file_write", "file_append"}
_SESSION_ROUTED_TOOLS = {"file_write", "file_append", "file_patch", "ast_patch"}
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2, "certain": 3}
_RAW_TEXT_TARGET_SUFFIXES = (".md", ".txt", ".text")


def _normalize_write_argument_aliases(args: dict[str, Any]) -> dict[str, Any]:
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


def _same_path(a: str, b: str, cwd: str | None) -> bool:
    if str(a or "").strip() == str(b or "").strip():
        return True
    try:
        from ..tools.fs import _same_target_path

        return _same_target_path(a, b, cwd)
    except Exception:
        return False


def _active_open_write_session(harness: Any) -> Any | None:
    session = getattr(getattr(harness, "state", None), "write_session", None)
    if session is None:
        return None
    if str(getattr(session, "status", "")).strip().lower() == "complete":
        return None
    return session


def _contains_write_tool_calls(partial_tool_calls: list[dict[str, Any]] | None) -> bool:
    if not isinstance(partial_tool_calls, list):
        return False
    for item in partial_tool_calls:
        candidate = PendingToolCall.from_payload(item)
        if candidate is not None and candidate.tool_name in _WRITE_TOOLS:
            return True
    return False


def _attach_session_metadata(intent: Any, *, harness: Any) -> None:
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
    active_session_id = str(getattr(session, "write_session_id", "") or "").strip()
    if active_session_id and intent.write_session_id and intent.write_session_id != active_session_id:
        intent.write_session_id = active_session_id
        intent.evidence.append("active_write_session_id_rebound")
    elif not intent.write_session_id:
        intent.write_session_id = active_session_id
        if intent.write_session_id:
            intent.evidence.append("active_write_session_id")
    if not intent.section_name:
        section_name = str(
            getattr(session, "write_next_section", "") or getattr(session, "write_current_section", "") or "imports"
        ).strip() or "imports"
        intent.section_name = section_name
        intent.evidence.append("active_section_name")
    if not intent.next_section_name and not intent.next_section_name_supplied:
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
                    intent.next_section_name_origin = "session_default"
                break


def _extract_inline_tool_paths(text: str) -> list[str]:
    paths: list[str] = []
    for pending in _extract_inline_session_tool_calls(text):
        payload = _normalize_write_argument_aliases(dict(getattr(pending, "args", {}) or {}))
        path = str(payload.get("path") or "").strip()
        if path:
            paths.append(path)
    return paths


def _extract_inline_tool_payload(text: str, *, target_path: str) -> tuple[dict[str, Any], list[str]]:
    for pending in _extract_inline_write_calls(text):
        payload = _normalize_write_argument_aliases(dict(getattr(pending, "args", {}) or {}))
        candidate_path = str(payload.get("path") or "").strip()
        if target_path and candidate_path and candidate_path != target_path:
            continue
        if not any(
            str(payload.get(key) or "").strip()
            for key in ("content", "section_name", "section_id", "next_section_name", "write_session_id", "replace_strategy")
        ):
            continue
        evidence: list[str] = []
        block_source = str(getattr(pending, "_recovery_block_source", "") or "").strip()
        if block_source:
            evidence.append(block_source)
        block_format = str(getattr(pending, "_recovery_block_format", "") or "").strip()
        if block_format:
            evidence.append(block_format)
        return payload, evidence
    return {}, []


def _extract_partial_write_payload(
    partial_tool_calls: list[dict[str, Any]] | None,
    *,
    target_path: str = "",
    cwd: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(partial_tool_calls, list):
        return {}, []

    for raw in reversed(partial_tool_calls):
        candidate = PendingToolCall.from_payload(raw)
        if candidate is None or candidate.tool_name not in _WRITE_TOOLS:
            continue
        payload = _normalize_write_argument_aliases(dict(getattr(candidate, "args", {}) or {}))
        candidate_path = str(payload.get("path") or "").strip()
        if target_path and candidate_path and not _same_path(candidate_path, target_path, cwd):
            continue
        if not any(
            str(payload.get(key) or "").strip()
            for key in ("content", "section_name", "section_id", "next_section_name", "write_session_id", "replace_strategy")
        ):
            continue
        return payload, ["partial_tool_arguments"]
    return {}, []


def _extract_inline_tool_content(text: str, *, target_path: str) -> tuple[str, list[str]]:
    payload, inline_evidence = _extract_inline_tool_payload(text, target_path=target_path)
    content = str(payload.get("content") or "")
    if content.strip():
        evidence = ["assistant_inline_tool_block"]
        evidence.extend(inline_evidence)
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


def _extract_inline_write_calls(text: str) -> list[PendingToolCall]:
    cleaned = str(text or "")
    if not cleaned.strip():
        return []

    results: list[PendingToolCall] = []
    seen: set[tuple[str, str]] = set()

    def _remember(candidate: PendingToolCall | None, *, block_source: str = "", block_format: str = "") -> None:
        if candidate is None or candidate.tool_name not in _WRITE_TOOLS:
            return
        candidate.args = _normalize_write_argument_aliases(dict(getattr(candidate, "args", {}) or {}))
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


def _extract_inline_session_tool_calls(text: str) -> list[PendingToolCall]:
    cleaned = str(text or "")
    if not cleaned.strip():
        return []

    results: list[PendingToolCall] = []
    seen: set[tuple[str, str]] = set()

    def _remember(candidate: PendingToolCall | None) -> None:
        if candidate is None or candidate.tool_name not in _SESSION_ROUTED_TOOLS:
            return
        candidate.args = _normalize_write_argument_aliases(dict(getattr(candidate, "args", {}) or {}))
        signature = (
            candidate.tool_name,
            json.dumps(candidate.args, ensure_ascii=True, sort_keys=True),
        )
        if signature in seen:
            return
        seen.add(signature)
        results.append(candidate)

    _remember(_pending_from_embedded_xml_text(cleaned))
    _remember(_pending_from_embedded_json_text(cleaned))

    for pattern in (r"<tool_code>(.*?)</tool_code>", r"<tool_call>(.*?)</tool_call>", r"<call>(.*?)</call>"):
        for match in re.finditer(pattern, cleaned, re.DOTALL):
            block = str(match.group(1) or "").strip()
            _remember(_pending_from_embedded_xml_text(block))
            _remember(_pending_from_embedded_json_text(block))

    for match in re.finditer(r"\[(file_write|file_append|file_patch|ast_patch)\]\s*(\{.*?\})", cleaned, re.DOTALL):
        _remember(
            PendingToolCall.from_payload(
                {
                    "function": {
                        "name": str(match.group(1) or "").strip(),
                        "arguments": str(match.group(2) or "").strip(),
                    }
                }
            )
        )

    fenced = _extract_first_fenced_block(cleaned)
    if fenced:
        _remember(_pending_from_embedded_xml_text(fenced))
        _remember(_pending_from_embedded_json_text(fenced))
    return results


def _python_content_looks_complete(content: str) -> bool:
    line_count = len([line for line in str(content or "").splitlines() if line.strip()])
    if line_count >= 120:
        return True

    has_imports = bool(re.search(r"(?m)^\s*(?:from\s+\S+\s+import|import\s+\S+)", content))
    has_defs_or_classes = bool(re.search(r"(?m)^\s*(?:async\s+def|def|class)\s+\w+", content))
    has_tests = bool(
        re.search(r"(?m)^\s*(?:class\s+Test\w*|def\s+test_\w+)", content)
        or "unittest.main" in content
        or "pytest" in content
    )
    has_entrypoint = bool(re.search(r'(?m)^\s*if\s+__name__\s*==\s*["\']__main__["\']\s*:', content))
    if has_imports and has_defs_or_classes and (has_tests or has_entrypoint):
        return True
    if line_count >= 60 and has_defs_or_classes and has_entrypoint:
        return True
    return False


def _javascript_content_looks_complete(content: str) -> bool:
    line_count = len([line for line in str(content or "").splitlines() if line.strip()])
    if line_count >= 140:
        return True

    has_imports = bool(re.search(r"(?m)^\s*(?:import\s+.+from\s+|const\s+\w+\s*=\s*require\()", content))
    has_exports = bool(re.search(r"(?m)^\s*export\s+", content) or "module.exports" in content)
    has_tests = bool(re.search(r"\b(?:describe|it|test)\s*\(", content))
    return has_imports and has_exports and has_tests


def _go_content_looks_complete(content: str) -> bool:
    line_count = len([line for line in str(content or "").splitlines() if line.strip()])
    if line_count >= 140:
        return True

    has_package = bool(re.search(r"(?m)^\s*package\s+\w+", content))
    has_imports = bool(re.search(r"(?m)^\s*import\s+(?:\(|\w)", content))
    has_functions = bool(re.search(r"(?m)^\s*func\s+\w+", content))
    has_tests = bool(re.search(r"(?m)^\s*func\s+Test\w+\s*\(", content))
    return has_package and has_imports and has_functions and has_tests


def recovered_content_looks_like_complete_file(intent: Any) -> bool:
    target_path = str(getattr(intent, "path", "") or "").strip().lower()
    content = str(getattr(intent, "content", "") or "")
    if not target_path or not content.strip():
        return False
    if target_path.endswith(".py"):
        return _python_content_looks_complete(content)
    if target_path.endswith((".js", ".ts", ".tsx")):
        return _javascript_content_looks_complete(content)
    if target_path.endswith(".go"):
        return _go_content_looks_complete(content)
    return False


def _force_finalize_if_complete_file(intent: Any) -> bool:
    if not recovered_content_looks_like_complete_file(intent):
        return False
    intent.next_section_name = ""
    intent.next_section_name_origin = ""
    intent.evidence.append("complete_file_content")
    return True


def maybe_finalize_recovered_assistant_write(intent: Any) -> bool:
    source = str(getattr(intent, "source", "") or "").strip()
    if source not in {"assistant_text", "tool_args_content", "partial_tool_arguments"}:
        return False
    origin = str(getattr(intent, "next_section_name_origin", "") or "").strip()
    if origin not in {"session_default", "tool_args", "partial_tool_calls", "assistant_payload"}:
        return False

    if _force_finalize_if_complete_file(intent):
        intent.evidence.append("cleared_session_default_next_section_name")
        return True
    return False
