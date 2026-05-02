from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .models.tool_result import ToolEnvelope
from .state import ArtifactRecord, EvidenceRecord, json_safe_value
from .tool_output_formatting import summarize_structured_output

_EVIDENCE_ID_PREFIX = "E"
_VERIFIER_COMMAND_MARKERS = (
    "pytest",
    "unittest",
    "nose",
    "go test",
    "cargo test",
    "npm test",
    "pnpm test",
    "yarn test",
    "mvn test",
    "gradle test",
    "ctest",
    "ruff",
    "mypy",
    "verifier",
    "lint",
)
_SHELL_OBSERVATION_LIST_MARKERS = (
    "rg ",
    "grep ",
    "findstr ",
    "ack ",
    "ag ",
    "journalctl",
    "tail ",
    "head ",
)


def normalize_tool_result(
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: ArtifactRecord | None = None,
    operation_id: str = "",
    phase: str = "",
    evidence_context: dict[str, Any] | None = None,
    replayed: bool = False,
) -> EvidenceRecord:
    context = dict(evidence_context or {})
    metadata = _build_metadata(
        tool_name=tool_name,
        result=result,
        artifact=artifact,
        context=context,
        operation_id=operation_id,
        phase=phase,
        replayed=replayed or _coerce_bool(context.get("replayed"), default=False),
    )
    artifact_id = str(
        (artifact.artifact_id if artifact else "")
        or context.get("artifact_id")
        or result.metadata.get("artifact_id")
        or ""
    ).strip()
    source = str((artifact.source if artifact else "") or metadata.get("source") or tool_name).strip()
    replayed_or_cached = bool(
        replayed
        or _coerce_bool(context.get("replayed"), default=False)
        or _coerce_bool(result.metadata.get("cache_hit"), default=False)
    )
    negative = not bool(result.success)
    evidence_type = "replayed_or_cached" if replayed_or_cached else "negative_evidence" if negative else "direct_observation"
    statement = _build_statement(
        tool_name=tool_name,
        result=result,
        artifact=artifact,
        artifact_id=artifact_id,
        replayed=replayed_or_cached,
    )
    evidence_id = _derive_evidence_id(
        artifact_id=artifact_id,
        operation_id=operation_id,
        tool_name=tool_name,
        result=result,
        context=context,
    )
    claim_ids = _coerce_string_list(context.get("claim_ids"))
    decision_ids = _coerce_string_list(context.get("decision_ids"))
    evidence_refs = _coerce_string_list(context.get("evidence_refs"))
    confidence = _derive_confidence(result=result, replayed=replayed_or_cached)
    kind = "negative_observation" if negative else "observation"

    return EvidenceRecord(
        evidence_id=evidence_id,
        kind=kind,
        statement=statement,
        phase=str((phase or context.get("phase") or "")).strip(),
        tool_name=str(tool_name or "").strip(),
        operation_id=str((operation_id or context.get("operation_id") or "")).strip(),
        artifact_id=artifact_id,
        source=source,
        evidence_type=evidence_type,
        confidence=confidence,
        negative=negative,
        replayed=replayed_or_cached,
        claim_ids=claim_ids,
        decision_ids=decision_ids,
        evidence_refs=evidence_refs,
        metadata=metadata,
    )


def _build_metadata(
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: ArtifactRecord | None,
    context: dict[str, Any],
    operation_id: str,
    phase: str,
    replayed: bool,
) -> dict[str, Any]:
    args = context.get("args") or result.metadata.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    metadata: dict[str, Any] = {
        "tool_name": tool_name,
        "operation_id": str(operation_id or context.get("operation_id") or "").strip(),
        "phase": str(phase or context.get("phase") or "").strip(),
        "success": bool(result.success),
        "status": result.status,
        "error": result.error,
        "cache_hit": bool(result.metadata.get("cache_hit")),
        "replayed": bool(replayed),
        "arguments": json_safe_value(args),
    }
    for key in (
        "path",
        "url",
        "command",
        "status",
        "reason",
        "question",
        "artifact_id",
        "source_artifact_id",
        "line_start",
        "line_end",
        "requested_start_line",
        "requested_end_line",
        "total_lines",
        "complete_file",
        "tool_name",
    ):
        value = result.metadata.get(key)
        if value is not None:
            metadata[key] = json_safe_value(value)
    if artifact is not None:
        metadata["artifact_summary"] = artifact.summary
        metadata["artifact_source"] = artifact.source
        metadata["artifact_kind"] = artifact.kind
        metadata["artifact_id"] = artifact.artifact_id
    if context.get("artifact_id") and "artifact_id" not in metadata:
        metadata["artifact_id"] = str(context.get("artifact_id"))
    if context.get("summary"):
        metadata["summary"] = str(context.get("summary"))
    if context.get("replay_source"):
        metadata["replay_source"] = str(context.get("replay_source"))
    if "path" not in metadata:
        arg_path = str(args.get("path") or "").strip()
        if arg_path:
            metadata["path"] = arg_path
    if "command" not in metadata:
        arg_command = str(args.get("command") or "").strip()
        if arg_command:
            metadata["command"] = arg_command

    adapter_payload = _build_observation_adapter_payload(
        tool_name=tool_name,
        result=result,
        artifact=artifact,
        metadata=metadata,
        args=args,
        replayed=replayed,
        context=context,
    )
    for key, value in adapter_payload.items():
        if value in (None, "", [], {}):
            continue
        metadata[key] = json_safe_value(value)
    return metadata


def _build_observation_adapter_payload(
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: ArtifactRecord | None,
    metadata: dict[str, Any],
    args: dict[str, Any],
    replayed: bool,
    context: dict[str, Any],
) -> dict[str, Any]:
    normalized_tool = str(tool_name or "").strip().lower()
    command = _first_text(
        metadata.get("command"),
        args.get("command"),
        context.get("command"),
        metadata.get("target"),
    )
    path = _first_text(
        metadata.get("path"),
        args.get("path"),
        context.get("path"),
        artifact.source if artifact is not None else "",
    )
    query = _first_text(
        metadata.get("query"),
        metadata.get("pattern"),
        args.get("query"),
        args.get("pattern"),
        args.get("needle"),
    )
    observation_items = _extract_observation_items(result.output)

    if replayed or _coerce_bool(metadata.get("cache_hit"), default=False):
        return {
            "observation_adapter": "artifact_replay",
            "observation_kind": "artifact_replay",
            "source_artifact_id": _first_text(
                metadata.get("source_artifact_id"),
                context.get("source_artifact_id"),
                metadata.get("artifact_id"),
            ),
            "replay_source": _first_text(metadata.get("replay_source"), context.get("replay_source")),
        }

    if normalized_tool == "file_read":
        payload: dict[str, Any] = {
            "observation_adapter": "file_read_fact",
            "observation_kind": "file_fact",
        }
        if path:
            payload["path"] = path
        return payload

    if normalized_tool in {"file_write", "file_append", "file_patch", "ast_patch", "file_delete", "dir_list"}:
        payload = {
            "observation_adapter": "file_state",
            "observation_kind": "file_fact",
        }
        if path:
            payload["path"] = path
        return payload

    if normalized_tool in {"shell_exec", "ssh_exec"}:
        verifier_payload = _build_verifier_adapter_payload(
            result=result,
            metadata=metadata,
            command=command,
            context=context,
        )
        if verifier_payload:
            return verifier_payload
        if _looks_like_shell_observation_list(command=command, metadata=metadata):
            payload = {
                "observation_adapter": "shell_observation_list",
                "observation_kind": "observation_list",
                "command": command,
            }
            if query:
                payload["query"] = query
            if observation_items:
                payload["observation_items"] = observation_items[:6]
            return payload
        payload = {
            "observation_adapter": "shell_observation",
            "observation_kind": "shell_observation",
        }
        if command:
            payload["command"] = command
        if observation_items:
            payload["observation_items"] = observation_items[:4]
        return payload

    if normalized_tool in {"search", "web_search"}:
        payload = {
            "observation_adapter": "search_observation_list",
            "observation_kind": "observation_list",
        }
        if query:
            payload["query"] = query
        if command:
            payload["command"] = command
        if path:
            payload["path"] = path
        if observation_items:
            payload["observation_items"] = observation_items[:6]
        return payload

    if normalized_tool == "web_fetch":
        payload = {
            "observation_adapter": "web_fetch_observation",
            "observation_kind": "web_observation",
        }
        if path:
            payload["path"] = path
        if query:
            payload["query"] = query
        web_item = _format_observation_item(result.output if isinstance(result.output, dict) else result.output)
        if web_item:
            payload["observation_items"] = [web_item]
        elif observation_items:
            payload["observation_items"] = observation_items[:2]
        return payload

    if normalized_tool == "artifact_read" and observation_items:
        payload = {
            "observation_adapter": "artifact_observation_list",
            "observation_kind": "observation_list",
            "source_artifact_id": _first_text(metadata.get("artifact_id"), metadata.get("source_artifact_id")),
            "observation_items": observation_items[:6],
        }
        if query:
            payload["query"] = query
        return payload

    return {}


def _build_verifier_adapter_payload(
    *,
    result: ToolEnvelope,
    metadata: dict[str, Any],
    command: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    output = result.output if isinstance(result.output, dict) else {}
    raw_verdict = _first_text(
        metadata.get("verdict"),
        output.get("verdict") if isinstance(output, dict) else "",
        context.get("verdict"),
    ).lower()
    exit_code = _coerce_exit_code(
        output.get("exit_code") if isinstance(output, dict) else None,
        metadata.get("exit_code"),
    )
    is_verifier = bool(raw_verdict) or exit_code is not None or _looks_like_verifier_command(command)
    if not is_verifier:
        return {}
    verdict = raw_verdict
    if not verdict:
        status = str(result.status or metadata.get("status") or "").strip().lower()
        if status == "needs_human":
            verdict = "needs_human"
        elif bool(result.success) and exit_code in (None, 0):
            verdict = "pass"
        else:
            verdict = "fail"

    payload: dict[str, Any] = {
        "observation_adapter": "verifier_verdict",
        "observation_kind": "verifier_verdict",
        "verdict": verdict,
    }
    if command:
        payload["command"] = command
    if exit_code is not None:
        payload["exit_code"] = exit_code
    failure_mode = _first_text(
        metadata.get("failure_mode"),
        output.get("failure_mode") if isinstance(output, dict) else "",
        context.get("failure_mode"),
    )
    if failure_mode:
        payload["failure_mode"] = failure_mode
    return payload


def _looks_like_verifier_command(command: str) -> bool:
    lowered = str(command or "").strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _VERIFIER_COMMAND_MARKERS)


def _looks_like_shell_observation_list(*, command: str, metadata: dict[str, Any]) -> bool:
    lowered = str(command or "").strip().lower()
    if any(marker in lowered for marker in _SHELL_OBSERVATION_LIST_MARKERS):
        return True
    if metadata.get("query") is not None or metadata.get("pattern") is not None:
        return True
    return bool(metadata.get("line_start") is not None and metadata.get("line_end") is not None)


def _extract_observation_items(output: Any, *, limit: int = 8) -> list[str]:
    items: list[str] = []
    if isinstance(output, list):
        items = [_format_observation_item(item) for item in output]
    elif isinstance(output, dict):
        for key in ("matches", "results", "items", "lines", "observations"):
            value = output.get(key)
            if isinstance(value, list) and value:
                items = [_format_observation_item(item) for item in value]
                break
        if not items:
            for key in ("stdout", "stderr", "output"):
                value = output.get(key)
                if isinstance(value, str) and value.strip():
                    items = _extract_non_empty_lines(value, limit=limit)
                    break
    elif isinstance(output, str):
        items = _extract_non_empty_lines(output, limit=limit)
    cleaned: list[str] = []
    for item in items:
        normalized = _shorten(item, limit=200)
        if not normalized:
            continue
        if normalized not in cleaned:
            cleaned.append(normalized)
        if len(cleaned) >= limit:
            break
    return cleaned


def _extract_non_empty_lines(text: str, *, limit: int = 8) -> list[str]:
    lines = [line.strip() for line in str(text or "").splitlines() if str(line).strip()]
    return lines[:limit]


def _format_observation_item(item: Any) -> str:
    if isinstance(item, dict):
        path = _first_text(item.get("path"), item.get("file"))
        line = _first_text(item.get("line"), item.get("line_number"), item.get("lineno"))
        text = _first_text(item.get("text"), item.get("message"), item.get("summary"), item.get("value"))
        title = _first_text(item.get("title"), item.get("name"))
        result_id = _first_text(item.get("fetch_id"), item.get("result_id"), item.get("source_id"))
        domain = _first_text(item.get("domain"))
        url = _first_text(item.get("url"), item.get("canonical_url"))
        snippet = _first_text(item.get("snippet"), item.get("text_excerpt"), item.get("untrusted_text"))
        head = path
        if line:
            head = f"{head}:{line}" if head else line
        if head and text:
            return f"{head} {text}".strip()
        if text:
            return text
        if title or url or snippet or result_id or domain:
            parts: list[str] = []
            if title:
                parts.append(title)
            elif url:
                parts.append(url)
            meta: list[str] = []
            if result_id:
                meta.append(result_id)
            if domain:
                meta.append(domain)
            if meta:
                parts.append(f"({' | '.join(meta)})")
            if snippet:
                parts.append(snippet)
            return " ".join(part for part in parts if part).strip()
        return json.dumps(json_safe_value(item), ensure_ascii=True)
    if isinstance(item, str):
        return item.strip()
    return str(item or "").strip()


def _coerce_exit_code(*values: Any) -> int | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _build_statement(
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: ArtifactRecord | None,
    artifact_id: str,
    replayed: bool,
) -> str:
    if not result.success:
        error = _shorten(str(result.error or result.metadata.get("error") or "tool failed"))
        return f"{tool_name} failed: {error}" if error else f"{tool_name} failed"

    if replayed and artifact_id:
        summary = _shorten(artifact.summary if artifact else "")
        if summary:
            return f"{tool_name}: reused {artifact_id}: {summary}"
        return f"{tool_name}: reused {artifact_id}"

    summary = ""
    if artifact is not None:
        summary = artifact.summary
    elif isinstance(result.output, str):
        summary = _shorten(result.output)
    elif isinstance(result.output, dict):
        structured_summary = summarize_structured_output(tool_name=tool_name, output=result.output)
        if structured_summary:
            summary = structured_summary
        else:
            keys = ", ".join(sorted(result.output.keys())[:5])
            summary = f"{tool_name} keys: {keys}" if keys else tool_name
    elif isinstance(result.output, list):
        summary = f"{tool_name} returned {len(result.output)} items"

    summary = _shorten(summary)
    if summary:
        return f"{tool_name}: {summary}" if not summary.startswith(tool_name) else summary
    return tool_name


def _derive_confidence(*, result: ToolEnvelope, replayed: bool) -> float:
    if not result.success:
        return 0.85
    if replayed:
        return 0.7
    return 0.95


def _derive_evidence_id(
    *,
    artifact_id: str,
    operation_id: str,
    tool_name: str,
    result: ToolEnvelope,
    context: dict[str, Any],
) -> str:
    seed = "|".join(
        [
            artifact_id,
            operation_id,
            tool_name,
            str(bool(result.success)),
            str(result.status or ""),
            str(result.error or ""),
            json.dumps(json_safe_value(context.get("args") or result.metadata.get("arguments") or {}), sort_keys=True, ensure_ascii=True, default=str),
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    if artifact_id:
        return f"{_EVIDENCE_ID_PREFIX}-{artifact_id}"
    if operation_id:
        return f"{_EVIDENCE_ID_PREFIX}-{operation_id}"
    return f"{_EVIDENCE_ID_PREFIX}-{digest}"


def _shorten(text: str, *, limit: int = 220) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set, frozenset)):
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    return []
