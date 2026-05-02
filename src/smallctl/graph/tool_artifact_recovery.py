from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state import PendingToolCall
from .tool_loop_guards import _tool_attempt_history


def _extract_artifact_id_from_args(args: dict[str, Any]) -> str | None:
    if not isinstance(args, dict):
        return None

    for key in ("artifact_id", "path", "id"):
        value = args.get(key)
        if not isinstance(value, str):
            continue
        candidate = Path(value.strip()).stem.strip()
        if candidate:
            return candidate
    return None


def _resolve_artifact_record(harness: Any, artifact_id: str) -> Any | None:
    artifact = harness.state.artifacts.get(artifact_id)
    if artifact is not None:
        return artifact

    if not artifact_id.startswith("A"):
        return None

    try:
        numeric_val = int(artifact_id[1:])
    except ValueError:
        return None

    for aid, record in harness.state.artifacts.items():
        if not isinstance(aid, str) or not aid.startswith("A"):
            continue
        try:
            if int(aid[1:]) == numeric_val:
                return record
        except ValueError:
            continue
    return None


def _read_artifact_text(artifact: Any) -> str:
    content_path = getattr(artifact, "content_path", None)
    if isinstance(content_path, str) and content_path.strip():
        path = Path(content_path)
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                pass

    inline_content = getattr(artifact, "inline_content", None)
    if isinstance(inline_content, str) and inline_content:
        return inline_content
    return ""


def _choose_artifact_grep_query(content: str) -> str | None:
    lowered = content.lower()
    if not any(marker in lowered for marker in ("nmap scan report", "/tcp", "/udp", "host is up")):
        return None
    for query in ("open", "port", "service", "banner", "nmap scan report", "host is up"):
        if query in lowered:
            return query
    return None


def _resolve_file_read_path(harness: Any, args: dict[str, Any]) -> Path | None:
    raw_path = args.get("path")
    if not isinstance(raw_path, str):
        return None
    candidate = Path(raw_path.strip() or ".")
    if candidate.is_absolute():
        try:
            return candidate.resolve()
        except Exception:
            return candidate
    cwd = getattr(getattr(harness, "state", None), "cwd", None)
    base = Path(cwd) if isinstance(cwd, str) and cwd else Path.cwd()
    try:
        return (base / candidate).resolve()
    except Exception:
        return base / candidate


def _coerce_int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _requested_file_read_range(args: dict[str, Any]) -> tuple[int | None, int | None]:
    if not isinstance(args, dict):
        return (None, None)
    start_line = args.get("requested_start_line", args.get("start_line"))
    end_line = args.get("requested_end_line", args.get("end_line"))
    return (_coerce_int_or_none(start_line), _coerce_int_or_none(end_line))


def _requested_artifact_read_target(args: dict[str, Any]) -> str:
    if not isinstance(args, dict):
        return ""
    artifact_id = args.get("artifact_id")
    return str(artifact_id or "").strip()


def _find_full_file_artifact_for_path(harness: Any, target_path: Path) -> Any | None:
    artifacts = getattr(getattr(harness, "state", None), "artifacts", {})
    if not isinstance(artifacts, dict) or not artifacts:
        return None

    for _, artifact in reversed(list(artifacts.items())):
        if getattr(artifact, "kind", "") != "file_read":
            continue
        metadata = getattr(artifact, "metadata", {})
        if not isinstance(metadata, dict) or not metadata.get("complete_file"):
            continue
        artifact_path = _resolve_file_read_path(
            harness,
            {"path": metadata.get("path") or getattr(artifact, "source", "")},
        )
        if artifact_path is None:
            continue
        if artifact_path == target_path:
            return artifact
    return None


def _fallback_repeated_artifact_read(harness: Any, pending: PendingToolCall) -> PendingToolCall | None:
    if pending.tool_name != "artifact_read":
        return None

    artifact_id = _extract_artifact_id_from_args(pending.args)
    if not artifact_id:
        return None

    artifact = _resolve_artifact_record(harness, artifact_id)
    if artifact is None:
        return None

    content = _read_artifact_text(artifact)
    if not content:
        return None

    query = _choose_artifact_grep_query(content)
    if not query:
        return None

    return PendingToolCall(
        tool_name="artifact_grep",
        args={
            "artifact_id": artifact.artifact_id,
            "query": query,
        },
        raw_arguments=json.dumps(
            {"artifact_id": artifact.artifact_id, "query": query},
            ensure_ascii=True,
            sort_keys=True,
        ),
        tool_call_id=pending.tool_call_id,
        source="system",
    )


def _fallback_repeated_file_read(harness: Any, pending: PendingToolCall) -> PendingToolCall | None:
    if pending.tool_name != "file_read":
        return None

    candidate_path = _resolve_file_read_path(harness, pending.args)
    if candidate_path is None:
        return None

    artifact = _find_full_file_artifact_for_path(harness, candidate_path)
    if artifact is None:
        return None

    recovered_args: dict[str, Any] = {"artifact_id": artifact.artifact_id}
    start_line, end_line = _requested_file_read_range(pending.args)
    if start_line is not None:
        recovered_args["start_line"] = start_line
    if end_line is not None:
        recovered_args["end_line"] = end_line
    if start_line is None and end_line is None:
        recovered_args["start_line"] = 1

    return PendingToolCall(
        tool_name="artifact_read",
        args=recovered_args,
        raw_arguments=json.dumps(recovered_args, ensure_ascii=True, sort_keys=True),
        tool_call_id=pending.tool_call_id,
        source="system",
    )


def _artifact_read_recovery_hint(harness: Any, guard_error: str) -> tuple[str, str] | None:
    if "artifact_read" not in guard_error and "max_consecutive_errors" not in guard_error:
        return None

    if "max_consecutive_errors" in guard_error:
        recent_errors = getattr(harness.state, "recent_errors", [])
        if not recent_errors or not all("artifact_read" in str(err) for err in recent_errors):
            return None

    history = getattr(harness.state, "tool_history", [])
    if not isinstance(history, list) or not history:
        return None

    for fingerprint in reversed(history):
        if not isinstance(fingerprint, str) or not fingerprint.startswith("artifact_read|"):
            continue
        parts = fingerprint.split("|", 2)
        if len(parts) < 2:
            continue
        try:
            args = json.loads(parts[1])
        except Exception:
            continue
        if not isinstance(args, dict):
            continue
        recovered = _fallback_repeated_artifact_read(
            harness,
            PendingToolCall(
                tool_name="artifact_read",
                args=args,
                raw_arguments=json.dumps(args, ensure_ascii=True, sort_keys=True),
                source="system",
            ),
        )
        if recovered is None:
            continue
        artifact_id = str(recovered.args.get("artifact_id", "")).strip()
        query = str(recovered.args.get("query", "")).strip()
        if artifact_id and query:
            return artifact_id, query
    return None


def _artifact_read_synthesis_hint(harness: Any, guard_error: str) -> str | None:
    if "artifact_read" not in guard_error and "max_consecutive_errors" not in guard_error:
        return None

    if "max_consecutive_errors" in guard_error:
        recent_errors = getattr(harness.state, "recent_errors", [])
        if not recent_errors or not all("artifact_read" in str(err) for err in recent_errors):
            return None

    history = _tool_attempt_history(harness)
    if not history:
        return None

    read_counts: dict[str, int] = {}
    grep_seen: set[str] = set()

    for item in history:
        tool_name = str(item.get("tool_name", ""))
        fingerprint = str(item.get("fingerprint", ""))
        if not tool_name or not fingerprint:
            continue
        try:
            payload = json.loads(fingerprint)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        args = payload.get("args", {})
        if not isinstance(args, dict):
            continue
        artifact_id = str(args.get("artifact_id", "")).strip()
        if not artifact_id:
            continue
        if tool_name == "artifact_read":
            read_counts[artifact_id] = read_counts.get(artifact_id, 0) + 1
        elif tool_name == "artifact_grep":
            grep_seen.add(artifact_id)

    for item in reversed(history):
        if str(item.get("tool_name", "")) != "artifact_read":
            continue
        fingerprint = str(item.get("fingerprint", ""))
        if not fingerprint:
            continue
        try:
            payload = json.loads(fingerprint)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        args = payload.get("args", {})
        if not isinstance(args, dict):
            continue
        artifact_id = str(args.get("artifact_id", "")).strip()
        if not artifact_id:
            continue
        if read_counts.get(artifact_id, 0) >= 3 and artifact_id in grep_seen:
            return artifact_id

    return None


def _should_suppress_resolved_plan_artifact_read(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "artifact_read":
        return False
    if not getattr(harness.state, "plan_resolved", False):
        return False
    plan_artifact_id = str(getattr(harness.state, "plan_artifact_id", "") or "").strip()
    if not plan_artifact_id:
        return False
    artifact_id = _extract_artifact_id_from_args(pending.args)
    if artifact_id != plan_artifact_id:
        return False
    return bool(harness.state.active_plan or harness.state.draft_plan or harness.state.working_memory.plan)


def _clear_artifact_read_guard_state(harness: Any, artifact_id: str) -> None:
    if not artifact_id:
        return

    recent_errors = getattr(harness.state, "recent_errors", None)
    if isinstance(recent_errors, list) and recent_errors:
        filtered_errors = [err for err in recent_errors if "artifact_read" not in str(err)]
        if len(filtered_errors) != len(recent_errors):
            harness.state.recent_errors = filtered_errors

    tool_history = getattr(harness.state, "tool_history", None)
    if isinstance(tool_history, list) and tool_history:
        kept_history: list[str] = []
        removed_entries = 0
        for entry in tool_history:
            if not isinstance(entry, str) or not entry.startswith("artifact_read|"):
                kept_history.append(entry)
                continue
            parts = entry.split("|", 2)
            if len(parts) < 3:
                kept_history.append(entry)
                continue
            try:
                args = json.loads(parts[1])
            except Exception:
                kept_history.append(entry)
                continue
            if isinstance(args, dict) and str(args.get("artifact_id", "")).strip() == artifact_id:
                removed_entries += 1
                continue
            kept_history.append(entry)
        if removed_entries:
            harness.state.tool_history = kept_history

    _clear_tool_attempt_history(harness)
