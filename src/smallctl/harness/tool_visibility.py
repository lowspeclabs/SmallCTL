from __future__ import annotations

from pathlib import Path
from typing import Any

from ..remote_scope import remote_scope_is_active
from ..state import json_safe_value
from ..tools.fs_sessions import _write_session_can_finalize

_INDEX_QUERY_TOOL_NAMES = {
    "index_query_symbol",
    "index_get_definition",
    "index_get_references",
}
_ARTIFACT_TOOL_NAMES = {
    "artifact_read",
    "artifact_grep",
    "artifact_print",
}
_PLAN_DEPENDENT_TOOL_NAMES = {
    "plan_step_update",
    "plan_export",
    "plan_request_execution",
}
_BACKGROUND_JOB_TOOL_NAMES = {
    "process_kill",
}
_LOOPISH_TOOL_MODES = {"loop", "execute"}
_RETRYABLE_HIDDEN_TOOL_NAMES = {
    "shell_exec",
    "ssh_exec",
    "ssh_file_read",
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
    "file_patch",
    "ast_patch",
    "file_write",
    "file_append",
    "finalize_write_session",
    "web_fetch",
}


def _tool_name(entry: dict[str, Any]) -> str:
    function = entry.get("function") if isinstance(entry, dict) else None
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()


def _normalize_turn_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized in _LOOPISH_TOOL_MODES:
        return "loop"
    if normalized in {"chat", "planning", "indexer"}:
        return normalized
    return "loop"


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for entry in tools:
        name = _tool_name(entry)
        if name:
            names.append(name)
    return names


def _schema_for_name(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {},
        },
    }


def _scratchpad(state: Any) -> dict[str, Any]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    return scratchpad


def recent_hidden_tool_recovery_artifact_id(
    state: Any,
    *,
    tool_name: str,
) -> str:
    normalized_tool_name = str(tool_name or "").strip()
    if normalized_tool_name != "web_fetch":
        return ""

    artifacts = getattr(state, "artifacts", {})
    handoff = _scratchpad(state).get("_last_task_handoff")
    if isinstance(handoff, dict):
        recent_ids = handoff.get("recent_research_artifact_ids")
        if isinstance(recent_ids, list):
            for artifact_id in reversed(recent_ids):
                normalized_id = str(artifact_id or "").strip()
                artifact = artifacts.get(normalized_id) if isinstance(artifacts, dict) else None
                if normalized_id and artifact is not None and str(getattr(artifact, "kind", "")).strip() == "web_fetch":
                    return normalized_id

    records = getattr(state, "tool_execution_records", None)
    if not isinstance(records, dict):
        return ""
    items = [record for record in records.values() if isinstance(record, dict)]
    items.sort(key=lambda record: (int(record.get("step_count") or 0), str(record.get("operation_id") or "")))
    for record in reversed(items):
        if str(record.get("tool_name") or "").strip() != "web_fetch":
            continue
        result = record.get("result")
        if not isinstance(result, dict) or not bool(result.get("success")):
            continue
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            continue
        artifact_id = str(metadata.get("artifact_id") or "").strip()
        artifact = artifacts.get(artifact_id) if isinstance(artifacts, dict) else None
        if artifact_id and artifact is not None and str(getattr(artifact, "kind", "")).strip() == "web_fetch":
            return artifact_id
    return ""


def _export_registry_tools(
    harness: Any,
    *,
    mode: str,
    phase: str | None = None,
    profiles: set[str] | None = None,
) -> list[dict[str, Any]]:
    registry = getattr(harness, "registry", None)
    export_fn = getattr(registry, "export_openai_tools", None) if registry is not None else None
    if callable(export_fn):
        kwargs: dict[str, Any] = {"mode": mode}
        if phase is not None:
            kwargs["phase"] = phase
        if profiles is not None:
            kwargs["profiles"] = profiles
        return list(export_fn(**kwargs))

    names_fn = getattr(registry, "names", None) if registry is not None else None
    if callable(names_fn):
        try:
            return [_schema_for_name(str(name).strip()) for name in names_fn() if str(name).strip()]
        except Exception:
            return []
    return []


def _retry_tool_schema(harness: Any, *, tool_name: str) -> dict[str, Any] | None:
    if tool_name in {"ssh_exec", "ssh_file_read", "ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"} and not remote_scope_is_active(getattr(harness, "state", None)):
        return None
    registry = getattr(harness, "registry", None)
    get_fn = getattr(registry, "get", None) if registry is not None else None
    spec = get_fn(tool_name) if callable(get_fn) else None
    if spec is not None:
        phase_allowed = getattr(spec, "phase_allowed", None)
        if callable(phase_allowed) and not phase_allowed(getattr(harness.state, "current_phase", "")):
            return None
        profile_allowed = getattr(spec, "profile_allowed", None)
        if callable(profile_allowed) and not profile_allowed(set(getattr(harness.state, "active_tool_profiles", []) or [])):
            return None
        openai_schema = getattr(spec, "openai_schema", None)
        if callable(openai_schema):
            schema = openai_schema()
            if isinstance(schema, dict):
                return schema
    fallback_schemas = _export_registry_tools(
        harness,
        mode="loop",
        phase=getattr(harness.state, "current_phase", None),
        profiles=set(getattr(harness.state, "active_tool_profiles", []) or []),
    )
    for entry in fallback_schemas:
        if _tool_name(entry) == tool_name:
            return entry
    return None


def schedule_retry_tool_exposure(
    state: Any,
    *,
    mode: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> bool:
    normalized_mode = _normalize_turn_mode(mode)
    normalized_tool_name = str(tool_name or "").strip()
    if normalized_mode not in {"chat", "planning", "loop"}:
        return False
    if normalized_mode == "loop" and normalized_tool_name != "web_fetch":
        return False
    if normalized_tool_name not in _RETRYABLE_HIDDEN_TOOL_NAMES:
        return False
    scratchpad = _scratchpad(state)
    payloads = scratchpad.get("_retry_tool_exposures")
    if not isinstance(payloads, list):
        payloads = []
    normalized_arguments = arguments if isinstance(arguments, dict) else {}
    filtered = [
        item
        for item in payloads
        if not (
            isinstance(item, dict)
            and str(item.get("mode") or "").strip().lower() == normalized_mode
            and str(item.get("tool_name") or "").strip() == normalized_tool_name
        )
    ]
    filtered.append(
        {
            "mode": normalized_mode,
            "tool_name": normalized_tool_name,
            "arguments": json_safe_value(normalized_arguments),
        }
    )
    scratchpad["_retry_tool_exposures"] = filtered
    return True


def consume_retry_tool_exposure(
    state: Any,
    *,
    mode: str,
    tool_name: str,
) -> None:
    scratchpad = _scratchpad(state)
    payloads = scratchpad.get("_retry_tool_exposures")
    if not isinstance(payloads, list) or not payloads:
        return
    normalized_mode = _normalize_turn_mode(mode)
    normalized_tool_name = str(tool_name or "").strip()
    filtered = [
        item
        for item in payloads
        if not (
            isinstance(item, dict)
            and str(item.get("mode") or "").strip().lower() == normalized_mode
            and str(item.get("tool_name") or "").strip() == normalized_tool_name
        )
    ]
    if filtered:
        scratchpad["_retry_tool_exposures"] = filtered
    else:
        scratchpad.pop("_retry_tool_exposures", None)


def _append_retry_tool_exposures(
    harness: Any,
    schemas: list[dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    scratchpad = _scratchpad(harness.state)
    payloads = scratchpad.get("_retry_tool_exposures")
    if not isinstance(payloads, list) or not payloads:
        return list(schemas)
    normalized_mode = _normalize_turn_mode(mode)
    existing_names = set(_tool_names(schemas))
    merged = list(schemas)
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if str(payload.get("mode") or "").strip().lower() != normalized_mode:
            continue
        tool_name = str(payload.get("tool_name") or "").strip()
        if not tool_name or tool_name in existing_names:
            continue
        schema = _retry_tool_schema(harness, tool_name=tool_name)
        if not isinstance(schema, dict):
            continue
        merged.append(schema)
        existing_names.add(tool_name)
    return merged


def _has_runtime_code_index(cwd: str | None) -> bool:
    if not str(cwd or "").strip():
        return False
    return (Path(str(cwd)) / ".smallctl" / "index.db").exists()


def _has_finalizable_write_session(state: Any) -> bool:
    session = getattr(state, "write_session", None)
    if session is None:
        return False
    if str(getattr(session, "status", "") or "").strip().lower() == "complete":
        return False
    if not _write_session_can_finalize(session):
        return False
    if not bool(getattr(session, "write_sections_completed", None)):
        return False
    if str(getattr(session, "write_next_section", "") or "").strip():
        return False
    return str(getattr(session, "status", "open") or "open").strip().lower() in {"open", "verifying"}


def _has_artifacts(state: Any) -> bool:
    artifacts = getattr(state, "artifacts", None)
    return isinstance(artifacts, dict) and bool(artifacts)


def _has_plan(state: Any) -> bool:
    return getattr(state, "draft_plan", None) is not None or getattr(state, "active_plan", None) is not None


def _has_background_jobs(state: Any) -> bool:
    jobs = getattr(state, "background_processes", None)
    return isinstance(jobs, dict) and bool(jobs)


def filter_tools_for_runtime_state(
    tools: list[dict[str, Any]],
    *,
    state: Any,
    mode: str,
) -> list[dict[str, Any]]:
    normalized_mode = _normalize_turn_mode(mode)
    if normalized_mode == "indexer":
        return list(tools)

    cwd = getattr(state, "cwd", None)
    has_index = _has_runtime_code_index(cwd)
    can_finalize_write_session = _has_finalizable_write_session(state)
    has_artifacts = _has_artifacts(state)
    has_plan = _has_plan(state)
    has_background_jobs = _has_background_jobs(state)

    filtered: list[dict[str, Any]] = []
    for entry in tools:
        tool_name = _tool_name(entry)
        if tool_name in _INDEX_QUERY_TOOL_NAMES and not has_index:
            continue
        if tool_name in _ARTIFACT_TOOL_NAMES and not has_artifacts:
            continue
        if tool_name in _PLAN_DEPENDENT_TOOL_NAMES and not has_plan:
            continue
        if tool_name in _BACKGROUND_JOB_TOOL_NAMES and not has_background_jobs:
            continue
        if tool_name == "finalize_write_session" and not can_finalize_write_session:
            continue
        filtered.append(entry)
    return filtered


def hidden_tool_reason(
    tool_name: str,
    *,
    state: Any,
    mode: str,
) -> str | None:
    normalized_mode = _normalize_turn_mode(mode)
    normalized_tool_name = str(tool_name or "").strip()
    if not normalized_tool_name or normalized_mode == "indexer":
        return None

    cwd = getattr(state, "cwd", None)
    if normalized_tool_name in _INDEX_QUERY_TOOL_NAMES and not _has_runtime_code_index(cwd):
        return "missing_index"
    if normalized_tool_name in _ARTIFACT_TOOL_NAMES and not _has_artifacts(state):
        return "no_artifacts"
    if normalized_tool_name in _PLAN_DEPENDENT_TOOL_NAMES and not _has_plan(state):
        return "no_active_plan"
    if normalized_tool_name in _BACKGROUND_JOB_TOOL_NAMES and not _has_background_jobs(state):
        return "no_background_jobs"
    if normalized_tool_name == "finalize_write_session" and not _has_finalizable_write_session(state):
        return "write_session_not_finalizable"
    return None


def resolve_turn_tool_exposure(harness: Any, mode: str) -> dict[str, list[Any]]:
    normalized_mode = _normalize_turn_mode(mode)

    if normalized_mode == "chat":
        chat_mode_tools = getattr(harness, "_chat_mode_tools", None)
        if callable(chat_mode_tools):
            schemas = list(chat_mode_tools())
        else:
            from .tool_dispatch import chat_mode_tools as chat_mode_tools_helper

            schemas = list(chat_mode_tools_helper(harness))
        schemas = _append_retry_tool_exposures(harness, schemas, mode=normalized_mode)
        return {
            "schemas": schemas,
            "names": _tool_names(schemas),
        }

    if normalized_mode == "indexer":
        schemas = _export_registry_tools(
            harness,
            mode="indexer",
            profiles={"indexer", "core", "support"},
        )
        return {
            "schemas": schemas,
            "names": _tool_names(schemas),
        }

    schemas = _export_registry_tools(
        harness,
        phase=harness.state.current_phase,
        mode=normalized_mode,
        profiles=set(harness.state.active_tool_profiles),
    )
    schemas = filter_tools_for_runtime_state(
        schemas,
        state=harness.state,
        mode=normalized_mode,
    )
    schemas = _append_retry_tool_exposures(harness, schemas, mode=normalized_mode)
    return {
        "schemas": schemas,
        "names": _tool_names(schemas),
    }
