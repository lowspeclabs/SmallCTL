from __future__ import annotations

from pathlib import Path
from typing import Any

from ..remote_scope import remote_scope_is_active
from ..state import json_safe_value
from ..tools.fs_sessions import _write_session_can_finalize
from .tool_visibility_support import (
    _LOOPISH_TOOL_MODES,
    _has_artifacts,
    _has_background_jobs,
    _has_finalizable_write_session,
    _has_plan,
    _has_planning_file_patch_context,
    _has_runtime_code_index,
    _normalize_turn_mode,
    _schema_for_name,
    _scratchpad,
    _tool_name,
    _tool_names,
)

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
_LOCAL_CODING_SSH_TOOLS = {
    "ssh_exec", "ssh_file_read", "ssh_file_write",
    "ssh_file_patch", "ssh_file_replace_between",
}
_RETRYABLE_HIDDEN_TOOL_NAMES = {
    "ask_human",
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
    "task_fail",
    "web_fetch",
}


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
    if normalized_mode == "loop" and normalized_tool_name not in {"ask_human", "task_fail", "web_fetch"}:
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
        if (
            normalized_mode == "planning"
            and tool_name == "file_patch"
            and not _has_planning_file_patch_context(harness.state)
        ):
            continue
        from ..harness.task_classifier import task_is_local_coding_target
        task_text = ""
        run_brief = getattr(harness.state, "run_brief", None)
        if run_brief is not None:
            task_text = str(getattr(run_brief, "original_task", "") or "")
        if task_is_local_coding_target(task_text) and tool_name in _LOCAL_CODING_SSH_TOOLS:
            continue
        schema = _retry_tool_schema(harness, tool_name=tool_name)
        if not isinstance(schema, dict):
            continue
        merged.append(schema)
        existing_names.add(tool_name)
    return merged


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
    has_planning_file_patch_context = (
        normalized_mode != "planning" or _has_planning_file_patch_context(state)
    )

    scratchpad = getattr(state, "scratchpad", {}) or {}
    suppressed_tool = str(scratchpad.get("_repeated_tool_loop_suppressed_tool") or "").strip()
    suppressed_ttl = int(scratchpad.get("_repeated_tool_loop_suppressed_ttl", 0) or 0)
    if suppressed_tool and suppressed_ttl > 0:
        scratchpad["_repeated_tool_loop_suppressed_ttl"] = suppressed_ttl - 1
    else:
        suppressed_tool = ""
        scratchpad.pop("_repeated_tool_loop_suppressed_tool", None)
        scratchpad.pop("_repeated_tool_loop_suppressed_ttl", None)

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
        if tool_name == "file_patch" and not has_planning_file_patch_context:
            continue
        if suppressed_tool and tool_name == suppressed_tool:
            continue
        filtered.append(entry)
    from ..harness.task_classifier import task_is_local_coding_target
    task_text = ""
    run_brief = getattr(state, "run_brief", None)
    if run_brief is not None:
        task_text = str(getattr(run_brief, "original_task", "") or "")
    if task_is_local_coding_target(task_text):
        filtered = [t for t in filtered if _tool_name(t) not in _LOCAL_CODING_SSH_TOOLS]
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
    if (
        normalized_mode == "planning"
        and normalized_tool_name == "file_patch"
        and not _has_planning_file_patch_context(state)
    ):
        return "planning_file_context_required"
    return None


def resolve_turn_tool_exposure(harness: Any, mode: str) -> dict[str, list[Any]]:
    normalized_mode = _normalize_turn_mode(mode)
    from ..fama.runtime import expire_for_turn
    from ..fama.tool_policy import apply_fama_tool_exposure, fama_hidden_tools_for_exposure
    from ..harness.task_classifier import task_is_local_coding_target

    expire_for_turn(harness, mode=normalized_mode)
    config = getattr(harness, "config", None)

    task_text = ""
    run_brief = getattr(harness.state, "run_brief", None)
    if run_brief is not None:
        task_text = str(getattr(run_brief, "original_task", "") or "")
    is_local_coding = task_is_local_coding_target(task_text)

    if normalized_mode == "chat":
        chat_mode_tools = getattr(harness, "_chat_mode_tools", None)
        if callable(chat_mode_tools):
            schemas = list(chat_mode_tools())
        else:
            from .tool_dispatch import chat_mode_tools as chat_mode_tools_helper

            schemas = list(chat_mode_tools_helper(harness))
        schemas = _append_retry_tool_exposures(harness, schemas, mode=normalized_mode)
        if not is_local_coding:
            hidden_tools = fama_hidden_tools_for_exposure(
                schemas,
                state=harness.state,
                mode=normalized_mode,
                config=config,
            )
            schemas = apply_fama_tool_exposure(
                schemas,
                state=harness.state,
                mode=normalized_mode,
                config=config,
            )
            if hidden_tools:
                _log_fama_tool_exposure(harness, hidden_tools=hidden_tools, mode=normalized_mode)
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

    profiles = set(harness.state.active_tool_profiles)
    active_intent = str(getattr(harness.state, "active_intent", "") or "").strip().lower()
    if active_intent in {
        "author_write",
        "write_file",
        "requested_write_file",
        "requested_file_write",
        "requested_file_append",
        "requested_file_patch",
        "requested_ast_patch",
        "requested_file_delete",
    }:
        profiles.add("mutate")
    if profiles != set(harness.state.active_tool_profiles):
        harness.state.active_tool_profiles = sorted(profiles)
    schemas = _export_registry_tools(
        harness,
        phase=harness.state.current_phase,
        mode=normalized_mode,
        profiles=profiles,
    )
    all_exported_names = set(_tool_names(schemas))
    schemas = filter_tools_for_runtime_state(
        schemas,
        state=harness.state,
        mode=normalized_mode,
    )
    if _has_recent_tool_evidence(harness.state):
        existing_names = set(_tool_names(schemas))
        for terminal_tool in ("task_complete", "task_fail"):
            if terminal_tool in existing_names:
                continue
            schema = _retry_tool_schema(harness, tool_name=terminal_tool)
            if schema is not None:
                schemas.append(schema)
                existing_names.add(terminal_tool)
    schemas = _append_retry_tool_exposures(harness, schemas, mode=normalized_mode)

    # Phase 4C: tool exposure monotonicity — never hide a tool once exposed
    exposed = set(harness.state.task_exposed_tools)
    current_names = set(_tool_names(schemas))
    missing_exposed = exposed - current_names
    if missing_exposed:
        for tool_name in missing_exposed:
            schema = _retry_tool_schema(harness, tool_name=tool_name)
            if schema is not None:
                schemas.append(schema)
                current_names.add(tool_name)

    # Fix 2: Expose shell_exec in planning mode for local debug tasks
    if normalized_mode == "planning" and is_local_coding:
        shell_schema = _retry_tool_schema(harness, tool_name="shell_exec")
        if shell_schema is not None:
            existing_names = _tool_names(schemas)
            if "shell_exec" not in existing_names:
                schemas = list(schemas) + [shell_schema]

    exposed_names = set(_tool_names(schemas))
    hidden_names = sorted(all_exported_names - exposed_names)
    hidden_reasons = {name: hidden_tool_reason(name, state=harness.state, mode=normalized_mode) or "filtered" for name in hidden_names}

    if not is_local_coding:
        hidden_tools = fama_hidden_tools_for_exposure(
            schemas,
            state=harness.state,
            mode=normalized_mode,
            config=config,
        )
        schemas = apply_fama_tool_exposure(
            schemas,
            state=harness.state,
            mode=normalized_mode,
            config=config,
        )
        if hidden_tools:
            _log_fama_tool_exposure(harness, hidden_tools=hidden_tools, mode=normalized_mode)
        hidden_reasons.update({name: "fama_policy" for name in hidden_tools})
    # Update exposed tools set so future turns preserve monotonicity
    harness.state.task_exposed_tools = harness.state.task_exposed_tools | set(_tool_names(schemas))

    from .tool_exposure_logging import log_tool_profile_exposure

    log_tool_profile_exposure(
        harness,
        mode=normalized_mode,
        phase=harness.state.current_phase,
        profiles=profiles,
        exposed_tools=list(_tool_names(schemas)),
        hidden_tools=hidden_names,
        reasons=hidden_reasons,
    )
    return {
        "schemas": schemas,
        "names": _tool_names(schemas),
    }


def _log_fama_tool_exposure(harness: Any, *, hidden_tools: set[str], mode: str) -> None:
    from .tool_exposure_logging import log_fama_tool_exposure

    log_fama_tool_exposure(
        harness,
        hidden_tools=hidden_tools,
        mode=mode,
        batch_duplicates=True,
    )


def _has_recent_tool_evidence(state: Any) -> bool:
    if bool(getattr(state, "artifacts", None)):
        return True
    for message in list(getattr(state, "recent_messages", []) or [])[-12:]:
        if getattr(message, "role", "") == "tool":
            return True
    return False
