from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models.tool_result import ToolEnvelope
from ..tools.fs_sessions import _normalize_replace_strategy
from .tool_visibility import filter_tools_for_runtime_state
from .task_classifier import (
    classify_runtime_intent,
    classify_task_mode,
    looks_like_author_write_request,
    looks_like_implementation_followup,
    looks_like_write_file_request,
    looks_like_write_patch_request,
    runtime_policy_for_intent,
)
from .run_mode import (
    ensure_remote_tool_profile,
    has_active_remote_handoff,
    is_contextual_affirmative_execution_continuation,
    resolve_mode_task,
    should_enable_complex_write_chat_draft,
)
from .tool_dispatch_cache import maybe_reuse_file_read, maybe_reuse_identical_read_call, _reuse_cached_file_read

_CHAT_WRITE_TOOL_NAMES = {"file_write", "file_patch", "ast_patch"}
_READONLY_CHAT_TOOL_BLOCKLIST = {
    "shell_exec",
    "ssh_exec",
    "ssh_file_read",
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
    "finalize_write_session",
    "file_write",
    "file_patch",
    "ast_patch",
    "file_delete",
}
_CHAT_TERMINAL_TOOL_NAMES = {"task_complete", "task_fail"}


def _scratchpad(harness: Any) -> dict[str, Any]:
    scratchpad = getattr(harness.state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        harness.state.scratchpad = scratchpad
    return scratchpad


def _resolved_followup_effective_task(harness: Any, task: str) -> str:
    scratchpad = _scratchpad(harness)
    resolved = scratchpad.get("_resolved_followup")
    if not isinstance(resolved, dict):
        return str(task or "").strip()
    effective = str(resolved.get("effective_task") or "").strip()
    if not effective:
        return str(task or "").strip()
    raw = str(resolved.get("raw_task") or "").strip()
    normalized_task = str(task or "").strip()
    if not raw or normalized_task == raw or normalized_task == effective:
        return effective
    return normalized_task


def _refresh_task_mode(harness: Any, task: str) -> str:
    task_mode = classify_task_mode(task)
    harness.state.task_mode = task_mode
    return task_mode


def _current_user_task(harness: Any) -> str:
    """Return the current raw user task, falling back to run_brief if needed."""
    run_brief = getattr(getattr(harness, "state", None), "run_brief", None)
    task = ""
    if run_brief is not None:
        task = str(getattr(run_brief, "original_task", "") or "").strip()
    if not task:
        task = str(getattr(getattr(harness, "state", None), "current_task", "") or "").strip()
    return task


def _refresh_runtime_intent(harness: Any, task: str) -> tuple[str, str]:
    runtime_intent = classify_runtime_intent(
        task,
        recent_messages=getattr(harness.state, "recent_messages", []),
        pending_interrupt=getattr(harness.state, "pending_interrupt", None),
    )
    _scratchpad(harness)["_chat_runtime_intent"] = runtime_intent.label
    return runtime_intent.label, runtime_intent.task_mode


def _has_active_write_session(harness: Any) -> bool:
    session = getattr(getattr(harness, "state", None), "write_session", None)
    if session is None:
        return False
    status = str(getattr(session, "status", "") or "").strip().lower()
    return status != "complete"


def _has_ask_human_affirmative_resume(harness: Any) -> bool:
    marker = _scratchpad(harness).get("_ask_human_affirmative_resume")
    return isinstance(marker, dict) and bool(str(marker.get("original_task") or "").strip())


def _chat_write_tools_allowed(harness: Any, task: str) -> bool:
    if _has_ask_human_affirmative_resume(harness):
        return True
    if _has_active_write_session(harness):
        return True
    if (
        looks_like_write_patch_request(task)
        or looks_like_write_file_request(task)
        or looks_like_author_write_request(task)
        or looks_like_implementation_followup(task)
    ):
        return True
    client = getattr(harness, "client", None)
    return should_enable_complex_write_chat_draft(
        task,
        model_name=getattr(client, "model", None),
        cwd=getattr(getattr(harness, "state", None), "cwd", None),
    )


def _filter_chat_tools_for_task_mode(
    harness: Any,
    tools: list[dict[str, Any]],
    *,
    task: str,
    task_mode: str,
) -> list[dict[str, Any]]:
    write_tools_allowed = _chat_write_tools_allowed(harness, task)
    if task_mode == "remote_execute":
        blocked_names = {"shell_exec", "finalize_write_session"}
    elif task_mode == "local_execute":
        blocked_names = {
            "ssh_exec",
            "ssh_file_read",
            "ssh_file_write",
            "ssh_file_patch",
            "ssh_file_replace_between",
        }
    else:
        blocked_names = set(_READONLY_CHAT_TOOL_BLOCKLIST)
    if write_tools_allowed:
        blocked_names -= _CHAT_WRITE_TOOL_NAMES
    else:
        blocked_names |= _CHAT_WRITE_TOOL_NAMES
    # Repair-phase exemption: never suppress core repair tools
    if getattr(getattr(harness, "state", None), "current_phase", None) == "repair":
        blocked_names = set()
    filtered: list[dict[str, Any]] = []
    for entry in tools:
        function = entry.get("function") if isinstance(entry, dict) else None
        tool_name = str(function.get("name") or "") if isinstance(function, dict) else ""
        if tool_name in blocked_names:
            continue
        filtered.append(entry)
    _scratchpad(harness)["_chat_task_mode"] = task_mode
    return filtered


def _chat_terminal_tools(harness: Any) -> list[dict[str, Any]]:
    tools = harness.registry.export_openai_tools(
        phase=harness.state.current_phase,
        mode="chat",
        profiles=set(harness.state.active_tool_profiles),
    )
    selected = [
        entry
        for entry in tools
        if isinstance(entry, dict)
        and isinstance(entry.get("function"), dict)
        and str(entry["function"].get("name") or "") in _CHAT_TERMINAL_TOOL_NAMES
    ]
    from ..fama.tool_policy import apply_fama_tool_exposure, fama_hidden_tools_for_exposure

    hidden_tools = fama_hidden_tools_for_exposure(
        selected,
        state=harness.state,
        mode="chat",
        config=getattr(harness, "config", None),
    )
    if hidden_tools:
        _log_fama_tool_exposure(harness, hidden_tools=hidden_tools, mode="chat")
    return apply_fama_tool_exposure(
        selected,
        state=harness.state,
        mode="chat",
        config=getattr(harness, "config", None),
    )


def _task_excerpt(task: str, *, limit: int = 160) -> str:
    text = " ".join(str(task or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _log_chat_tool_selection_error(
    harness: Any,
    *,
    exc: Exception,
    task: str,
    phase: str,
) -> None:
    harness._runlog(
        "chat_tool_selection_error",
        "chat tool exposure failed",
        exception_type=type(exc).__name__,
        error=str(exc),
        phase=phase,
        mode="chat",
        task_excerpt=_task_excerpt(task),
        suppression_reason=str(_scratchpad(harness).get("_chat_tools_suppressed_reason") or ""),
        tool_profiles=list(getattr(harness.state, "active_tool_profiles", []) or []),
        runtime_intent=str(_scratchpad(harness).get("_chat_runtime_intent") or ""),
    )


def chat_mode_requires_tools(harness: Any, task: str) -> bool:
    # Repair phase must always have tools available to fix errors
    if getattr(getattr(harness, "state", None), "current_phase", None) == "repair":
        return True
    transaction = _scratchpad(harness).get("_task_transaction")
    if isinstance(transaction, dict) and str(transaction.get("turn_type") or "").strip() == "CLARIFICATION":
        return False
    _refresh_task_mode(harness, task)
    if has_active_remote_handoff(harness):
        ensure_remote_tool_profile(harness)
        return True
    # Fix for RCA 8ec35471: classify intent against the original user task,
    # not recovery nudges or system messages injected after a blocked tool.
    task_for_intent = task
    if getattr(task, "startswith", lambda x: False)("Registered but unavailable on this turn:"):
        task_for_intent = _current_user_task(harness)
    runtime_intent = classify_runtime_intent(
        task_for_intent,
        recent_messages=getattr(harness.state, "recent_messages", []),
        pending_interrupt=getattr(harness.state, "pending_interrupt", None),
    )
    _scratchpad(harness)["_chat_runtime_intent"] = runtime_intent.label
    runtime_policy = runtime_policy_for_intent(runtime_intent)
    model_name = getattr(harness.client, "model", None)
    if _has_active_write_session(harness):
        return True
    if _has_ask_human_affirmative_resume(harness):
        return True
    if should_enable_complex_write_chat_draft(
        task_for_intent,
        model_name=model_name,
        cwd=getattr(harness.state, "cwd", None),
    ):
        return True
    raw_task, resolved_task = resolve_mode_task(harness, task_for_intent)
    if is_contextual_affirmative_execution_continuation(
        harness,
        raw_task=raw_task,
        resolved_task=resolved_task,
    ):
        return True
    return runtime_policy.chat_requires_tools


def chat_mode_tools(harness: Any) -> list[dict[str, Any]]:
    task = ""
    selection_phase = "current_user_task"
    try:
        task = _resolved_followup_effective_task(harness, harness._current_user_task())
        if _has_ask_human_affirmative_resume(harness):
            marker = _scratchpad(harness).get("_ask_human_affirmative_resume")
            if isinstance(marker, dict):
                resumed_task = str(marker.get("original_task") or "").strip()
                if resumed_task:
                    task = resumed_task
        # Fix for RCA 8ec35471: do not let a blocked-tool recovery nudge become
        # the task text used for intent classification and tool filtering.
        # Intents must be derived from the user's actual request.
        task_for_intent = task
        if task.startswith("Registered but unavailable on this turn:"):
            task_for_intent = _current_user_task(harness)
        runtime_intent_label, task_mode = _refresh_runtime_intent(harness, task_for_intent)
        selection_phase = "requires_tools"
        if has_active_remote_handoff(harness):
            ensure_remote_tool_profile(harness)
            runtime_intent_label = "remote_handoff"
            task_mode = "remote_execute"
        elif runtime_intent_label == "smalltalk":
            terminal_tools = _chat_terminal_tools(harness)
            _scratchpad(harness)["_chat_tools_exposed"] = bool(terminal_tools)
            _scratchpad(harness)["_chat_tools_suppressed_reason"] = "smalltalk_terminal_only"
            harness._runlog(
                "chat_tool_selection",
                "chat tool exposure reduced to terminal tools for smalltalk",
                task=task,
                reason="smalltalk_terminal_only",
                tool_names=[
                    str(entry["function"]["name"])
                    for entry in terminal_tools
                    if isinstance(entry, dict) and isinstance(entry.get("function"), dict)
                ],
                runtime_intent=runtime_intent_label,
            )
            return terminal_tools
        if not chat_mode_requires_tools(harness, task_for_intent):
            terminal_tools = _chat_terminal_tools(harness)
            _scratchpad(harness)["_chat_tools_exposed"] = bool(terminal_tools)
            _scratchpad(harness)["_chat_tools_suppressed_reason"] = "non_lookup_chat_terminal_only"
            harness._runlog(
                "chat_tool_selection",
                "chat tool exposure reduced to terminal tools",
                task=task,
                reason="non_lookup_chat_terminal_only",
                tool_names=[
                    str(entry["function"]["name"])
                    for entry in terminal_tools
                    if isinstance(entry, dict) and isinstance(entry.get("function"), dict)
                ],
                runtime_intent=runtime_intent_label,
            )
            return terminal_tools

        _scratchpad(harness)["_chat_tools_exposed"] = True
        _scratchpad(harness).pop("_chat_tools_suppressed_reason", None)
        selection_phase = "tool_export"
        tools = harness.registry.export_openai_tools(
            phase=harness.state.current_phase,
            mode="chat",
            profiles=set(harness.state.active_tool_profiles),
        )
        selection_phase = "task_mode_filter"
        tools = _filter_chat_tools_for_task_mode(harness, tools, task=task, task_mode=task_mode)
        selection_phase = "runtime_filter"
        tools = filter_tools_for_runtime_state(
            tools,
            state=harness.state,
            mode="chat",
        )
        selection_phase = "approval_gated_shell"
        shell_spec = harness.registry.get("shell_exec")
        active_tool_names = {
            str(entry["function"]["name"])
            for entry in tools
            if isinstance(entry, dict)
            and isinstance(entry.get("function"), dict)
            and "name" in entry["function"]
        }
        if (
            task_mode == "local_execute"
            and shell_spec is not None
            and shell_spec.profile_allowed(set(harness.state.active_tool_profiles))
            and "shell_exec" not in active_tool_names
        ):
            tools.append(shell_spec.openai_schema())
            harness._runlog(
                "chat_tool_selection",
                "shell execution exposed in chat mode",
                task=task,
                reason="approval_gated_shell",
                runtime_intent=runtime_intent_label,
            )
        from ..fama.tool_policy import apply_fama_tool_exposure, fama_hidden_tools_for_exposure

        hidden_tools = fama_hidden_tools_for_exposure(
            tools,
            state=harness.state,
            mode="chat",
            config=getattr(harness, "config", None),
        )
        if hidden_tools:
            _log_fama_tool_exposure(harness, hidden_tools=hidden_tools, mode="chat")
        return apply_fama_tool_exposure(
            tools,
            state=harness.state,
            mode="chat",
            config=getattr(harness, "config", None),
        )
    except Exception as exc:
        _log_chat_tool_selection_error(
            harness,
            exc=exc,
            task=task,
            phase=selection_phase,
        )
        try:
            terminal_tools = _chat_terminal_tools(harness)
        except Exception:
            terminal_tools = []
        _scratchpad(harness)["_chat_tools_exposed"] = bool(terminal_tools)
        _scratchpad(harness)["_chat_tools_suppressed_reason"] = "chat_tool_selection_error"
        return terminal_tools


async def dispatch_tool_call(harness: Any, tool_name: str, args: dict[str, Any]) -> ToolEnvelope:
    # Track tools dispatched count (Fix 7)
    if hasattr(harness, "state") and harness.state is not None:
        scratchpad = getattr(harness.state, "scratchpad", None)
        if isinstance(scratchpad, dict):
            scratchpad["_tools_dispatched"] = int(scratchpad.get("_tools_dispatched", 0)) + 1
    # Hard block: SSH tools are never valid for local coding tasks.
    task_mode = str(getattr(harness.state, "task_mode", "") or "").strip().lower()
    if (
        tool_name in {"ssh_exec", "ssh_file_read", "ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}
        and task_mode == "local_execute"
    ):
        return ToolEnvelope(
            success=False,
            error="SSH tools are not available for local coding tasks. Use local file_write, file_read, and shell_exec only.",
            metadata={"tool_name": tool_name, "blocked_reason": "local_coding_ssh_block"},
        )

    cached = maybe_reuse_file_read(harness, tool_name=tool_name, args=args)
    if cached is not None:
        return cached
    cached = maybe_reuse_identical_read_call(harness, tool_name=tool_name, args=args)
    if cached is not None:
        return cached

    if tool_name not in harness.registry.names():
        sanitized = attempt_tool_sanitization(harness, tool_name)
        if sanitized:
            harness._runlog(
                "tool_sanitization",
                "hallucinated tool name detected and split",
                hallucinated=tool_name,
                sanitized=sanitized,
            )
            tool_name = sanitized

    from ..fama.tool_policy import enforce_fama_tool_call

    blocked = enforce_fama_tool_call(
        tool_name,
        args,
        state=harness.state,
        mode=getattr(harness.state, "run_mode", "loop"),
        config=getattr(harness, "config", None),
    )
    if blocked is not None:
        harness._runlog(
            "fama_tool_call_blocked",
            "FAMA blocked tool call",
            tool_name=tool_name,
            active_mitigation=blocked.metadata.get("active_mitigation"),
            required_fingerprints=blocked.metadata.get("required_fingerprints", []),
            actual_fingerprint=blocked.metadata.get("actual_fingerprint", ""),
            fingerprint_match=blocked.metadata.get("fingerprint_match"),
            mode=getattr(harness.state, "run_mode", "loop"),
        )
        return blocked

    patch_first = maybe_block_full_write_for_iteration(harness, tool_name=tool_name, args=args)
    if patch_first is not None:
        return patch_first

    # Terminal-state breaker: block exploratory read-only tools when terminal readiness is reached
    from ..challenge_progress import terminal_readiness_state
    if terminal_readiness_state(harness.state):
        if tool_name in {"dir_list", "file_read", "loop_status", "artifact_read", "artifact_grep", "artifact_print"}:
            return ToolEnvelope(
                success=False,
                status="blocked",
                error="Terminal readiness reached. The required artifact exists and is verified. Stop exploratory reads and call task_complete or task_fail.",
                metadata={
                    "tool_name": tool_name,
                    "reason": "terminal_readiness_breaker",
                    "active_mitigation": "terminal_state_block",
                },
            )

    # Timeout override: cap at harness limit
    timeout_override_metadata: dict[str, Any] = {}
    requested_timeout = args.get("timeout_sec")
    if requested_timeout is not None:
        harness_timeout = getattr(getattr(harness, "config", None), "graph_dispatch_tools_timeout_sec", None)
        if harness_timeout is not None and int(requested_timeout) > int(harness_timeout):
            args = dict(args)
            args["timeout_sec"] = int(harness_timeout)
            timeout_override_metadata = {
                "effective_timeout_sec": int(harness_timeout),
                "timeout_override_reason": f"capped by harness graph_dispatch_tools_timeout_sec ({harness_timeout}s)",
            }

    result = await harness.dispatcher.dispatch(tool_name, args)
    if timeout_override_metadata and isinstance(result, ToolEnvelope):
        if result.metadata is None:
            result.metadata = {}
        result.metadata.update(timeout_override_metadata)
    if getattr(result, "success", False) or (isinstance(result, dict) and result.get("success")):
        import time
        scratchpad = harness.state.scratchpad
        if "_first_tool_dispatch_complete_time" not in scratchpad:
            scratchpad["_first_tool_dispatch_complete_time"] = time.time()
    return result


def attempt_tool_sanitization(harness: Any, tool_name: str) -> str | None:
    for registered_name in harness.registry.names():
        if not registered_name:
            continue
        if tool_name.startswith(registered_name) and tool_name != registered_name:
            remainder = tool_name[len(registered_name) :]
            if remainder in harness.registry.names():
                return registered_name
    return None


def _log_fama_tool_exposure(harness: Any, *, hidden_tools: set[str], mode: str) -> None:
    from .tool_exposure_logging import log_fama_tool_exposure

    log_fama_tool_exposure(harness, hidden_tools=hidden_tools, mode=mode)


def maybe_block_full_write_for_iteration(
    harness: Any,
    *,
    tool_name: str,
    args: dict[str, Any],
) -> ToolEnvelope | None:
    if tool_name not in {"file_write", "ssh_file_write"}:
        return None
    scratchpad = _scratchpad(harness)
    transaction = scratchpad.get("_task_transaction")
    if not isinstance(transaction, dict):
        return None
    turn_type = str(transaction.get("turn_type") or "").strip()
    if turn_type not in {"ITERATION", "CORRECTION"}:
        return None
    if _full_rewrite_explicitly_requested(harness, args):
        return None
    if _has_active_write_session(harness) or str(args.get("write_session_id") or "").strip():
        return None

    path = str(args.get("path") or args.get("target_path") or "").strip()
    if not path:
        return None

    # Allow an explicit replace_strategy=overwrite to bypass the patch-first
    # guard. The tool schema already advertises this override; honouring it
    # lets models recover from a blocked full rewrite without repeating
    # failing patch attempts.
    if _normalize_replace_strategy(args.get("replace_strategy")) == "overwrite":
        return None

    suggested = "file_patch" if tool_name == "file_write" else "ssh_file_patch"
    if not _patch_tool_available(harness, suggested):
        return None
    if tool_name == "file_write":
        if not _local_target_exists(harness, path):
            return None
    elif not _remote_target_known_existing(transaction, scratchpad, path):
        return None

    note = {
        "tool_name": tool_name,
        "path": path,
        "turn_type": turn_type,
        "suggested_tool": suggested,
    }
    scratchpad["_patch_first_blocked_write"] = note
    return ToolEnvelope(
        success=False,
        status="recoverable",
        error=(
            f"Patch-first policy for {turn_type}: `{tool_name}` would rewrite existing target `{path}`. "
            f"Use `{suggested}` for the narrow edit, or explicitly request a full rewrite "
            f"with `replace_strategy='overwrite'`."
        ),
        metadata={
            "reason": "patch_first_required",
            "tool_name": tool_name,
            "path": path,
            "turn_type": turn_type,
            "suggested_tool": suggested,
            "recoverable": True,
        },
    )


def _patch_tool_available(harness: Any, tool_name: str) -> bool:
    registry = getattr(harness, "registry", None)
    names = getattr(registry, "names", None)
    if not callable(names):
        return True
    try:
        return tool_name in set(names())
    except Exception:
        return True


def _local_target_exists(harness: Any, path: str) -> bool:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.exists()
    cwd = getattr(getattr(harness, "state", None), "cwd", None)
    base = Path(cwd) if isinstance(cwd, str) and cwd else Path.cwd()
    return (base / candidate).exists()


def _remote_target_known_existing(
    transaction: dict[str, Any],
    scratchpad: dict[str, Any],
    path: str,
) -> bool:
    normalized = _normalize_path(path)
    candidates: list[Any] = []
    for key in ("allowed_paths", "remote_target_paths", "target_paths"):
        value = transaction.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    handoff = scratchpad.get("_last_task_handoff")
    if isinstance(handoff, dict):
        for key in ("remote_target_paths", "allowed_paths"):
            value = handoff.get(key)
            if isinstance(value, list):
                candidates.extend(value)
    return any(_normalize_path(item) == normalized for item in candidates)


def _normalize_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/").lower()


def _full_rewrite_explicitly_requested(harness: Any, args: dict[str, Any]) -> bool:
    strategy = str(args.get("replace_strategy") or args.get("mode") or "").strip().lower()
    if strategy in {"overwrite", "rewrite", "full", "replace_all"}:
        return True
    task = ""
    current_task = getattr(harness, "_current_user_task", None)
    if callable(current_task):
        try:
            task = str(current_task() or "")
        except Exception:
            task = ""
    text = f"{task} {args.get('instruction') or ''} {args.get('reason') or ''}".lower()
    if any(
        phrase in text
        for phrase in (
            "rewrite the whole file",
            "rewrite the entire file",
            "replace the whole file",
            "replace the entire file",
            "regenerate the file",
            "full rewrite",
            "complete rewrite",
            "overwrite",
        )
    ):
        return True
    return _task_requests_full_artifact_authoring(task, args)


def _task_requests_full_artifact_authoring(task: str, args: dict[str, Any]) -> bool:
    path = str(args.get("path") or args.get("target_path") or "").strip()
    if not path:
        return False
    normalized_path = _normalize_path(path)
    text = " ".join(str(task or "").lower().split())
    if not text or normalized_path not in text.replace("\\", "/"):
        return False

    authoring_markers = (
        "build ",
        "create ",
        "generate ",
        "produce ",
        "write ",
        "implement ",
    )
    artifact_markers = (
        "self-contained",
        "complete script",
        "complete file",
        "standalone script",
        "standalone file",
        "from scratch",
    )
    return any(marker in text for marker in authoring_markers) and any(
        marker in text for marker in artifact_markers
    )
