from __future__ import annotations

from typing import Any

from ..state_memory import trim_recent_messages


def _coerce_bool_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _artifact_success_state(metadata: dict[str, Any]) -> bool:
    explicit = _coerce_bool_flag(metadata.get("success"))
    if explicit is not None:
        return explicit
    exit_code = metadata.get("exit_code")
    if exit_code in (None, ""):
        return True
    try:
        return int(exit_code) == 0
    except (TypeError, ValueError):
        return False


def _artifact_result_envelope(artifact: Any) -> Any:
    from ..models.tool_result import ToolEnvelope

    metadata = dict(artifact.metadata or {}) if isinstance(getattr(artifact, "metadata", None), dict) else {}
    error = str(metadata.get("error") or "").strip() or None
    status = str(metadata.get("status") or "").strip() or None
    output = metadata.get("output")
    if not isinstance(output, dict):
        output = None
    return ToolEnvelope(
        success=_artifact_success_state(metadata),
        status=status,
        output=output,
        error=error,
        metadata=metadata,
    )


def _shell_artifact_reference(artifact: Any) -> str:
    metadata = artifact.metadata if isinstance(getattr(artifact, "metadata", None), dict) else {}
    command = str(metadata.get("command") or "").strip()
    if not command:
        arguments = metadata.get("arguments")
        if isinstance(arguments, dict):
            command = str(arguments.get("command") or "").strip()
    label = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "shell_exec").strip()
    artifact_id = str(getattr(artifact, "artifact_id", "") or "").strip()
    status = "SUCCESS" if _artifact_success_state(metadata) else "FAILED"
    if command and artifact_id:
        return (
            f"Artifact {artifact_id}: {label} {status}: {command}. "
            f"Use `artifact_read(artifact_id='{artifact_id}')` for the full transcript."
        )
    summary = str(getattr(artifact, "summary", "") or getattr(artifact, "source", "") or label or "tool result").strip()
    return f"Artifact {artifact_id}: {summary}" if artifact_id else summary


def _current_tool_compaction_limits(harness: Any, *, soft_limit: int) -> tuple[int, int, bool]:
    from ..context.policy import estimate_text_tokens

    policy = getattr(harness, "context_policy", None)
    summarize_ratio = float(getattr(policy, "summarize_at_ratio", 0.8) or 0.8)
    inline_limit = int(getattr(policy, "tool_result_inline_token_limit", 325) or 325)
    effective_soft_limit = max(0, int(soft_limit or 0))
    total_recent_tokens = sum(
        estimate_text_tokens(str(message.content or ""))
        for message in getattr(getattr(harness, "state", None), "recent_messages", []) or []
    )
    prompt_pressure_threshold = int(effective_soft_limit * summarize_ratio) if effective_soft_limit > 0 else 0
    under_pressure = bool(prompt_pressure_threshold and total_recent_tokens >= prompt_pressure_threshold)
    single_message_pressure_threshold = max(
        inline_limit,
        int(effective_soft_limit * 0.5) if effective_soft_limit > 0 else inline_limit,
    )
    return inline_limit, single_message_pressure_threshold, under_pressure


def compact_oversized_tool_messages(harness: Any, *, soft_limit: int) -> bool:
    """Replace large tool-result message content with compact artifact references."""
    from ..context.policy import estimate_text_tokens

    threshold, single_message_pressure_threshold, under_pressure = _current_tool_compaction_limits(
        harness,
        soft_limit=soft_limit,
    )
    compacted_any = False
    for message in reversed(harness.state.recent_messages):
        if message.role != "tool":
            continue
        content = message.content or ""
        content_tokens = estimate_text_tokens(content)
        if content_tokens <= threshold:
            continue
        if not under_pressure and content_tokens < single_message_pressure_threshold:
            continue
        if message.name == "artifact_read":
            # Preserve raw artifact slices once read. Replacing them with a
            # compact artifact summary makes later recovery loops look like the
            # tool returned boilerplate instead of the requested content.
            continue
        artifact_id = message.metadata.get("artifact_id") if message.metadata else None
        if not isinstance(artifact_id, str) or not artifact_id:
            char_cap = threshold * 4
            if len(content) > char_cap:
                message.content = content[:char_cap] + " [truncated]"
                compacted_any = True
            continue
        artifact = harness.state.artifacts.get(artifact_id)
        if artifact is None:
            continue
        dummy_result = _artifact_result_envelope(artifact)
        compact = harness.artifact_store.compact_tool_message(
            artifact,
            dummy_result,
            request_text=harness._current_user_task(),
        )
        if str(getattr(artifact, "tool_name", "") or "").strip() in {"shell_exec", "ssh_exec"}:
            normalized_compact = " ".join(str(compact or "").strip().lower().split())
            if normalized_compact in {"ok", "exit_code=0 ok"}:
                compact = _shell_artifact_reference(artifact)
        if estimate_text_tokens(compact) < estimate_text_tokens(content):
            harness._runlog(
                "budget_policy",
                "compacted oversized tool message to artifact reference",
                artifact_id=artifact_id,
                original_tokens=content_tokens,
                compacted_tokens=estimate_text_tokens(compact),
            )
            if artifact and str(artifact.tool_name or "").strip() in {"shell_exec", "ssh_exec"}:
                exit_code = artifact.metadata.get("exit_code")
                if exit_code is not None:
                    status_tag = "EXIT_CODE=0" if exit_code == 0 else f"EXIT_CODE={exit_code} (FAILED)"
                    compact = f"{status_tag}\n{compact}"
            message.content = compact
            compacted_any = True
    return compacted_any


def trim_recent_messages_window(messages: list[Any], *, limit: int) -> list[Any]:
    return trim_recent_messages(messages, limit=limit)
