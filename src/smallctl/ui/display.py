"""
smallctl/ui/display.py
----------------------
Display formatting and rendering helpers for the SmallctlApp UI.

This module extracts display/formatting logic from the main app to keep
the orchestration code clean and pipeline-like.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any

from ..models.events import (
    UIEvent,
    UIEventType,
    UIStatusSnapshot,
)


_CRITICAL_EVENTS = {
    "context_invalidated",
    "context_lane_dropped",
    "file_patch_read_autocontinue",
    "write_overwrite_guard_read_autocontinue",
    "fama_health_warning",
    "fama_capsule_health",
    "fama_signal_detected",
    "fama_signal_to_mitigation",
    "fama_mitigation_activated",
    "long_running_remote_timeout_write_guard",
    "model_output_degenerate_loop_exhausted",
    "partial_tool_call_cancelled",
    "reflexion_created",
    "recovery_human_resteer_recorded",
    "same_scope_iteration_recorded",
    "ssh_host_key_recovery_required",
    "subtask_transition",
    "task_interrupted",
    "terminal_control_failed",
    "tool_blocked_not_exposed",
    "tool_dispatch_cancelled",
    "verifier_loop_detected",
    "generic_tool_loop_nudge",
    "recent_message_limit_tuned",
}

# UI visibility set for high-signal harness diagnostics that should be rendered
# in the TUI transcript even though they are normal harness events.
_UI_VISIBLE_EVENTS = {
    "mode_decision",
    "model_output_degenerate_loop_exhausted",
    "partial_tool_call_cancelled",
    "tool_blocked_not_exposed",
}

_DUPLICATE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "completed",
    "complete",
    "for",
    "found",
    "from",
    "identified",
    "in",
    "including",
    "into",
    "is",
    "of",
    "on",
    "or",
    "successfully",
    "the",
    "to",
    "with",
}

_COMMAND_OUTPUT_ERROR_TOOLS = {"shell_exec", "ssh_exec"}


def _sanitize_recent_error_for_ui(err_text: str) -> str:
    """Keep diagnostics useful without surfacing command output as system text."""
    text = str(err_text or "").strip()
    if not text:
        return ""
    lower = text.lower()
    for tool_name in _COMMAND_OUTPUT_ERROR_TOOLS:
        if re.search(rf"(?:^|[\s\['\"]){re.escape(tool_name)}\s*:", lower):
            return f"{tool_name} failed; see tool output"
    return text


def format_restore_status(status: dict[str, Any]) -> str:
    """Format the restore status dict into a user-friendly message."""
    if status.get("status") == "not_found":
        thread_id = str(status.get("thread_id") or "").strip()
        if thread_id:
            return f"No persisted graph state found for thread {thread_id}."
        return "No persisted graph state found."

    thread_id = str(status.get("thread_id") or "").strip() or "unknown"
    interrupt = status.get("interrupt")
    if isinstance(interrupt, dict):
        question = str(interrupt.get("question") or "").strip()
        if question:
            return (
                f"Restored graph state for thread {thread_id}. "
                f"Submit a reply to continue: {question}"
            )
    return f"Restored graph state for thread {thread_id}."


def should_render_run_log_row(row: dict[str, Any]) -> bool:
    """
    Determine if a run log row should be rendered in the UI.
    Keeps model/tool/chat protocol logs out of the main transcript.
    """
    channel = str(row.get("channel") or "")
    event = str(row.get("event") or "")

    # Critical backend state changes should always be visible
    if event in _CRITICAL_EVENTS or event in _UI_VISIBLE_EVENTS:
        return True
    if channel in {"tools", "chat", "model_output"}:
        return False
    if channel != "harness":
        return False
    if event in {
        "chunk",
        "model_token",
        "model_output",
        "model_thinking",
        "harness_tool_dispatch",
        "harness_tool_result",
        "tool_replay_hit",
    }:
        return False
    return True


def should_render_event(event: UIEvent, *, show_system_messages: bool, show_tool_calls: bool) -> bool:
    """Determine if an event should be rendered based on user preferences."""
    if event.data.get("hidden_from_ui") or event.data.get("ui_hidden"):
        return False
    if event.event_type == UIEventType.STATUS:
        return False
    if event.data.get("ui_kind") == "subtask_checklist":
        return True
    # Critical backend state changes should always be visible
    if event.data.get("ui_kind") in _CRITICAL_EVENTS or event.data.get("event") in _CRITICAL_EVENTS:
        return True
    if event.event_type in {UIEventType.SYSTEM, UIEventType.METRICS} and not show_system_messages:
        return False
    # Treat most ALERTs as system-level noise, but preserve interactive prompts
    # (plan approvals/interrupts) that require user action or record a question.
    if event.event_type == UIEventType.ALERT and not show_system_messages:
        if event.data.get("ui_kind") in {"approve_prompt", "sudo_password_prompt"}:
            return True
        if "interrupt" in event.data:
            return True
        return False
    if event.event_type == UIEventType.SHELL_STREAM and not show_tool_calls:
        return False
    if event.event_type in {UIEventType.TOOL_CALL, UIEventType.TOOL_RESULT} and not show_tool_calls:
        return False
    return True


def format_run_log_row(row: dict[str, Any]) -> str:
    """Format a run log row for display."""
    event = str(row.get("event") or "").strip()
    data = row.get("data") or {}
    if event == "verifier_path_false_negative_guard":
        target = str(data.get("target") or "")
        command = str(data.get("command") or "")
        return f"[harness] ⚠️ verifier path-failure overridden: {target} ({command})"
    if event == "timeout_override":
        requested = data.get("requested_timeout_sec")
        effective = data.get("effective_timeout_sec")
        reason = str(data.get("reason") or "")
        return f"[harness] ⏱️ timeout capped: {requested}s → {effective}s ({reason})"
    if event == "fama_capsule_health_warning":
        warning = str(data.get("warning") or "")
        return f"[harness] 🚨 FAMA health: {warning}"
    if event == "fama_signal_detected":
        kind = str(data.get("kind") or "").strip()
        failure_class = str(data.get("failure_class") or "").strip()
        tool_name = str(data.get("tool_name") or "").strip()
        bits = [bit for bit in (kind, failure_class, tool_name) if bit]
        return f"[harness] FAMA signal: {' | '.join(bits) if bits else 'detected'}"
    if event == "fama_signal_to_mitigation":
        signal_kind = str(data.get("signal_kind") or "").strip()
        activated = data.get("activated_mitigations")
        if isinstance(activated, list):
            activated_text = ", ".join(str(item) for item in activated if str(item).strip())
        else:
            activated_text = ""
        suffix = f" -> {activated_text}" if activated_text else ""
        return f"[harness] FAMA routed: {signal_kind or 'signal'}{suffix}"
    if event == "fama_mitigation_activated":
        mitigation = str(data.get("mitigation") or "").strip()
        reason = str(data.get("reason") or "").strip()
        suffix = f" ({reason})" if reason else ""
        return f"[harness] FAMA mitigation active: {mitigation or 'mitigation'}{suffix}"
    if event == "verifier_loop_detected":
        rejection_count = data.get("rejection_count", "?")
        return f"[harness] ⚠️ Verifier loop detected ({rejection_count} rejections)"
    if event == "file_patch_read_autocontinue":
        target = str(data.get("target_path") or "").strip()
        error_kind = str(data.get("error_kind") or "patch mismatch").strip()
        suffix = f" for {target}" if target else ""
        return f"[harness] Patch recovery: auto-reading current file{suffix} after {error_kind}"
    if event == "write_overwrite_guard_read_autocontinue":
        target = str(data.get("target_path") or "").strip()
        session_id = str(data.get("session_id") or "").strip()
        target_text = f" for {target}" if target else ""
        session_text = f" (session {session_id})" if session_id else ""
        return f"[harness] Write recovery: auto-reading staged content{target_text}{session_text}; next edit should be file_patch/ast_patch or same-section repair"
    if event == "context_lane_dropped":
        lane = str(data.get("lane") or data.get("context_lane") or "context").strip()
        reason = str(data.get("reason") or data.get("drop_reason") or "stale or over budget").strip()
        return f"[harness] Context refreshed: dropped {lane} ({reason})"
    if event == "recovery_human_resteer_recorded":
        turn_type = str(data.get("turn_type") or data.get("reason") or "follow-up").strip()
        return f"[harness] Follow-up classified for recovery: {turn_type}"
    if event == "same_scope_iteration_recorded":
        turn_type = str(data.get("turn_type") or "follow-up").strip()
        return f"[harness] Same-scope follow-up recorded: {turn_type}"
    if event == "reflexion_created":
        failure = data.get("failure")
        failure_class = ""
        if isinstance(failure, dict):
            failure_class = str(failure.get("failure_class") or failure.get("kind") or "").strip()
        suffix = f": {failure_class}" if failure_class else ""
        return f"[harness] Recovery memory created{suffix}"
    if event == "task_interrupted":
        result = data.get("result")
        result_reason = str(result.get("reason") or "").strip() if isinstance(result, dict) else ""
        reason = str(data.get("reason") or result_reason).strip()
        return f"[harness] Task interrupted{': ' + reason if reason else ''}"
    if event == "generic_tool_loop_nudge":
        tool_name = str(data.get("tool_name") or "")
        return f"[harness] ⚠️ Loop guard nudge: {tool_name}"
    if event == "ssh_host_key_recovery_required":
        host = str(data.get("host") or "").strip()
        command = str(data.get("suggested_command") or "").strip()
        host_text = f" for {host}" if host else ""
        command_text = f" `{command}`" if command else ""
        return f"[harness] ⚠️ SSH host key changed{host_text}. Approve{command_text} or fix known_hosts manually."
    if event == "recent_message_limit_tuned":
        adjusted = data.get("adjusted_limit")
        reasons = data.get("reasons", [])
        reason_str = ", ".join(str(r) for r in reasons) if reasons else "pressure"
        return f"[harness] 📉 Message window reduced to {adjusted} ({reason_str})"
    if event == "context_invalidated":
        details = data.get("details") or {}
        reason = str(data.get("reason") or details.get("reason") or "").strip()
        if reason == "phase_advanced":
            from_phase = str(details.get("from_phase") or "?").strip()
            to_phase = str(details.get("to_phase") or "?").strip()
            return f"[harness] Phase changed: {from_phase} -> {to_phase}"
        artifact_count = int(data.get("invalidated_artifact_count", 0) or 0)
        observation_count = int(data.get("invalidated_observation_count", 0) or 0)
        summary_count = int(data.get("invalidated_summary_count", 0) or 0)
        return (
            f"[harness] Context invalidated: artifacts={artifact_count}, "
            f"observations={observation_count}, summaries={summary_count}"
        )
    if event == "subtask_transition":
        subtask_id = str(data.get("subtask_id") or "").strip()
        title = str(data.get("title") or "").strip()
        old_status = str(data.get("old_status") or "").strip()
        new_status = str(data.get("new_status") or "").strip()
        bits = [bit for bit in (f"{subtask_id}", title, f"{old_status}->{new_status}") if bit]
        return f"[harness] Subtask update: {' | '.join(bits)}"
    if event == "mode_decision":
        mode = str(data.get("mode") or data.get("normalized") or "").strip()
        raw = str(data.get("raw") or "").strip()
        reason = raw or "model fallback"
        return f"[harness] Mode: {mode} ({reason})"
    if event == "tool_blocked_not_exposed":
        tool_name = str(data.get("tool_name") or "").strip()
        allowed = data.get("allowed_tools") or []
        allowed_text = ", ".join(str(n) for n in allowed[:4]) or "none"
        return f"[harness] Blocked: `{tool_name}` not exposed this turn (allowed: {allowed_text})"
    if event == "tool_dispatch_cancelled":
        tool_name = str(data.get("tool_name") or "tool").strip()
        elapsed = data.get("elapsed_sec")
        suffix = f" after {elapsed}s" if elapsed not in (None, "") else ""
        return f"[harness] Tool dispatch cancelled: {tool_name}{suffix}"
    if event == "model_output_degenerate_loop_exhausted":
        details = data.get("details") or {}
        phrase = str(details.get("repeated_phrase") or data.get("repeated_phrase") or "").strip()
        suffix = f" repeating `{phrase}`" if phrase else ""
        return f"[harness] Model output loop detected{suffix}; recovery nudge injected"
    if event == "partial_tool_call_cancelled":
        tool_name = str(data.get("tool_name") or "tool").strip()
        argument_chars = data.get("argument_chars")
        suffix = f" ({argument_chars} argument chars received)" if argument_chars not in (None, "") else ""
        return f"[harness] Partial tool call cancelled before dispatch: {tool_name}{suffix}"
    msg = row.get("message") or ""
    if len(msg) > 1024:
        msg = msg[:1024] + "... [truncated]"
    return f"[{row.get('channel')}] {event}: {msg}"


def format_recovery_banner(event: str, data: dict[str, Any]) -> str:
    """Build a compact status-bar banner for recovery/guard state."""
    if event == "upstream_install_source_invalid_pivot":
        return "Blocked: installer source invalid or unavailable"
    if event == "verifier_loop_detected":
        return f"Recovery: verifier loop ({data.get('rejection_count', '?')} rejects)"
    if event == "generic_tool_loop_nudge":
        tool_name = str(data.get("tool_name") or "tool").strip()
        return f"Recovery: loop guard nudged {tool_name}"
    if event == "remote_tool_guard_nudge":
        tool_name = str(data.get("tool_name") or "tool").strip()
        return f"Blocked: local tool {tool_name} blocked for remote task. Clarify if you want local operations."
    if event == "remote_tool_guard_nudge":
        tool_name = str(data.get("tool_name") or "tool").strip()
        return f"Blocked: local tool {tool_name} blocked for remote task. Clarify if you want local operations."
    if event == "recent_message_limit_tuned":
        adjusted = data.get("adjusted_limit")
        return f"Recovery: message window reduced to {adjusted}"
    if event == "ssh_host_key_recovery_required":
        host = str(data.get("host") or "remote host").strip()
        command = str(data.get("suggested_command") or "ssh-keygen -R <host> -f ~/.ssh/known_hosts").strip()
        return f"Blocked: SSH host key changed for {host}. Approve `{command}` or fix known_hosts manually."
    if event == "long_running_remote_timeout_write_guard":
        tool_name = str(data.get("tool_name") or "file write").strip()
        return f"Blocked: {tool_name} blocked after remote installer timeout. Retry with larger timeout_sec or detached execution."
    interrupt = data.get("interrupt") if isinstance(data, dict) else None
    if isinstance(interrupt, dict) and str(interrupt.get("kind") or "").strip() == "apt_deb822_validator_approval":
        host = str(interrupt.get("host") or "").strip()
        target = host or "localhost"
        return f"Recovery: apt deb822 validation awaiting approval for {target}"
    if str(data.get("ui_kind") or "").strip() == "apt_deb822_validator_approval":
        return "Recovery: apt deb822 validation awaiting approval"
    return ""


def compute_activity_for_event(
    event: UIEvent,
    *,
    active_task_done: bool | None = None,
) -> str | None:
    """
    Compute the activity status text for a given UI event.
    Returns None if no activity update is needed.
    """
    if event.data.get("kind") == "test_time_scaling":
        phase = str(event.data.get("phase") or "").strip()
        candidate_count = event.data.get("candidate_count")
        selected = event.data.get("selected_candidate")
        score = event.data.get("selected_score")
        if phase in {"proposal_start", "branch_start"}:
            policy = str(event.data.get("policy") or "").strip()
            return f"scaling {policy or 'candidates'}..."
        if selected is not None:
            detail = f"scaling selected #{selected}"
            if candidate_count is not None:
                detail += f"/{candidate_count}"
            if score is not None:
                detail += f" score {score}"
            return detail
    # Check for explicit status_activity in data
    if "status_activity" in event.data:
        return str(event.data.get("status_activity") or "").strip()

    if event.event_type == UIEventType.TOOL_CALL:
        tool_name = str(event.content or event.data.get("tool_name") or "").strip()
        return f"running {tool_name}..." if tool_name else "running tool..."

    if event.event_type == UIEventType.TOOL_RESULT:
        if active_task_done is False:
            return "thinking..."
        return ""

    if event.event_type in {UIEventType.ASSISTANT, UIEventType.THINKING}:
        return "responding..."

    if event.event_type == UIEventType.ERROR:
        return ""

    return None


def format_test_time_scaling_event(event: UIEvent) -> str:
    """Format a test-time scaling event as a compact candidate detail panel."""
    content = str(event.content or event.data.get("status_activity") or "").strip()
    lines: list[str] = [content] if content else ["Test-time scaling update."]
    policy = str(event.data.get("policy") or "").strip()
    phase = str(event.data.get("phase") or "").strip()
    detail_bits = []
    if policy:
        detail_bits.append(f"policy: {policy}")
    if phase:
        detail_bits.append(f"phase: {phase}")
    if event.data.get("candidate_count") is not None:
        detail_bits.append(f"candidates: {event.data.get('candidate_count')}")
    if event.data.get("read_only_candidate_count") is not None:
        detail_bits.append(f"read-only: {event.data.get('read_only_candidate_count')}")
    if event.data.get("read_only_branch_parallel_count") is not None:
        detail_bits.append(f"parallel read-only branches: {event.data.get('read_only_branch_parallel_count')}")
    if event.data.get("all_failed_action"):
        detail_bits.append(f"all-fail action: {event.data.get('all_failed_action')}")
    if detail_bits:
        lines.append(" | ".join(detail_bits))

    history = event.data.get("candidate_history")
    if isinstance(history, list) and history:
        lines.append("")
        lines.append("Candidates:")
        for raw_candidate in history:
            if not isinstance(raw_candidate, dict):
                continue
            lines.extend(_format_test_time_scaling_candidate(raw_candidate))
    return "\n".join(lines)


def _format_test_time_scaling_candidate(candidate: dict[str, Any]) -> list[str]:
    idx = candidate.get("candidate")
    label = f"#{idx}" if idx is not None else "#?"
    if candidate.get("selected"):
        label += " selected"
    bits = [label]
    if candidate.get("score") is not None:
        bits.append(f"score {candidate.get('score')}")
    if candidate.get("token_cost") not in (None, 0, ""):
        bits.append(f"{candidate.get('token_cost')} tokens")
    if candidate.get("latency_ms") not in (None, 0, ""):
        bits.append(f"{candidate.get('latency_ms')} ms")
    if candidate.get("read_only"):
        bits.append("read-only")
    if candidate.get("isolated"):
        bits.append("isolated")
    lines = ["- " + " | ".join(bits)]
    tools = candidate.get("tools")
    if isinstance(tools, list) and tools:
        lines.append("  tools: " + ", ".join(str(tool) for tool in tools if str(tool).strip()))
    prompt_variant = str(candidate.get("prompt_variant") or "").strip()
    if prompt_variant:
        lines.append(f"  variant: {prompt_variant}")
    unsafe_reason = str(candidate.get("unsafe_reason") or "").strip()
    failed = candidate.get("failed_criteria")
    failed_items = [str(item) for item in failed if str(item).strip()] if isinstance(failed, list) else []
    if unsafe_reason and unsafe_reason not in failed_items:
        failed_items.append(unsafe_reason)
    if failed_items:
        lines.append("  failed: " + ", ".join(failed_items))
    return lines


def format_tool_call_for_display(tool_name: str, args: dict[str, Any]) -> str:
    """Format a tool call for display in the UI."""
    display_text = f"**{tool_name}**"
    if args:
        # Format args as compact key=value pairs
        arg_parts = []
        for k, v in args.items():
            if isinstance(v, str):
                # Truncate long strings
                if len(v) > 50:
                    v = v[:47] + "..."
                arg_parts.append(f"{k}={v!r}")
            else:
                arg_parts.append(f"{k}={v}")
        if arg_parts:
            display_text += f"({', '.join(arg_parts)})"
    return display_text


def format_tool_result_for_display(
    result: Any,
    *,
    max_length: int = 200,
) -> str:
    """Format a tool result for display in the UI."""
    if result is None:
        return "(no result)"

    if isinstance(result, dict):
        # Check for summary or message fields
        text = result.get("summary") or result.get("message") or result.get("output")
        if text:
            text = str(text)
        else:
            # Compact JSON representation
            import json
            text = json.dumps(result, default=str)
    else:
        text = str(result)

    if len(text) > max_length:
        text = text[: max_length - 3] + "..."

    return text


def should_promote_tool_args_to_assistant(tool_name: str, args: dict[str, Any]) -> str | None:
    """
    Check if tool arguments should be promoted to assistant text.
    Returns the text to promote, or None if no promotion should happen.
    """
    if tool_name == "task_complete":
        promote_text = str(args.get("message") or "").strip()
        if promote_text:
            return promote_text
    elif tool_name == "ask_human":
        promote_text = str(args.get("question") or "").strip()
        if promote_text:
            return promote_text
    return None


def check_duplicate_promotion(
    promote_text: str,
    active_assistant_text: str,
) -> bool:
    """
    Check if the promotion text would duplicate existing assistant text.
    Returns True if promotion should be skipped as a duplicate.
    """
    if not active_assistant_text:
        return False

    promote_norm = _normalize_duplicate_text(promote_text)
    active_norm = _normalize_duplicate_text(active_assistant_text)
    if not promote_norm or not active_norm:
        return False

    if promote_norm in active_norm or active_norm in promote_norm:
        return True

    similarity = SequenceMatcher(None, promote_norm, active_norm).ratio()
    if similarity >= 0.72:
        return True

    promote_tokens = set(_salient_duplicate_tokens(promote_text))
    active_tokens = set(_salient_duplicate_tokens(active_assistant_text))
    if not promote_tokens or not active_tokens:
        return False

    shared = promote_tokens & active_tokens
    overlap = len(shared) / min(len(promote_tokens), len(active_tokens))
    if len(shared) >= 4 and overlap >= 0.7:
        return True

    # For short texts (≤10 salient tokens), be more aggressive:
    # semantically equivalent smalltalk often shares most tokens but
    # falls just below the default 0.7 overlap threshold.
    min_tokens = min(len(promote_tokens), len(active_tokens))
    if min_tokens <= 10 and len(shared) >= 4 and overlap >= 0.5:
        return True

    return False


def _normalize_duplicate_text(text: str) -> str:
    return " ".join(_raw_duplicate_tokens(text))


def _salient_duplicate_tokens(text: str) -> list[str]:
    salient: list[str] = []
    for token in _raw_duplicate_tokens(text):
        if token in _DUPLICATE_STOPWORDS:
            continue
        if _is_standalone_ip(token):
            continue
        salient.append(token)
    return salient


def _raw_duplicate_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9./:+_-]+", str(text or "").lower()):
        normalized = token.strip(".,;!?()[]{}")
        if normalized:
            tokens.append(normalized)
    return tokens


def _is_standalone_ip(token: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", token))


def _build_backend_rca_strip(harness: Any) -> str:
    """Build a compact backend RCA strip for cancelled/interrupted runs."""
    if harness is None:
        return ""
    state = getattr(harness, "state", None)
    if state is None:
        return ""
    parts: list[str] = []
    if str(getattr(state, "current_phase", "") or "").strip().lower() == "repair":
        parts.append("Phase: repair")

    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        blocker = scratchpad.get("_install_source_invalid_blocker")
        if isinstance(blocker, dict):
            host = str(blocker.get("host") or "installer source").strip() or "installer source"
            parts.append(f"Blocked: invalid upstream install source ({host})")

    records = getattr(state, "tool_execution_records", None)
    if isinstance(records, dict):
        for record in reversed(list(records.values())):
            if not isinstance(record, dict):
                continue
            result = record.get("result")
            if not isinstance(result, dict):
                continue
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if str(metadata.get("reason") or "").strip() != "tool_dispatch_cancelled":
                continue
            tool_name = str(record.get("tool_name") or metadata.get("tool_name") or "tool").strip()
            source = str(metadata.get("cancellation_source") or "cancel_requested").strip()
            elapsed = metadata.get("elapsed_sec")
            detail = f"Cancelled while waiting on `{tool_name}`"
            if isinstance(elapsed, (int, float)):
                detail += f" after {float(elapsed):.1f}s"
            detail += f" ({source})"
            parts.append(detail)
            break

    # Last failing verifier
    last_verifier = getattr(state, "last_verifier_verdict", None)
    if isinstance(last_verifier, dict):
        verdict = str(last_verifier.get("verdict") or "").strip().lower()
        cmd = str(last_verifier.get("command") or "").strip()
        latest_blocker = last_verifier.get("latest_blocker")
        if isinstance(latest_blocker, dict):
            salient = str(latest_blocker.get("salient_error") or "").strip()
            if salient:
                parts.append(f"Primary blocker: {salient[:160]}")
        if verdict == "fail" and cmd:
            parts.append(f"Last failing verifier: `{cmd}`")
        elif verdict == "pass" and cmd:
            parts.append(f"Last passing verifier: `{cmd}`")
        if bool(last_verifier.get("insufficient_verifier")):
            parts.append("Verifier insufficient for objective")

    # Last 3 critical tool failures
    critical_failures: list[str] = []
    recent_errors = getattr(state, "recent_errors", None)
    if isinstance(recent_errors, list):
        for err in recent_errors[-3:]:
            err_text = _sanitize_recent_error_for_ui(str(err or ""))
            if err_text and len(err_text) <= 120:
                critical_failures.append(err_text)
            elif err_text:
                critical_failures.append(err_text[:117] + "...")
    if critical_failures:
        parts.append("Recent failures: " + "; ".join(critical_failures))

    # FAMA health
    if isinstance(scratchpad, dict):
        invalidations = scratchpad.get("_context_invalidations")
        if isinstance(invalidations, list) and invalidations:
            latest = invalidations[-1]
            if isinstance(latest, dict):
                artifact_count = int(latest.get("invalidated_artifact_count", 0) or 0)
                observation_count = int(latest.get("invalidated_observation_count", 0) or 0)
                summary_count = int(latest.get("invalidated_summary_count", 0) or 0)
                if artifact_count or observation_count or summary_count:
                    parts.append(
                        "Context invalidated: "
                        f"artifacts={artifact_count}, observations={observation_count}, summaries={summary_count}"
                    )
        fama = scratchpad.get("_fama")
        if isinstance(fama, dict):
            signals = fama.get("signals")
            if isinstance(signals, list) and signals:
                parts.append(f"FAMA signals: {len(signals)}")
            else:
                parts.append("FAMA: no mitigations rendered recently")

        # Hidden artifact IDs
        hidden = scratchpad.get("suppressed_truncated_artifact_ids")
        if isinstance(hidden, list) and hidden:
            parts.append(f"Hidden artifacts: {', '.join(str(a) for a in hidden[-3:])}")

    if not parts:
        return ""
    return "RCA: " + " | ".join(parts)


class StatusState:
    """
    Immutable state object for status bar values.
    Encapsulates all status information to simplify refresh logic.
    """

    def __init__(
        self,
        *,
        model: str = "n/a",
        phase: str = "explore",
        step: int | str = 0,
        mode: str = "execution",
        plan: str = "",
        active_step: str = "",
        activity: str = "",
        contract_flow_ui: bool = False,
        contract_phase: str = "",
        acceptance_progress: str = "",
        latest_verdict: str = "",
        token_usage: int = 0,
        token_total: int = 0,
        token_limit: int = 0,
        context_window: int = 0,
        api_errors: int = 0,
        recovery_banner: str = "",
    ) -> None:
        self.model = model
        self.phase = phase
        self.step = step
        self.mode = mode
        self.plan = plan
        self.active_step = active_step
        self.activity = activity
        self.contract_flow_ui = contract_flow_ui
        self.contract_phase = contract_phase
        self.acceptance_progress = acceptance_progress
        self.latest_verdict = latest_verdict
        self.token_usage = token_usage
        self.token_total = token_total
        self.token_limit = token_limit
        self.context_window = context_window
        self.api_errors = api_errors
        self.recovery_banner = recovery_banner

    @classmethod
    def from_harness(
        cls,
        harness: Any,
        config: Any,
        *,
        activity: str = "",
        api_errors: int = 0,
        active_task: Any = None,
    ) -> "StatusState":
        snapshot_activity = activity
        if not snapshot_activity and active_task is not None and not active_task.done():
            snapshot_activity = "thinking..."
        snapshot = UIStatusSnapshot.from_harness(
            harness,
            config,
            activity=snapshot_activity,
            api_errors=api_errors,
        )
        return cls.from_snapshot(snapshot)

    @classmethod
    def from_snapshot(cls, snapshot: UIStatusSnapshot | dict[str, Any]) -> "StatusState":
        if isinstance(snapshot, UIStatusSnapshot):
            payload = snapshot.to_dict()
        else:
            payload = dict(snapshot)
        return cls(
            model=str(payload.get("model", "n/a") or "n/a"),
            phase=str(payload.get("phase", "explore") or "explore"),
            step=payload.get("step", 0),
            mode=str(payload.get("mode", "execution") or "execution"),
            plan=str(payload.get("plan", "") or ""),
            active_step=str(payload.get("active_step", "") or ""),
            activity=str(payload.get("activity", "") or ""),
            contract_flow_ui=bool(payload.get("contract_flow_ui", False)),
            contract_phase=str(payload.get("contract_phase", "") or ""),
            acceptance_progress=str(payload.get("acceptance_progress", "") or ""),
            latest_verdict=str(payload.get("latest_verdict", "") or ""),
            token_usage=max(0, int(payload.get("token_usage", 0) or 0)),
            token_total=max(0, int(payload.get("token_total", 0) or 0)),
            token_limit=max(0, int(payload.get("token_limit", 0) or 0)),
            context_window=max(0, int(payload.get("context_window", 0) or 0)),
            api_errors=max(0, int(payload.get("api_errors", 0) or 0)),
            recovery_banner=str(payload.get("recovery_banner", "") or ""),
        )
