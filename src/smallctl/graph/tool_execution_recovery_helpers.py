from __future__ import annotations

import logging
from typing import Any

from ..docker_retry_normalization import docker_failure_is_registry_resolution
from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..state import json_safe_value
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord


def _build_repair_recovery_message(harness: Any, record: ToolExecutionRecord) -> str:
    tool_name = record.tool_name
    failure_class = str(getattr(harness.state, "last_failure_class", "") or "").strip()
    repair_cycle_id = str(getattr(harness.state, "repair_cycle_id", "") or "").strip()
    path = str(record.args.get("path") or record.args.get("command") or record.args.get("host") or "").strip()
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    counters = getattr(harness.state, "stagnation_counters", {}) or {}
    counter_bits = ", ".join(
        f"{name}={value}"
        for name, value in sorted(counters.items())
        if int(value or 0) > 0
    )
    lead = "Repair loop stalled."
    if failure_class:
        lead = f"Repair loop stalled on {failure_class} failures."
    bits = [lead]
    if repair_cycle_id:
        bits.append(f"system repair cycle {repair_cycle_id} (diagnostic only)")
    if counter_bits:
        bits.append(f"stagnation: {counter_bits}")
    if path:
        bits.append(f"focus target: {path}")
    if tool_name in {"file_write", "file_append", "file_patch", "ast_patch", "file_delete"}:
        if tool_name in {"file_patch", "ast_patch"}:
            source_path = str(metadata.get("source_path") or "").strip()
            if str(metadata.get("staged_only") or "").lower() in {"true", "1"} and source_path:
                bits.append(f"staged copy: {source_path}")
            ambiguity_hint = str(metadata.get("ambiguity_hint") or "").strip()
            if ambiguity_hint:
                bits.append(ambiguity_hint)
                bits.append(
                    "If exact text keeps missing but the edit target is a function, class, import, call, or field, prefer `ast_patch` with a narrower structural locator."
                )
            elif str(metadata.get("error_kind") or "") == "patch_occurrence_mismatch":
                actual = metadata.get("actual_occurrences")
                expected = metadata.get("expected_occurrences")
                if actual is not None and expected is not None:
                    bits.append(
                        f"patch ambiguity: target matched {actual} time(s), expected {expected}. "
                        "Read a smaller slice and make `target_text` more specific before retrying."
                    )
                    bits.append(
                        "If exact text keeps missing but the edit target is a function, class, import, call, or field, prefer `ast_patch` with a narrower structural locator."
                    )
            elif tool_name == "ast_patch" and str(metadata.get("error_kind") or "") in {"ast_target_not_found", "ast_target_ambiguous"}:
                bits.append(
                    "If exact text keeps missing but the edit target is a function, class, import, call, or field, prefer `ast_patch` with a narrower structural locator."
                )
        return (
            " | ".join(bits)
            + ". Do not broad-rewrite the file. Read the current file or failing evidence first, "
            "then make one narrow patch or one narrow verification step."
        )
    if tool_name in {"shell_exec", "ssh_exec"}:
        return (
            " | ".join(bits)
            + ". Do not repeat the same command blindly. Read the smallest relevant evidence first, "
            "classify the failure, then run one narrow check or one narrow patch."
        )
    return (
        " | ".join(bits)
        + ". Do not repeat the same recovery shape; read the smallest relevant evidence first, "
        "classify the failure, then make one narrow change."
    )


def _build_docker_registry_recovery_message(harness: Any, record: ToolExecutionRecord) -> str:
    verdict = getattr(harness.state, "last_verifier_verdict", {}) or {}
    failure_class = str(getattr(harness.state, "last_failure_class", "") or "").strip()
    repair_cycle_id = str(getattr(harness.state, "repair_cycle_id", "") or "").strip()
    docker_retry_count = int(verdict.get("docker_retry_count", 0) or 0)
    image_ref = str(verdict.get("docker_image_ref") or "").strip()
    command_kind = str(verdict.get("docker_command_kind") or "").strip()
    family_count = int(verdict.get("docker_retry_family_count", 0) or 0)

    lead = "Repair loop stalled on Docker registry/image resolution failures."
    if docker_retry_count >= 4 or family_count >= 4:
        lead = "Docker image retry path exhausted for this image reference."

    bits = [lead]
    if repair_cycle_id:
        bits.append(f"system repair cycle {repair_cycle_id} (diagnostic only)")
    if image_ref:
        bits.append(f"image ref: {image_ref}")
    if failure_class:
        bits.append(f"failure class: {failure_class}")
    if command_kind:
        bits.append(f"command kind: {command_kind}")
    if docker_retry_count > 0:
        bits.append(f"equivalent failures: {docker_retry_count}")

    return (
        " | ".join(bits)
        + ". Retrying the same image ref will not help. Verify the exact repo/tag in trusted docs or web search first, "
        "check `docker image ls` on the remote host for already-present alternatives, then either use a different valid image ref, "
        "reuse an existing local image, or switch to a different package."
    )


def _maybe_emit_repair_recovery_nudge(
    harness: Any,
    record: ToolExecutionRecord,
    deps: Any,
) -> bool:
    del deps
    verdict = getattr(harness.state, "last_verifier_verdict", {}) or {}
    failure_class = str(getattr(harness.state, "last_failure_class", "") or "").strip()
    docker_retry_key = str(verdict.get("docker_retry_key") or "").strip()
    docker_retry_count = int(verdict.get("docker_retry_count", 0) or 0)
    docker_retry_family_count = int(verdict.get("docker_retry_family_count", 0) or 0)
    docker_tier = ""
    if (
        record.tool_name in {"shell_exec", "ssh_exec"}
        and not record.result.success
        and docker_failure_is_registry_resolution(failure_class)
        and docker_retry_key
        and docker_retry_count >= 2
    ):
        docker_tier = "repeat"
        if max(docker_retry_count, docker_retry_family_count) >= 4:
            docker_tier = "exhausted"
        failure_signature = f"docker_registry::{docker_tier}::{docker_retry_key}"
        if harness.state.scratchpad.get("_repair_recovery_nudged") == failure_signature:
            return False
        harness.state.scratchpad["_repair_recovery_nudged"] = failure_signature
        message = _build_docker_registry_recovery_message(harness, record)
        harness.state.append_message(
            ConversationMessage(
                role="user",
                content=message,
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "docker_registry_repair",
                    "tool_name": record.tool_name,
                    "tool_call_id": record.tool_call_id,
                    "failure_class": failure_class,
                    "system_repair_cycle_id": getattr(harness.state, "repair_cycle_id", ""),
                    "docker_retry_key": docker_retry_key,
                    "docker_retry_count": docker_retry_count,
                    "docker_retry_family_count": docker_retry_family_count,
                },
            )
        )
        harness._runlog(
            "docker_registry_repair_recovery",
            "injected Docker-specific repair recovery nudge",
            tool_name=record.tool_name,
            tool_call_id=record.tool_call_id,
            failure_class=failure_class,
            system_repair_cycle_id=getattr(harness.state, "repair_cycle_id", ""),
            docker_retry_key=docker_retry_key,
            docker_retry_count=docker_retry_count,
            docker_retry_family_count=docker_retry_family_count,
        )
        return True

    counters = getattr(harness.state, "stagnation_counters", {}) or {}
    repeat_patch = int(counters.get("repeat_patch", 0) or 0)
    no_progress = int(counters.get("no_progress", 0) or 0)
    repeat_command = int(counters.get("repeat_command", 0) or 0)
    if max(repeat_patch, no_progress, repeat_command) < 2:
        return False
    if record.result.success or getattr(record.result, "status", None) == "needs_human":
        return False

    failure_signature = "|".join(
        [
            str(record.tool_name),
            str(record.args.get("path") or ""),
            str(record.args.get("command") or ""),
            str(getattr(harness.state, "last_failure_class", "") or ""),
            str(record.result.error or ""),
            str(max(repeat_patch, no_progress, repeat_command)),
        ]
    )
    if harness.state.scratchpad.get("_repair_recovery_nudged") == failure_signature:
        return False
    harness.state.scratchpad["_repair_recovery_nudged"] = failure_signature

    message = _build_repair_recovery_message(harness, record)
    harness.state.append_message(
        ConversationMessage(
            role="user",
            content=message,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "repair_stall",
                "tool_name": record.tool_name,
                "tool_call_id": record.tool_call_id,
                "failure_class": getattr(harness.state, "last_failure_class", ""),
                "system_repair_cycle_id": getattr(harness.state, "repair_cycle_id", ""),
            },
        )
    )
    harness._runlog(
        "repair_stall_recovery",
        "injected repair recovery nudge after repeated failures",
        tool_name=record.tool_name,
        tool_call_id=record.tool_call_id,
        failure_class=getattr(harness.state, "last_failure_class", ""),
        system_repair_cycle_id=getattr(harness.state, "repair_cycle_id", ""),
        stagnation_counters=json_safe_value(counters),
    )
    return True


def _maybe_schedule_repair_loop_status_autocontinue(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name not in {"shell_exec", "ssh_exec"}:
        return False
    if record.result.success:
        return False
    if getattr(record.result, "status", None) == "needs_human":
        return False
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    if str(metadata.get("status") or "").strip().lower() == "needs_human":
        return False

    counters = getattr(harness.state, "stagnation_counters", {}) or {}
    repeat_patch = int(counters.get("repeat_patch", 0) or 0)
    no_progress = int(counters.get("no_progress", 0) or 0)
    repeat_command = int(counters.get("repeat_command", 0) or 0)
    stall_level = max(repeat_patch, no_progress, repeat_command)
    verdict = getattr(harness.state, "last_verifier_verdict", {}) or {}
    docker_retry_count = int(verdict.get("docker_retry_count", 0) or 0)
    if stall_level < 2 and docker_retry_count < 4:
        return False
    if graph_state.pending_tool_calls:
        return False

    signature = "|".join(
        [
            str(record.tool_name),
            str(record.tool_call_id or ""),
            str(record.operation_id or ""),
            str(getattr(harness.state, "last_failure_class", "") or ""),
            str(max(stall_level, docker_retry_count)),
        ]
    )
    if harness.state.scratchpad.get("_repair_loop_status_autocontinue") == signature:
        return False
    harness.state.scratchpad["_repair_loop_status_autocontinue"] = signature

    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="loop_status",
            args={},
            raw_arguments="{}",
            source="system",
        )
    ]
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Auto-continuing repair recovery with `loop_status` after repeated shell/ssh "
                "failures so the next step is grounded in current loop state."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "repair_stall_loop_status_autocontinue",
                "tool_name": record.tool_name,
            },
        )
    )
    harness._runlog(
        "repair_stall_loop_status_autocontinue",
        "scheduled automatic loop_status after repeated shell/ssh failures",
        tool_name=record.tool_name,
        tool_call_id=record.tool_call_id,
        operation_id=record.operation_id,
        stagnation_counters=json_safe_value(counters),
    )
    return True
