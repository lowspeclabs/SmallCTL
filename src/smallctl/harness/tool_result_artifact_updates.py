from __future__ import annotations

from typing import Any

from ..models.tool_result import ToolEnvelope
from ..challenge_progress import record_code_change
from ..normalization import dedupe_keep_tail
from .artifact_tracking import consolidate_shell_attempt_family as _consolidate_shell_attempt_family
from .artifact_read_ledger import record_artifact_read_ledger as _record_artifact_read_ledger
from .tool_result_evidence import record_evidence as _record_evidence
from .tool_result_support import (
    auto_mirror_session_anchor as _auto_mirror_session_anchor,
    invalidate_file_read_cache as _invalidate_file_read_cache,
    maybe_support_claim_from_evidence as _maybe_support_claim_from_evidence,
)
from .tool_result_verification import (
    _annotate_verifier_artifact,
    _store_verifier_verdict,
)
from .verifier_monitor import track_verifier_rejection
from ..fama.runtime import _handle_signal
from ..fama.signals import FamaFailureKind, FamaSignal, current_step
from ..shell_utils import (
    file_read_cache_key as _file_read_cache_key,
    ssh_file_read_cache_key as _ssh_file_read_cache_key,
)
from ..tools.fs import is_file_mutating_tool
from ..tools.ssh_files import SSH_FILE_MUTATING_TOOLS
from .tool_result_pure_helpers import (
    is_dry_run_invariant_violation as _is_dry_run_invariant_violation,
    should_auto_record_known_fact as _should_auto_record_known_fact,
)
from .remote_mutation_helpers import _SSH_FILE_VERIFIER_TOOLS

# Re-export extracted helpers so existing imports continue to work
from .tool_result_artifact_lifecycle import (
    _supersede_prior_read_artifacts,
    _mark_prior_read_artifacts_stale,
    _maybe_emit_artifact_read_eof_overread_nudge,
)
from .tool_result_verifier_staleness import _mark_verifier_stale_after_file_change
from .tool_result_remote_mutation import (
    _observe_remote_installer_preflight_check,
    _record_remote_mutation_requirement,
    _emit_remote_mutation_nudge,
    _clear_remote_mutation_requirement_from_tool,
    _record_failed_verification_attempt,
    _maybe_emit_bounded_region_trap_nudge,
    _maybe_emit_small_file_rewrite_nudge,
    _recent_ssh_file_read_size,
    _handle_remote_mutation_verifier_result,
    record_remote_mutation_provenance,
    observe_runtime_projection_read,
)
from .tool_result_ssh_memory import _remember_session_ssh_target
from .tool_result_web_memory import _remember_web_search_results
from .tool_result_context_invalidation import _emit_context_invalidation
from .tool_result_touched_symbols import _record_touched_symbols_from_mutation
from .tool_result_critical_errors import _extract_and_pin_critical_errors
from .tool_result_stderr_circuit_breaker import _record_stderr_signature_circuit_breaker
from .tool_result_subtask_ledger import _update_subtask_ledger_from_verifier


def _apply_shell_exec_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
) -> None:
    """Apply side-effects for shell_exec and ssh_exec results."""
    artifact_id = str(getattr(artifact, "artifact_id", "") or "").strip() if artifact else ""
    if artifact_id:
        _consolidate_shell_attempt_family(state=service.harness.state, artifact_id=artifact_id, result=result)
        _record_stderr_signature_circuit_breaker(service, tool_name=tool_name, result=result)
    if tool_name == "ssh_exec":
        _remember_session_ssh_target(service, tool_name=tool_name, result=result, arguments=arguments)
        _observe_remote_installer_preflight_check(service, result=result, arguments=arguments)
        _handle_remote_mutation_verifier_result(service, result=result, arguments=arguments)
        observe_runtime_projection_read(service, result=result, arguments=arguments)
        if not result.success:
            _record_failed_verification_attempt(service, tool_name=tool_name, result=result, arguments=arguments)
        _record_remote_mutation_requirement(service, result=result, arguments=arguments)


def _apply_ssh_file_verifier_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
) -> None:
    """Apply side-effects for SSH file verifier tools."""
    _remember_session_ssh_target(service, tool_name=tool_name, result=result, arguments=arguments)
    if not result.success:
        _record_failed_verification_attempt(service, tool_name=tool_name, result=result, arguments=arguments)
        _maybe_emit_bounded_region_trap_nudge(service, tool_name=tool_name, result=result, arguments=arguments)
    _clear_remote_mutation_requirement_from_tool(service, tool_name=tool_name, result=result, arguments=arguments)


def _apply_web_search_updates(
    service: Any,
    *,
    result: ToolEnvelope,
    artifact: Any,
) -> None:
    """Apply side-effects for web_search results."""
    _remember_web_search_results(service, result=result, artifact=artifact)


def _apply_file_read_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
) -> None:
    """Apply side-effects for file_read and ssh_file_read results."""
    if tool_name == "file_read":
        cache_key = _file_read_cache_key(service.harness.state.cwd, result.metadata)
        if cache_key:
            cache = service.harness.state.scratchpad.setdefault("file_read_cache", {})
            if isinstance(cache, dict):
                cache[cache_key] = artifact.artifact_id
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        read_path = str(metadata.get("path") or "").strip()
        if not read_path and isinstance(arguments, dict):
            read_path = str(arguments.get("path") or "").strip()
        if read_path:
            _supersede_prior_read_artifacts(
                service,
                new_artifact_id=artifact.artifact_id,
                tool_name="file_read",
                path=read_path,
            )
    elif tool_name == "ssh_file_read":
        cache_key = _ssh_file_read_cache_key(result.metadata)
        if cache_key:
            cache = service.harness.state.scratchpad.setdefault("ssh_file_read_cache", {})
            if isinstance(cache, dict):
                cache[cache_key] = artifact.artifact_id
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        read_path = str(metadata.get("path") or "").strip()
        read_host = str(metadata.get("host") or "").strip()
        if not read_path and isinstance(arguments, dict):
            read_path = str(arguments.get("path") or "").strip()
            read_host = str(arguments.get("host") or "").strip()
        if read_path:
            _supersede_prior_read_artifacts(
                service,
                new_artifact_id=artifact.artifact_id,
                tool_name="ssh_file_read",
                path=read_path,
                host=read_host,
            )


def _apply_file_mutation_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
    artifact: Any,
) -> None:
    """Apply side-effects for local file mutating tools."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if bool(metadata.get("dry_run")) or metadata.get("changed") is False:
        return
    staged_only = bool(metadata.get("staged_only")) or metadata.get("write_session_finalized") is False
    mutated_path = ""
    if isinstance(result.metadata, dict):
        mutated_path = str(result.metadata.get("path") or "").strip()
    if not mutated_path and isinstance(arguments, dict):
        mutated_path = str(arguments.get("path") or "").strip()
    if mutated_path and not staged_only:
        _invalidate_file_read_cache(service.harness, mutated_path)
        _mark_prior_read_artifacts_stale(
            service,
            path=mutated_path,
            reason=f"{tool_name}_applied",
        )
        _emit_context_invalidation(
            service,
            reason="file_changed",
            paths=[mutated_path],
            details={
                "tool_name": tool_name,
                "state_change": f"File changed: {mutated_path}",
            },
        )
        _mark_verifier_stale_after_file_change(
            service,
            tool_name=tool_name,
            paths=[mutated_path],
        )
    if staged_only:
        staging_path = str(metadata.get("staging_path") or "").strip()
        if staging_path:
            _invalidate_file_read_cache(service.harness, staging_path)
    else:
        record_code_change(
            service.harness.state,
            tool_name=tool_name,
            path=mutated_path,
            changed=True,
        )
    _record_touched_symbols_from_mutation(
        service,
        tool_name=tool_name,
        result=result,
        arguments=arguments,
        artifact=artifact,
        mutated_path=mutated_path,
    )


def _apply_ssh_file_mutation_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
    artifact: Any,
) -> None:
    """Apply side-effects for SSH file mutating tools."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if bool(metadata.get("dry_run")) or metadata.get("changed") is False:
        return
    mutated_path = ""
    host = ""
    if isinstance(result.metadata, dict):
        mutated_path = str(result.metadata.get("path") or "").strip()
        host = str(result.metadata.get("host") or "").strip().lower()
    if not mutated_path and isinstance(arguments, dict):
        mutated_path = str(arguments.get("path") or "").strip()
        host = str(arguments.get("host") or "").strip().lower()
    if mutated_path and host:
        record_remote_mutation_provenance(
            service, host=host, path=mutated_path, arguments=arguments
        )
        cache = service.harness.state.scratchpad.get("ssh_file_read_cache")
        if isinstance(cache, dict):
            prefix = f"ssh://{host}{mutated_path}"
            for key in list(cache.keys()):
                if key.startswith(prefix):
                    cache.pop(key, None)
        # Do not emit context invalidation for SSH file mutations; remote writes
        # are intentional mutations and should not invalidate prior observations.
        _mark_verifier_stale_after_file_change(
            service,
            tool_name=tool_name,
            paths=[f"{host}:{mutated_path}"],
        )
        from ..graph.tool_call_parser import allow_repeated_tool_call_once

        one_shot_args: dict[str, Any] = {"path": mutated_path, "host": host}
        if isinstance(arguments, dict):
            for conn_key in ("user", "password", "target", "port", "identity_file"):
                val = arguments.get(conn_key)
                if val is not None:
                    one_shot_args[conn_key] = val
        allow_repeated_tool_call_once(
            service.harness,
            "ssh_file_read",
            one_shot_args,
        )
        # Mark remote installer preflight clean when ssh_file_write verifies a script path
        if tool_name == "ssh_file_write":
            from ..tools.shell_support import _mark_remote_installer_preflight_clean_from_write

            user = str(arguments.get("user") or "").strip() or None
            _mark_remote_installer_preflight_clean_from_write(
                service.harness.state,
                host=host,
                user=user,
                script_path=mutated_path,
            )
    _record_touched_symbols_from_mutation(
        service,
        tool_name=tool_name,
        result=result,
        arguments=arguments,
        artifact=artifact,
        mutated_path=mutated_path,
    )


def _apply_plan_and_read_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
) -> None:
    """Apply side-effects for plan tools, memory_update, and artifact_read."""
    if tool_name in {"plan_set", "plan_step_update", "plan_request_execution", "plan_export"}:
        playbook_artifact_id = str(result.metadata.get("artifact_id", "") or "").strip()
        if playbook_artifact_id:
            service.harness.state.plan_artifact_id = playbook_artifact_id
            service.harness.state.plan_resolved = True
            service.harness.state.retrieval_cache = [playbook_artifact_id]

    if tool_name == "memory_update" and result.success:
        section = str(result.metadata.get("section", "")).strip().lower()
        if section == "plan":
            service.harness.state.plan_artifact_id = artifact.artifact_id
            service.harness.state.plan_resolved = True
    elif tool_name == "artifact_read" and result.success and artifact:
        artifact_id = str(result.metadata.get("artifact_id", "")).strip()
        if artifact_id:
            if artifact_id == service.harness.state.plan_artifact_id:
                service.harness.state.plan_resolved = True
            elif (
                not service.harness.state.plan_artifact_id
                and artifact.tool_name == "memory_update"
                and str(artifact.metadata.get("section", "")).strip().lower() == "plan"
            ):
                service.harness.state.plan_artifact_id = artifact_id
                service.harness.state.plan_resolved = True
            if result.metadata.get("truncated"):
                suppressed = service.harness.state.scratchpad.get("suppressed_truncated_artifact_ids", [])
                if isinstance(suppressed, list):
                    service.harness.state.scratchpad["suppressed_truncated_artifact_ids"] = dedupe_keep_tail(
                        suppressed + [artifact_id],
                        limit=12,
                    )
                else:
                    service.harness.state.scratchpad["suppressed_truncated_artifact_ids"] = [artifact_id]
                service.harness._runlog(
                    "artifact_read_truncated",
                    "artifact_read returned a truncated slice",
                    artifact_id=artifact_id,
                    total_lines=result.metadata.get("total_lines"),
                    start_line=result.metadata.get("start_line"),
                    end_line=result.metadata.get("end_line"),
                    marker_included=True,
                )
        _record_artifact_read_ledger(
            service,
            result=result,
            arguments=arguments,
            artifact=artifact,
        )
        _maybe_emit_artifact_read_eof_overread_nudge(
            service,
            result=result,
            artifact=artifact,
        )


def _apply_verifier_and_evidence_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
) -> Any:
    """Store verifier verdict, update ledger, emit invalidations, and record evidence."""
    verifier_verdict = _store_verifier_verdict(
        service.harness.state,
        tool_name=tool_name,
        result=result,
        arguments=arguments,
    )
    if isinstance(verifier_verdict, dict):
        service.harness._runlog(
            "verifier_decision",
            "recorded verifier verdict",
            tool_name=tool_name,
            verdict=str(verifier_verdict.get("verdict", "")),
            verifier_kind=str(verifier_verdict.get("verifier_kind", "")),
            command=str(verifier_verdict.get("command", "")),
            target=str(verifier_verdict.get("target", "")),
            failure_class=str(verifier_verdict.get("failure_class", "")),
        )
    loop_info = track_verifier_rejection(service.harness.state, verifier_verdict)
    if loop_info.get("is_loop"):
        service.harness._runlog(
            "verifier_loop_detected",
            "Verifier rejecting task_complete repeatedly",
            rejection_count=loop_info["rejection_count"],
            last_verdict=loop_info["verdict"],
        )
        _handle_signal(
            service.harness,
            state=service.harness.state,
            config=getattr(service.harness, "config", None),
            signal=FamaSignal(
                kind=FamaFailureKind.EARLY_STOP,
                severity=2,
                source="verifier",
                evidence=f"verifier_loop_detected with {loop_info['rejection_count']} rejections",
                step=current_step(service.harness.state),
                tool_name="task_complete",
                failure_class="verifier_failed",
                next_safe_action="Read the failing output and patch one narrow cause, then rerun the smallest check.",
            ),
            dedupe=True,
        )
    rejection_count = loop_info.get("rejection_count", 0)
    if rejection_count >= 3:
        _handle_verifier_loop_hard_stop(service, verifier_verdict, rejection_count)
    _update_subtask_ledger_from_verifier(service, verifier_verdict)
    if isinstance(verifier_verdict, dict):
        verdict_label = str(verifier_verdict.get("verdict") or "").strip().lower()
        if verdict_label == "fail":
            _emit_context_invalidation(
                service,
                reason="verifier_failed",
                details={
                    "command": str(verifier_verdict.get("command") or ""),
                    "target": str(verifier_verdict.get("target") or ""),
                    "failure_mode": str(getattr(service.harness.state, "last_failure_class", "") or ""),
                    "state_change": "Verifier failure invalidated optimistic context",
                },
            )
    if artifact and isinstance(verifier_verdict, dict) and verifier_verdict:
        _annotate_verifier_artifact(artifact, verifier_verdict=verifier_verdict)
    return verifier_verdict


def _handle_verifier_loop_hard_stop(
    service: Any,
    verifier_verdict: dict[str, Any] | None,
    rejection_count: int,
) -> None:
    """When verifier has rejected task_complete >=3 times, enforce a required
    action-class change instead of allowing another same-class verifier.
    """
    if not isinstance(verifier_verdict, dict):
        return
    state = service.harness.state
    from ..models.conversation import ConversationMessage

    allowed_classes = ["research", "mutation", "ask_user", "stop_blocked"]
    state.scratchpad["_verifier_loop_required_action_classes"] = allowed_classes
    state.scratchpad["_verifier_loop_rejection_count"] = rejection_count

    command = str(verifier_verdict.get("command") or "").strip()
    tool = str(verifier_verdict.get("tool") or "").strip()
    target = str(verifier_verdict.get("target") or "").strip()

    # Try to infer a fully-specified verifier readback call
    readback_tool = None
    readback_args: dict[str, Any] = {}
    if tool == "ssh_exec" and target:
        parts = target.split("::", 1)
        host = parts[0].strip() if parts else ""
        remote_command = parts[1].strip() if len(parts) > 1 else ""
        if host and remote_command:
            readback_tool = "ssh_exec"
            readback_args = {"host": host, "command": remote_command}
    if not readback_tool:
        # Check if the target looks like a file path we could read back
        path = target or command
        if path and path.startswith("/"):
            readback_tool = "ssh_file_read"
            readback_args = {"host": "", "path": path}

    if readback_tool:
        state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    f"VERIFIER LOOP HARD STOP: The verifier has rejected task_complete {rejection_count} times. "
                    f"Your next action must change class. Allowed classes: research, mutation, ask_user, stop_blocked. "
                    f"Auto-executing required verification: {readback_tool}({readback_args})."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "verifier_loop_hard_stop_auto_verifier",
                    "rejection_count": rejection_count,
                    "auto_tool": readback_tool,
                    "auto_args": readback_args,
                    "allowed_action_classes": allowed_classes,
                },
            )
        )
    else:
        state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    f"VERIFIER LOOP HARD STOP: The verifier has rejected task_complete {rejection_count} times. "
                    "Your next action MUST change class. Allowed classes: research (web_search/web_fetch/ask_human), "
                    "mutation (file/SSH write or patch), ask_user (ask_human/escalate), or stop_blocked (task_fail). "
                    "Do not run another verifier in the same class as the failing one."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "verifier_loop_hard_stop_nudge",
                    "rejection_count": rejection_count,
                    "allowed_action_classes": allowed_classes,
                },
            )
        )


def _apply_finalization_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
    operation_id: str | None,
    verifier_verdict: Any,
) -> Any:
    """Record evidence, update known facts, and return the evidence object."""
    evidence = _record_evidence(
        service,
        tool_name=tool_name,
        result=result,
        artifact=artifact,
        operation_id=operation_id,
    )
    _maybe_support_claim_from_evidence(
        state=service.harness.state,
        tool_name=tool_name,
        result=result,
        evidence=evidence,
        harness=service.harness,
        operation_id=operation_id,
    )
    _auto_mirror_session_anchor(service.harness, tool_name=tool_name, result=result, arguments=arguments)

    if _should_auto_record_known_fact(tool_name, result):
        fact_label = evidence.statement or (artifact.summary if artifact else "") or tool_name
        prefix = f"{tool_name}: "
        if fact_label.startswith(prefix):
            fact_label = fact_label[len(prefix):]
        service.harness.state.working_memory.known_facts = dedupe_keep_tail(
            service.harness.state.working_memory.known_facts + [f"{tool_name}: {fact_label}"],
            limit=12,
        )
    _extract_and_pin_critical_errors(service, tool_name=tool_name, result=result, artifact=artifact)
    if result.success:
        service.harness.state.recent_errors = []
    else:
        if result.metadata.get("reason") == "web_fetch_duplicate_result_id":
            return evidence
        error_text = str(result.error or result.metadata.get("error") or "tool failed").strip()
        error_kind = str(result.metadata.get("error_kind") or "").strip()
        if error_kind:
            error_text = f"{error_kind}: {error_text}" if error_text else error_kind
        if error_text:
            recent_errors = list(getattr(service.harness.state, "recent_errors", []) or [])
            recent_errors.append(f"{tool_name}: {error_text}")
            service.harness.state.recent_errors = recent_errors[-8:]
    return evidence


def apply_artifact_success_outcome(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
    operation_id: str | None,
) -> Any:
    artifact_id = str(getattr(artifact, "artifact_id", "") or "").strip() if artifact else ""
    if artifact:
        service.harness.state.artifacts[artifact.artifact_id] = artifact
        service.harness.state.retrieval_cache = [artifact.artifact_id]
    if _is_dry_run_invariant_violation(tool_name, result):
        result.success = False
        result.status = "failed"
        if not result.error:
            result.error = "Tool returned contradictory metadata: dry_run=true with changed=yes. This is an invariant violation."
        result.metadata = {**(result.metadata if isinstance(result.metadata, dict) else {}), "dry_run_invariant_violation": True}
        service.harness._runlog(
            "dry_run_invariant_violation",
            "SSH mutating tool returned dry_run=true and changed=yes; treating as failure",
            tool_name=tool_name,
        )
    if tool_name in {"shell_exec", "ssh_exec"}:
        _apply_shell_exec_updates(service, tool_name=tool_name, result=result, artifact=artifact, arguments=arguments)
    elif tool_name in _SSH_FILE_VERIFIER_TOOLS:
        _apply_ssh_file_verifier_updates(service, tool_name=tool_name, result=result, arguments=arguments)
    elif tool_name == "web_search" and result.success:
        _apply_web_search_updates(service, result=result, artifact=artifact)

    verifier_verdict = _apply_verifier_and_evidence_updates(
        service, tool_name=tool_name, result=result, artifact=artifact, arguments=arguments
    )

    if tool_name in {"file_read", "ssh_file_read"} and result.success and artifact:
        _apply_file_read_updates(service, tool_name=tool_name, result=result, artifact=artifact, arguments=arguments)
    elif is_file_mutating_tool(tool_name) and result.success:
        _apply_file_mutation_updates(service, tool_name=tool_name, result=result, arguments=arguments, artifact=artifact)
    elif tool_name in SSH_FILE_MUTATING_TOOLS and result.success:
        _apply_ssh_file_mutation_updates(service, tool_name=tool_name, result=result, arguments=arguments, artifact=artifact)

    _apply_plan_and_read_updates(service, tool_name=tool_name, result=result, artifact=artifact, arguments=arguments)

    return _apply_finalization_updates(
        service,
        tool_name=tool_name,
        result=result,
        artifact=artifact,
        arguments=arguments,
        operation_id=operation_id,
        verifier_verdict=verifier_verdict,
    )
