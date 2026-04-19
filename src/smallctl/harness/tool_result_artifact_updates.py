from __future__ import annotations

from typing import Any

from ..models.tool_result import ToolEnvelope
from ..normalization import dedupe_keep_tail
from .artifact_tracking import consolidate_shell_attempt_family as _consolidate_shell_attempt_family
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
from ..shell_utils import file_read_cache_key as _file_read_cache_key
from ..tools.fs import is_file_mutating_tool


def _remember_session_ssh_target(
    service: Any,
    *,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
) -> None:
    if not isinstance(arguments, dict):
        return
    host = str(arguments.get("host") or "").strip().lower()
    user = str(arguments.get("user") or "").strip()
    if not host or not user:
        return

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if not (
        result.success
        or bool(metadata.get("ssh_transport_succeeded"))
        or str(metadata.get("failure_kind") or "").strip() == "remote_command"
    ):
        return

    targets = service.harness.state.scratchpad.setdefault("_session_ssh_targets", {})
    if not isinstance(targets, dict):
        targets = {}
        service.harness.state.scratchpad["_session_ssh_targets"] = targets

    remembered: dict[str, Any] = {
        "host": host,
        "user": user,
    }
    password = str(arguments.get("password") or "").strip()
    if password:
        remembered["password"] = password
    identity_file = str(arguments.get("identity_file") or "").strip()
    if identity_file:
        remembered["identity_file"] = identity_file
    port = arguments.get("port")
    if isinstance(port, int):
        remembered["port"] = port
    targets[host] = remembered


def _emit_context_invalidation(
    service: Any,
    *,
    reason: str,
    paths: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    event = service.harness.state.invalidate_context(
        reason=reason,
        paths=paths,
        details=details,
    )
    runlog = getattr(service.harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "context_invalidated",
            "context invalidation applied",
            reason=event.get("reason", reason),
            paths=event.get("paths", []),
            invalidated_fact_count=event.get("invalidated_fact_count", 0),
            invalidated_memory_count=event.get("invalidated_memory_count", 0),
            invalidated_facts=event.get("invalidated_facts", []),
            invalidated_memory_ids=event.get("invalidated_memory_ids", []),
            invalidated_turn_bundle_count=event.get("invalidated_turn_bundle_count", 0),
            invalidated_turn_bundle_ids=event.get("invalidated_turn_bundle_ids", []),
            invalidated_brief_count=event.get("invalidated_brief_count", 0),
            invalidated_brief_ids=event.get("invalidated_brief_ids", []),
            invalidated_summary_count=event.get("invalidated_summary_count", 0),
            invalidated_summary_ids=event.get("invalidated_summary_ids", []),
            invalidated_artifact_count=event.get("invalidated_artifact_count", 0),
            invalidated_artifact_ids=event.get("invalidated_artifact_ids", []),
            invalidated_observation_count=event.get("invalidated_observation_count", 0),
            invalidated_observation_ids=event.get("invalidated_observation_ids", []),
            details=event.get("details", {}),
        )


def apply_artifact_success_outcome(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
    operation_id: str | None,
) -> Any:
    service.harness.state.artifacts[artifact.artifact_id] = artifact
    service.harness.state.retrieval_cache = [artifact.artifact_id]
    if tool_name in {"shell_exec", "ssh_exec"}:
        _consolidate_shell_attempt_family(state=service.harness.state, artifact_id=artifact.artifact_id, result=result)
    if tool_name == "ssh_exec":
        _remember_session_ssh_target(
            service,
            result=result,
            arguments=arguments,
        )
    verifier_verdict = _store_verifier_verdict(
        service.harness.state,
        tool_name=tool_name,
        result=result,
        arguments=arguments,
    )
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
    if tool_name == "file_read" and result.success:
        cache_key = _file_read_cache_key(service.harness.state.cwd, result.metadata)
        if cache_key:
            cache = service.harness.state.scratchpad.setdefault("file_read_cache", {})
            if isinstance(cache, dict):
                cache[cache_key] = artifact.artifact_id
    elif is_file_mutating_tool(tool_name) and result.success:
        mutated_path = ""
        if isinstance(result.metadata, dict):
            mutated_path = str(result.metadata.get("path") or "").strip()
        if not mutated_path and isinstance(arguments, dict):
            mutated_path = str(arguments.get("path") or "").strip()
        if mutated_path:
            _invalidate_file_read_cache(service.harness, mutated_path)
            _emit_context_invalidation(
                service,
                reason="file_changed",
                paths=[mutated_path],
                details={
                    "tool_name": tool_name,
                    "state_change": f"File changed: {mutated_path}",
                },
            )

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
    elif tool_name == "artifact_read" and result.success:
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

    if tool_name != "shell_exec" and not result.metadata.get("skip_auto_fact_record"):
        fact_label = evidence.statement or artifact.summary or tool_name
        prefix = f"{tool_name}: "
        if fact_label.startswith(prefix):
            fact_label = fact_label[len(prefix):]
        service.harness.state.working_memory.known_facts = dedupe_keep_tail(
            service.harness.state.working_memory.known_facts + [f"{tool_name}: {fact_label}"],
            limit=12,
        )
    service.harness.state.recent_errors = []
    return evidence
