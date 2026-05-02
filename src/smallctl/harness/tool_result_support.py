from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..guards import is_small_model_name
from ..models.tool_result import ToolEnvelope
from ..reasoning_policy import task_requires_claim_support
from ..state import ClaimRecord
from ..shell_utils import is_read_only_shell_evidence_action as _is_read_only_evidence_action
from ..tools.fs import is_file_mutating_tool
from ..tools.memory import append_session_notepad_entry


def invalidate_file_read_cache(harness: Any, path: str) -> None:
    cache = harness.state.scratchpad.get("file_read_cache")
    if not isinstance(cache, dict) or not cache:
        return

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(harness.state.cwd) / candidate
    try:
        resolved = str(candidate.resolve())
    except Exception:
        resolved = str(candidate)

    prefix = f"{resolved}|"
    removed = [key for key in cache if isinstance(key, str) and (key == resolved or key.startswith(prefix))]
    if not removed:
        return

    for key in removed:
        cache.pop(key, None)
    harness._runlog(
        "tool_cache_invalidate",
        "invalidated file_read cache after file mutation",
        path=resolved,
        removed_entries=len(removed),
    )


def is_small_model(harness: Any) -> bool:
    scratchpad = getattr(harness.state, "scratchpad", {})
    if isinstance(scratchpad, dict) and "_model_is_small" in scratchpad:
        return bool(scratchpad.get("_model_is_small"))
    model_name = getattr(getattr(harness, "client", None), "model", None)
    if not model_name and isinstance(scratchpad, dict):
        model_name = scratchpad.get("_model_name")
    return is_small_model_name(str(model_name or ""))


def note_anchor(harness: Any, *, content: str, tag: str = "anchor") -> None:
    entry, duplicate, count = append_session_notepad_entry(
        harness.state,
        content=content,
        tag=tag,
    )
    if entry and not duplicate:
        harness._runlog(
            "session_notepad_append",
            "appended session notepad entry",
            entry=entry,
            count=count,
        )


def auto_mirror_session_anchor(
    harness: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
) -> None:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    output = result.output

    if tool_name in {"file_read", "file_write", "file_append", "file_patch", "ast_patch", "file_delete", "dir_list"}:
        path = str(metadata.get("path") or "").strip()
        if not path and isinstance(arguments, dict):
            path = str(arguments.get("path") or "").strip()
        if path:
            action = {
                "file_read": "read",
                "file_write": "wrote",
                "file_append": "appended",
                "file_patch": "patched",
                "ast_patch": "structurally patched",
                "file_delete": "deleted",
                "dir_list": "listed",
            }.get(tool_name, tool_name)
            note_anchor(harness, content=f"{action} path: {path}", tag="path")
        return

    if tool_name in {"shell_exec", "ssh_exec"}:
        command = str(metadata.get("command") or "").strip()
        if not command and isinstance(arguments, dict):
            command = str(arguments.get("command") or "").strip()
        if command:
            note_anchor(harness, content=f"{tool_name} command: {command}", tag="cmd")
        return

    if tool_name == "memory_update" and result.success:
        section = str(metadata.get("section") or "").strip().lower()
        if section == "known_facts":
            text = str(arguments.get("content") or "").strip() if isinstance(arguments, dict) else ""
            if text:
                note_anchor(harness, content=f"known fact: {text}", tag="fact")


def maybe_support_claim_from_evidence(
    *,
    state: Any,
    tool_name: str,
    result: ToolEnvelope,
    evidence: Any,
    harness: Any | None = None,
    operation_id: str | None = None,
) -> ClaimRecord | None:
    if not task_requires_claim_support(state):
        return None
    if evidence is None or not isinstance(getattr(evidence, "evidence_id", ""), str):
        return None
    if not evidence.evidence_id or getattr(evidence, "replayed", False):
        return None
    if getattr(evidence, "evidence_type", "") == "replayed_or_cached":
        return None

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    reason = str(metadata.get("reason") or "").strip().lower()
    if reason in {
        "missing_supported_claim",
        "approval_denied",
        "spec_not_approved",
        "authoring_target_missing",
    }:
        return None
    if tool_name in {
        "task_complete",
        "task_fail",
        "loop_status",
        "memory_update",
        "plan_set",
        "plan_step_update",
        "plan_request_execution",
        "plan_export",
        "artifact_print",
    }:
        return None
    if is_file_mutating_tool(tool_name):
        return None

    if tool_name in {"shell_exec", "ssh_exec"}:
        command = claim_command_from_result(result)
        if not getattr(evidence, "negative", False) and not _is_read_only_evidence_action(command):
            return None
    elif tool_name not in {"file_read", "artifact_read", "dir_list"} and not getattr(evidence, "negative", False):
        return None

    graph = getattr(state, "reasoning_graph", None)
    if graph is None:
        return None

    for claim in getattr(graph, "claim_records", []) or []:
        if evidence.evidence_id in getattr(claim, "supporting_evidence_ids", []):
            if claim.claim_id and claim.claim_id not in evidence.claim_ids:
                evidence.claim_ids.append(claim.claim_id)
            return claim

    statement = claim_statement_from_evidence(evidence)
    if not statement:
        return None

    existing_claim = None
    for claim in getattr(graph, "claim_records", []) or []:
        if (
            getattr(claim, "statement", "") == statement
            and isinstance(getattr(claim, "metadata", None), dict)
            and claim.metadata.get("auto_generated")
        ):
            existing_claim = claim
            break

    if existing_claim is not None:
        if evidence.evidence_id not in existing_claim.supporting_evidence_ids:
            existing_claim.supporting_evidence_ids.append(evidence.evidence_id)
        existing_claim.status = "confirmed"
        existing_claim.confidence = max(float(existing_claim.confidence or 0.0), claim_confidence(evidence))
        if existing_claim.claim_id and existing_claim.claim_id not in evidence.claim_ids:
            evidence.claim_ids.append(existing_claim.claim_id)
        graph.touch_ids()
        return existing_claim

    claim_id = derive_claim_id(tool_name=tool_name, evidence_id=evidence.evidence_id, statement=statement)
    claim = ClaimRecord(
        claim_id=claim_id,
        kind="causal" if getattr(evidence, "negative", False) else "state",
        statement=statement,
        supporting_evidence_ids=[evidence.evidence_id],
        confidence=claim_confidence(evidence),
        status="confirmed",
        metadata={
            "auto_generated": True,
            "source": "evidence",
            "tool_name": tool_name,
            "evidence_id": evidence.evidence_id,
            "phase": str(getattr(evidence, "phase", "") or ""),
        },
    )
    graph.claim_records.append(claim)
    graph.touch_ids()
    evidence.claim_ids.append(claim.claim_id)
    if harness is not None and operation_id:
        stored = harness.state.tool_execution_records.get(operation_id)
        if isinstance(stored, dict):
            claim_ids = stored.get("claim_ids")
            if isinstance(claim_ids, list):
                if claim.claim_id not in claim_ids:
                    claim_ids.append(claim.claim_id)
            else:
                stored["claim_ids"] = [claim.claim_id]
            evidence_record = stored.get("evidence_record")
            if isinstance(evidence_record, dict):
                record_claim_ids = evidence_record.get("claim_ids")
                if isinstance(record_claim_ids, list):
                    if claim.claim_id not in record_claim_ids:
                        record_claim_ids.append(claim.claim_id)
                else:
                    evidence_record["claim_ids"] = [claim.claim_id]
            harness.state.tool_execution_records[operation_id] = stored
    return claim


def claim_command_from_result(result: ToolEnvelope) -> str:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    command = str(metadata.get("command") or "").strip()
    if command:
        return command
    arguments = metadata.get("arguments")
    if isinstance(arguments, dict):
        return str(arguments.get("command") or "").strip()
    return ""


def claim_statement_from_evidence(evidence: Any) -> str:
    statement = str(getattr(evidence, "statement", "") or "").strip()
    if not statement:
        return ""
    prefix = "Observed issue: " if getattr(evidence, "negative", False) else "Observed state: "
    if statement.lower().startswith(prefix.lower()):
        return statement
    return prefix + statement


def claim_confidence(evidence: Any) -> float:
    return max(0.75, min(1.0, float(getattr(evidence, "confidence", 0.0) or 0.0)))


def derive_claim_id(*, tool_name: str, evidence_id: str, statement: str) -> str:
    digest = hashlib.sha1(f"{tool_name}|{evidence_id}|{statement}".encode("utf-8")).hexdigest()[:12]
    return f"C-{digest}"
