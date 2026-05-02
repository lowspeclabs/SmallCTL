from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .models.conversation import ConversationMessage
from .state_schema import (
    ClaimRecord,
    DecisionRecord,
    EvidenceRecord,
    ExecutionPlan,
    ExperienceMemory,
    MemoryEntry,
    PlanStep,
    PlanInterrupt,
    ReasoningGraph,
    RunBrief,
    StepEvidenceArtifact,
    StepOutputSpec,
    StepVerificationResult,
    StepVerifierSpec,
    WorkingMemory,
)
from .state_support import (
    _coerce_bool,
    _coerce_conversation_message_payload,
    _coerce_dict_payload,
    _coerce_float,
    _coerce_int,
    _coerce_json_dict_payload,
    _coerce_list_payload,
    _coerce_string_list,
    _coerce_timestamp_string,
    _coerce_write_section_ranges,
    _filter_dataclass_payload,
    json_safe_value,
    clip_string_list,
    clip_text_value,
    normalize_intent_label,
)

from .state_memory import (
    _coerce_experience_memory,
    _coerce_memory_entry_list,
    _coerce_working_memory,
    _compact_plan_step_lines,
    _trim_recent_messages,
    align_memory_entries,
    memory_entry_is_stale,
)
from .state_session_records import (
    _coerce_artifact_record,
    _coerce_background_process_record,
    _coerce_context_brief,
    _coerce_episodic_summary,
    _coerce_pending_interrupt_payload,
    _coerce_prompt_budget,
    _coerce_tool_envelope_payload,
    _coerce_tool_execution_record,
    _coerce_turn_bundle,
    _coerce_write_session,
)


def _coerce_evidence_record(value: Any) -> Any | None:
    if isinstance(value, EvidenceRecord):
        return value
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    evidence_id = str(payload.get("evidence_id", "") or "").strip()
    if not evidence_id:
        return None
    payload["evidence_id"] = evidence_id
    payload["kind"] = str(payload.get("kind", "observation") or "observation")
    payload["statement"] = str(payload.get("statement", "") or "")
    payload["phase"] = str(payload.get("phase", "") or "")
    payload["tool_name"] = str(payload.get("tool_name", "") or "")
    payload["operation_id"] = str(payload.get("operation_id", "") or "")
    payload["artifact_id"] = str(payload.get("artifact_id", "") or "")
    payload["source"] = str(payload.get("source", "") or "")
    payload["evidence_type"] = str(payload.get("evidence_type", "direct_observation") or "direct_observation")
    payload["confidence"] = max(0.0, min(1.0, _coerce_float(payload.get("confidence"), default=0.0)))
    payload["negative"] = _coerce_bool(payload.get("negative"), default=False)
    payload["replayed"] = _coerce_bool(payload.get("replayed"), default=False)
    payload["claim_ids"] = _coerce_string_list(payload.get("claim_ids"))
    payload["decision_ids"] = _coerce_string_list(payload.get("decision_ids"))
    payload["evidence_refs"] = _coerce_string_list(payload.get("evidence_refs"))
    metadata = json_safe_value(payload.get("metadata") or {})
    payload["metadata"] = metadata if isinstance(metadata, dict) else {}
    return EvidenceRecord(**_filter_dataclass_payload(EvidenceRecord, payload))


def _coerce_decision_record(value: Any) -> Any | None:
    if isinstance(value, DecisionRecord):
        return value
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    decision_id = str(payload.get("decision_id", "") or "").strip()
    if not decision_id:
        return None
    payload["decision_id"] = decision_id
    payload["phase"] = str(payload.get("phase", "") or "")
    payload["intent_label"] = str(payload.get("intent_label", "") or "")
    payload["requested_tool"] = str(payload.get("requested_tool", "") or "")
    payload["argument_fingerprint"] = str(payload.get("argument_fingerprint", "") or "")
    payload["plan_step_id"] = str(payload.get("plan_step_id", "") or "")
    payload["evidence_refs"] = _coerce_string_list(payload.get("evidence_refs"))
    payload["rationale_summary"] = str(payload.get("rationale_summary", "") or "")
    payload["created_at"] = _coerce_timestamp_string(payload.get("created_at"))
    payload["status"] = str(payload.get("status", "active") or "active")
    metadata = json_safe_value(payload.get("metadata") or {})
    payload["metadata"] = metadata if isinstance(metadata, dict) else {}
    return DecisionRecord(**_filter_dataclass_payload(DecisionRecord, payload))


def _coerce_claim_record(value: Any) -> Any | None:
    if isinstance(value, ClaimRecord):
        return value
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    claim_id = str(payload.get("claim_id", "") or "").strip()
    if not claim_id:
        return None
    payload["claim_id"] = claim_id
    payload["kind"] = str(payload.get("kind", "hypothesis") or "hypothesis")
    payload["statement"] = str(payload.get("statement", "") or "")
    payload["supporting_evidence_ids"] = _coerce_string_list(payload.get("supporting_evidence_ids"))
    payload["missing_evidence"] = _coerce_string_list(payload.get("missing_evidence"))
    payload["alternative_explanations"] = _coerce_string_list(payload.get("alternative_explanations"))
    payload["confidence"] = max(0.0, min(1.0, _coerce_float(payload.get("confidence"), default=0.0)))
    payload["status"] = str(payload.get("status", "candidate") or "candidate")
    payload["decision_ids"] = _coerce_string_list(payload.get("decision_ids"))
    metadata = json_safe_value(payload.get("metadata") or {})
    payload["metadata"] = metadata if isinstance(metadata, dict) else {}
    return ClaimRecord(**_filter_dataclass_payload(ClaimRecord, payload))


def _coerce_reasoning_graph(value: Any) -> Any:
    if isinstance(value, ReasoningGraph):
        graph = value
    elif isinstance(value, dict):
        payload = dict(value)
        payload["graph_version"] = _coerce_int(payload.get("graph_version"), default=1)
        payload["evidence_records"] = [
            record
            for item in _coerce_list_payload(payload.get("evidence_records"))
            if (record := _coerce_evidence_record(item)) is not None
        ]
        payload["decision_records"] = [
            record
            for item in _coerce_list_payload(payload.get("decision_records"))
            if (record := _coerce_decision_record(item)) is not None
        ]
        payload["claim_records"] = [
            record
            for item in _coerce_list_payload(payload.get("claim_records"))
            if (record := _coerce_claim_record(item)) is not None
        ]
        payload["evidence_ids"] = _coerce_string_list(payload.get("evidence_ids"))
        payload["decision_ids"] = _coerce_string_list(payload.get("decision_ids"))
        payload["claim_ids"] = _coerce_string_list(payload.get("claim_ids"))
        graph = ReasoningGraph(**_filter_dataclass_payload(ReasoningGraph, payload))
    else:
        graph = ReasoningGraph()
    graph.touch_ids()
    return graph


def _coerce_run_brief(value: Any) -> Any:
    if isinstance(value, RunBrief):
        return value
    if isinstance(value, dict):
        payload = _filter_dataclass_payload(RunBrief, value)
        payload["task_contract"] = str(payload.get("task_contract") or "")
        payload["inputs"] = _coerce_string_list(payload.get("inputs"))
        payload["outputs"] = _coerce_string_list(payload.get("outputs"))
        payload["constraints"] = _coerce_string_list(payload.get("constraints"))
        payload["acceptance_criteria"] = _coerce_string_list(payload.get("acceptance_criteria"))
        payload["implementation_plan"] = _coerce_string_list(payload.get("implementation_plan"))
        if "original_task" in payload:
            payload["original_task"] = str(payload.get("original_task") or "")
        if "current_phase_objective" in payload:
            payload["current_phase_objective"] = str(payload.get("current_phase_objective") or "")
        return RunBrief(**payload)
    return RunBrief()


def _coerce_conversation_message(value: Any) -> Any | None:
    from .models.conversation import ConversationMessage

    if not isinstance(value, dict):
        return None
    normalized = json_safe_value(value)
    if not isinstance(normalized, dict):
        return None
    payload = _filter_dataclass_payload(ConversationMessage, normalized)
    role = payload.get("role")
    if not isinstance(role, str) or not role.strip():
        return None
    for key in ("content", "name", "tool_call_id", "retrieval_safe_text"):
        if key in payload and payload[key] is not None:
            payload[key] = str(payload[key])
    if "tool_calls" in payload:
        tool_calls = json_safe_value(payload.get("tool_calls") or [])
        payload["tool_calls"] = tool_calls if isinstance(tool_calls, list) else []
    if "metadata" in payload:
        metadata = json_safe_value(payload.get("metadata") or {})
        payload["metadata"] = metadata if isinstance(metadata, dict) else {}
    return ConversationMessage(**payload)


def _coerce_plan_step(value: Any) -> Any | None:
    if isinstance(value, PlanStep):
        return value
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    step_id = str(payload.get("step_id", "") or "").strip()
    title = str(payload.get("title", "") or "").strip()
    if not step_id and not title:
        return None
    payload["step_id"] = step_id or title
    payload["title"] = title or step_id
    payload["description"] = str(payload.get("description", "") or "")
    payload["status"] = str(payload.get("status", "pending") or "pending")
    payload["notes"] = _coerce_string_list(payload.get("notes"))
    payload["depends_on"] = _coerce_string_list(payload.get("depends_on"))
    payload["evidence_refs"] = _coerce_string_list(payload.get("evidence_refs"))
    payload["claim_refs"] = _coerce_string_list(payload.get("claim_refs"))
    payload["task"] = str(payload.get("task", "") or "")
    payload["tool_allowlist"] = _coerce_string_list(payload.get("tool_allowlist"))
    payload["prompt_token_budget"] = max(0, _coerce_int(payload.get("prompt_token_budget"), default=0))
    payload["acceptance"] = _coerce_string_list(payload.get("acceptance"))
    payload["verifiers"] = [
        spec
        for item in _coerce_list_payload(payload.get("verifiers"))
        if (spec := _coerce_step_verifier_spec(item)) is not None
    ]
    payload["outputs_expected"] = [
        spec
        for item in _coerce_list_payload(payload.get("outputs_expected"))
        if (spec := _coerce_step_output_spec(item)) is not None
    ]
    payload["max_retries"] = max(0, _coerce_int(payload.get("max_retries"), default=3))
    payload["retry_count"] = max(0, _coerce_int(payload.get("retry_count"), default=0))
    payload["failure_reasons"] = _coerce_string_list(payload.get("failure_reasons"))[-10:]
    payload["substeps"] = [
        substep
        for item in _coerce_list_payload(payload.get("substeps"))
        if (substep := _coerce_plan_step(item)) is not None
    ]
    return PlanStep(**_filter_dataclass_payload(PlanStep, payload))


def _coerce_step_verifier_spec(value: Any) -> StepVerifierSpec | None:
    if isinstance(value, StepVerifierSpec):
        return value
    if isinstance(value, str):
        kind = value.strip()
        if not kind:
            return None
        return StepVerifierSpec(kind=kind)
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    kind = str(payload.get("kind", "") or "").strip()
    if not kind:
        return None
    payload["kind"] = kind
    args = json_safe_value(payload.get("args") or {})
    payload["args"] = args if isinstance(args, dict) else {}
    payload["required"] = _coerce_bool(payload.get("required"), default=True)
    payload["timeout_sec"] = max(1, _coerce_int(payload.get("timeout_sec"), default=30))
    return StepVerifierSpec(**_filter_dataclass_payload(StepVerifierSpec, payload))


def _coerce_step_output_spec(value: Any) -> StepOutputSpec | None:
    if isinstance(value, StepOutputSpec):
        return value
    if isinstance(value, str):
        ref = value.strip()
        if not ref:
            return None
        return StepOutputSpec(ref=ref)
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    payload["kind"] = str(payload.get("kind", "artifact") or "artifact")
    payload["ref"] = str(payload.get("ref", "") or "")
    payload["description"] = str(payload.get("description", "") or "")
    payload["required"] = _coerce_bool(payload.get("required"), default=True)
    if not payload["ref"] and not payload["description"]:
        return None
    return StepOutputSpec(**_filter_dataclass_payload(StepOutputSpec, payload))


def _coerce_step_verification_result(value: Any) -> StepVerificationResult | None:
    if isinstance(value, StepVerificationResult):
        return value
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    step_id = str(payload.get("step_id", "") or "").strip()
    if not step_id:
        return None
    payload["step_id"] = step_id
    payload["step_run_id"] = str(payload.get("step_run_id", "") or "")
    payload["passed"] = _coerce_bool(payload.get("passed"), default=False)
    payload["failed_criteria"] = _coerce_string_list(payload.get("failed_criteria"))
    results = json_safe_value(payload.get("verifier_results") or [])
    payload["verifier_results"] = results if isinstance(results, list) else []
    payload["evidence_artifact_id"] = str(payload.get("evidence_artifact_id", "") or "")
    return StepVerificationResult(**_filter_dataclass_payload(StepVerificationResult, payload))


def _coerce_step_evidence_artifact(value: Any, *, step_id: str | None = None) -> StepEvidenceArtifact | None:
    if isinstance(value, StepEvidenceArtifact):
        return value
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    resolved_step_id = str(payload.get("step_id", "") or step_id or "").strip()
    if not resolved_step_id:
        return None
    payload["step_id"] = resolved_step_id
    payload["step_run_id"] = str(payload.get("step_run_id", "") or "")
    payload["attempt"] = max(1, _coerce_int(payload.get("attempt"), default=1))
    payload["summary"] = str(payload.get("summary", "") or "")
    payload["artifact_ids"] = _coerce_string_list(payload.get("artifact_ids"))
    payload["files_touched"] = _coerce_string_list(payload.get("files_touched"))
    payload["decisions"] = _coerce_string_list(payload.get("decisions"))
    results = json_safe_value(payload.get("verifier_results") or [])
    payload["verifier_results"] = results if isinstance(results, list) else []
    payload["tool_operation_ids"] = _coerce_string_list(payload.get("tool_operation_ids"))
    payload["created_at"] = _coerce_timestamp_string(payload.get("created_at"))
    return StepEvidenceArtifact(**_filter_dataclass_payload(StepEvidenceArtifact, payload))


def _coerce_execution_plan(value: Any) -> Any | None:
    if isinstance(value, ExecutionPlan):
        return value
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    plan_id = str(payload.get("plan_id", "") or "").strip()
    goal = str(payload.get("goal", "") or "").strip()
    if not plan_id and not goal:
        return None
    payload["plan_id"] = plan_id or f"plan-{datetime.now(timezone.utc).strftime('%H%M%S')}"
    payload["goal"] = goal or plan_id
    payload["summary"] = str(payload.get("summary", "") or "")
    payload["inputs"] = _coerce_string_list(payload.get("inputs"))
    payload["outputs"] = _coerce_string_list(payload.get("outputs"))
    payload["constraints"] = _coerce_string_list(payload.get("constraints"))
    payload["acceptance_criteria"] = _coerce_string_list(payload.get("acceptance_criteria"))
    payload["implementation_plan"] = _coerce_string_list(payload.get("implementation_plan"))
    payload["claim_refs"] = _coerce_string_list(payload.get("claim_refs"))
    payload["status"] = str(payload.get("status", "draft") or "draft")
    payload["requested_output_path"] = (
        None if payload.get("requested_output_path") in (None, "") else str(payload.get("requested_output_path"))
    )
    payload["requested_output_format"] = (
        None if payload.get("requested_output_format") in (None, "") else str(payload.get("requested_output_format"))
    )
    payload["approved"] = bool(payload.get("approved", False))
    payload["steps"] = [
        step
        for item in _coerce_list_payload(payload.get("steps"))
        if (step := _coerce_plan_step(item)) is not None
    ]
    payload["created_at"] = _coerce_timestamp_string(payload.get("created_at"))
    payload["updated_at"] = _coerce_timestamp_string(payload.get("updated_at"))
    return ExecutionPlan(**_filter_dataclass_payload(ExecutionPlan, payload))


def _coerce_plan_interrupt(value: Any) -> Any | None:
    if isinstance(value, PlanInterrupt):
        return value
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    payload["kind"] = str(payload.get("kind", "plan_execute_approval") or "plan_execute_approval")
    payload["question"] = str(payload.get("question", "Plan ready. Execute it now?") or "Plan ready. Execute it now?")
    payload["plan_id"] = str(payload.get("plan_id", "") or "")
    payload["approved"] = bool(payload.get("approved", False))
    payload["response_mode"] = str(payload.get("response_mode", "yes/no/revise") or "yes/no/revise")
    return PlanInterrupt(**_filter_dataclass_payload(PlanInterrupt, payload))




def _coerce_memory_entry(value: Any, *, current_step: int, current_phase: str) -> Any | None:
    if isinstance(value, MemoryEntry):
        return value
    if not isinstance(value, dict):
        if value is None:
            return None
        return MemoryEntry(
            content=str(value),
            created_at_step=current_step,
            created_phase=current_phase,
        )
    payload = dict(value)
    content = str(payload.get("content", "") or "")
    if not content:
        return None
    created_at_step = _coerce_int(payload.get("created_at_step"), default=current_step)
    created_phase = str(payload.get("created_phase", "") or current_phase)
    freshness = str(payload.get("freshness", "") or "current")
    confidence_raw = payload.get("confidence")
    confidence = None
    if confidence_raw not in (None, ""):
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = None
    return MemoryEntry(
        content=content,
        created_at_step=created_at_step,
        created_phase=created_phase,
        freshness=freshness,
        confidence=confidence,
    )
