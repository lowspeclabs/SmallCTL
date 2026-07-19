from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models.conversation import ConversationMessage
from .recovery_schema import FailureEvent, ReflectionMemory, SubtaskLedger
from .redaction import redact_sensitive_data
from .state_flow import LoopStateFlowMixin
from .state_schema import (
    ArtifactRecord,
    ArtifactSnippet,
    ChallengeProgressState,
    ClaimRecord,
    ContextBrief,
    DecisionRecord,
    EpisodicSummary,
    EvidenceRecord,
    ExecutionPlan,
    ExperienceMemory,
    MemoryEntry,
    PlanInterrupt,
    PlanStep,
    PromptBudgetSnapshot,
    ReasoningGraph,
    RunBrief,
    StepEvidenceArtifact,
    StepOutputSpec,
    StepVerificationResult,
    StepVerifierSpec,
    TurnBundle,
    WorkingMemory,
    WriteSession,
)
from .state_coercion import (
    _coerce_write_session,
    _coerce_reasoning_graph,
    _coerce_run_brief,
    _coerce_conversation_message,
    _coerce_working_memory,
    _coerce_execution_plan,
    _coerce_plan_interrupt,
    _coerce_step_evidence_artifact,
    _coerce_step_verification_result,
    _coerce_artifact_record,
    _coerce_episodic_summary,
    _coerce_challenge_progress_state,
    _coerce_context_brief,
    _coerce_prompt_budget,
    _coerce_background_process_record,
    _coerce_tool_envelope_payload,
    _coerce_tool_execution_record,
    _coerce_pending_interrupt_payload,
    _coerce_experience_memory,
    _coerce_turn_bundle,
    _coerce_memory_entry_list,
)
from .state_session_records import _coerce_active_write_sessions_by_path
from .recovery_coercion import (
    _coerce_failure_event,
    _coerce_reflection_memory,
    _coerce_subtask_ledger,
)
from .state_memory import align_memory_entries, memory_entry_is_stale, _trim_recent_messages
from .state_support import (
    LOOP_STATE_SCHEMA_VERSION,
    _coerce_bool,
    _coerce_conversation_message_payload,
    _coerce_dict_payload,
    _coerce_float,
    _coerce_json_dict_payload,
    _coerce_int,
    _coerce_int_map,
    _coerce_list_payload,
    _coerce_string_list,
    _coerce_string_map,
    _coerce_timestamp_string,
    _coerce_write_section_ranges,
    _filter_dataclass_payload,
    _migrate_loop_state_payload,
    clip_string_list,
    clip_text_value,
    json_safe_value,
    normalize_intent_label,
)


@dataclass
class LoopState(LoopStateFlowMixin):
    schema_version: int = LOOP_STATE_SCHEMA_VERSION
    current_phase: str = "explore"
    thread_id: str = ""
    step_count: int = 0
    token_usage: int = 0
    elapsed_seconds: float = 0.0
    inactive_steps: int = 0
    recent_errors: list[str] = field(default_factory=list)
    scratchpad: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] | None = None
    recent_messages: list[ConversationMessage] = field(default_factory=list)
    transcript_messages: list[ConversationMessage] = field(default_factory=list)
    run_brief: RunBrief = field(default_factory=RunBrief)
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    reasoning_graph: ReasoningGraph = field(default_factory=ReasoningGraph)
    acceptance_ledger: dict[str, str] = field(default_factory=dict)
    acceptance_waivers: list[str] = field(default_factory=list)
    acceptance_waived: bool = False
    last_verifier_verdict: dict[str, Any] | None = None
    challenge_progress: ChallengeProgressState = field(default_factory=ChallengeProgressState)
    last_failure_class: str = ""
    failure_events: list[FailureEvent] = field(default_factory=list)
    reflexion_memory: list[ReflectionMemory] = field(default_factory=list)
    subtask_ledger: SubtaskLedger | None = None
    files_changed_this_cycle: list[str] = field(default_factory=list)
    repair_cycle_id: str = ""
    stagnation_counters: dict[str, int] = field(default_factory=dict)
    draft_plan: ExecutionPlan | None = None
    active_plan: ExecutionPlan | None = None
    plan_resolved: bool = False
    plan_artifact_id: str = ""
    planning_mode_enabled: bool = False
    planner_requested_output_path: str = ""
    planner_requested_output_format: str = ""
    planner_resume_target_mode: str = "loop"
    planner_interrupt: PlanInterrupt | None = None
    plan_execution_mode: bool = False
    active_step_id: str = ""
    active_step_run_id: str = ""
    step_sandbox_history: list[ConversationMessage] = field(default_factory=list)
    step_evidence: dict[str, StepEvidenceArtifact] = field(default_factory=dict)
    step_verification_result: StepVerificationResult | None = None
    artifacts: dict[str, ArtifactRecord] = field(default_factory=dict)
    episodic_summaries: list[EpisodicSummary] = field(default_factory=list)
    context_briefs: list[ContextBrief] = field(default_factory=list)
    turn_bundles: list[TurnBundle] = field(default_factory=list)
    prompt_budget: PromptBudgetSnapshot = field(default_factory=PromptBudgetSnapshot)
    retrieval_cache: list[str] = field(default_factory=list)
    task_mode: str = ""
    active_intent: str = ""
    secondary_intents: list[str] = field(default_factory=list)
    intent_tags: list[str] = field(default_factory=list)
    warm_experiences: list[ExperienceMemory] = field(default_factory=list)
    retrieved_experience_ids: list[str] = field(default_factory=list)
    tool_execution_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_execution_records_limit: int = 2000
    pending_interrupt: dict[str, Any] | None = None
    background_processes: dict[str, dict[str, Any]] = field(default_factory=dict)
    cwd: str = field(default_factory=lambda: str(Path.cwd()))
    active_tool_profiles: list[str] = field(default_factory=lambda: ["core"])
    task_exposed_tools: set[str] = field(default_factory=set)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    recent_message_limit: int = 6
    transcript_message_limit: int = 5000
    reasoning_graph_max_records_per_lane: int = 5000
    artifact_limit: int = 5000
    last_completion_tokens: int = 0
    tool_history: list[str] = field(default_factory=list)
    write_session: WriteSession | None = None
    active_write_sessions_by_path: dict[str, WriteSession] = field(default_factory=dict)
    task_received_at: str = ""
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("smallctl.state"))
    
    @property
    def state(self) -> "LoopState":
        return self

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def append_message(self, message: ConversationMessage) -> None:
        self.transcript_messages.append(message)
        self.trim_transcript_messages()
        self.recent_messages.append(message)
        self.recent_messages = _trim_recent_messages(
            self.recent_messages,
            limit=self.recent_message_limit,
        )
        self.touch()

    def append_tool_history(self, fingerprint: str, limit: int = 15) -> None:
        self.tool_history.append(fingerprint)
        if len(self.tool_history) > limit:
            self.tool_history = self.tool_history[-limit:]
        self.touch()

    def trim_transcript_messages(self, limit: int | None = None) -> None:
        limit = limit if limit is not None and limit > 0 else self.transcript_message_limit
        if limit <= 0:
            return
        if len(self.transcript_messages) > limit:
            self.transcript_messages = self.transcript_messages[-limit:]

    def _protected_artifact_ids(self) -> set[str]:
        protected: set[str] = set()
        active_session_ids: set[str] = set()
        if self.write_session is not None:
            active_session_ids.add(str(getattr(self.write_session, "write_session_id", "") or ""))
        for session in (self.active_write_sessions_by_path or {}).values():
            if session is not None:
                active_session_ids.add(str(getattr(session, "write_session_id", "") or ""))
        active_session_ids.discard("")
        for artifact_id, record in (self.artifacts or {}).items():
            if str(getattr(record, "session_id", "") or "") in active_session_ids:
                protected.add(artifact_id)
        for record in (self.tool_execution_records or {}).values():
            if not isinstance(record, dict):
                continue
            for key in ("artifact_id", "evidence_artifact_id"):
                value = record.get(key)
                if isinstance(value, str) and value:
                    protected.add(value)
            evidence_record = record.get("evidence_record")
            if isinstance(evidence_record, dict):
                value = evidence_record.get("artifact_id")
                if isinstance(value, str) and value:
                    protected.add(value)
            result = record.get("result")
            if isinstance(result, dict):
                metadata = result.get("metadata") or {}
                if isinstance(metadata, dict):
                    value = metadata.get("artifact_id")
                    if isinstance(value, str) and value:
                        protected.add(value)
        if getattr(self, "plan_artifact_id", ""):
            protected.add(self.plan_artifact_id)
        for evidence in (self.step_evidence or {}).values():
            for artifact_id in getattr(evidence, "artifact_ids", []) or []:
                if artifact_id:
                    protected.add(artifact_id)
        if self.step_verification_result is not None:
            value = getattr(self.step_verification_result, "evidence_artifact_id", "")
            if value:
                protected.add(str(value))
        for summary in (self.episodic_summaries or []):
            for artifact_id in getattr(summary, "artifact_ids", []) or []:
                if artifact_id:
                    protected.add(artifact_id)
            full_id = getattr(summary, "full_summary_artifact_id", "")
            if full_id:
                protected.add(str(full_id))
        for brief in (self.context_briefs or []):
            for artifact_id in getattr(brief, "artifact_ids", []) or []:
                if artifact_id:
                    protected.add(artifact_id)
            full_id = getattr(brief, "full_artifact_id", "")
            if full_id:
                protected.add(str(full_id))
        for bundle in (self.turn_bundles or []):
            for artifact_id in getattr(bundle, "artifact_ids", []) or []:
                if artifact_id:
                    protected.add(artifact_id)
        for item in (self.retrieval_cache or []):
            if isinstance(item, str) and item.startswith("A") and item[1:].isdigit():
                protected.add(item)
        return protected

    def _protected_tool_execution_record_ids(self) -> set[str]:
        protected: set[str] = set()
        protected_artifacts = self._protected_artifact_ids()

        active_plan_ids: set[str] = set()
        for plan in (self.active_plan, self.draft_plan):
            plan_id = str(getattr(plan, "plan_id", "") or "").strip()
            if plan_id:
                active_plan_ids.add(plan_id)

        recent_evidence_ids: set[str] = set()
        for evidence in getattr(self.reasoning_graph, "evidence_records", []) or []:
            evidence_id = str(getattr(evidence, "evidence_id", "") or "").strip()
            if evidence_id:
                recent_evidence_ids.add(evidence_id)

        for evidence in (self.step_evidence or {}).values():
            for operation_id in getattr(evidence, "tool_operation_ids", []) or []:
                if operation_id:
                    protected.add(str(operation_id))

        for operation_id, record in (self.tool_execution_records or {}).items():
            if not isinstance(record, dict):
                continue
            if str(record.get("plan_id") or "").strip() in active_plan_ids:
                protected.add(operation_id)
            if str(record.get("artifact_id") or "").strip() in protected_artifacts:
                protected.add(operation_id)
            if str(record.get("evidence_id") or "").strip() in recent_evidence_ids:
                protected.add(operation_id)

        return protected

    def trim_tool_execution_records(self, limit: int | None = None) -> None:
        limit = limit if limit is not None and limit > 0 else self.tool_execution_records_limit
        if limit <= 0:
            return
        if len(self.tool_execution_records) <= limit:
            return
        protected = self._protected_tool_execution_record_ids()

        def _sort_key(item: tuple[str, Any]) -> tuple[int, str]:
            operation_id, record = item
            step_count = 0
            if isinstance(record, dict):
                step_count = int(record.get("step_count") or 0)
            return (step_count, operation_id)

        sorted_items = sorted(self.tool_execution_records.items(), key=_sort_key, reverse=True)
        kept: dict[str, Any] = {}
        for operation_id, record in sorted_items:
            if operation_id in protected or len(kept) < limit:
                kept[operation_id] = record
        self.tool_execution_records = kept

    def trim_artifacts(self, limit: int | None = None, artifact_store: Any = None) -> None:
        limit = limit if limit is not None and limit > 0 else self.artifact_limit
        if limit <= 0:
            return
        if len(self.artifacts) <= limit:
            return
        protected = self._protected_artifact_ids()

        def _sort_key(item: tuple[str, Any]) -> tuple[str, str]:
            artifact_id, record = item
            created_at = str(getattr(record, "created_at", "") or "")
            return (created_at, artifact_id)

        sorted_items = sorted(self.artifacts.items(), key=_sort_key, reverse=True)
        kept: dict[str, Any] = {}
        dropped_inline: list[tuple[str, Any, Any]] = []
        for artifact_id, record in sorted_items:
            if artifact_id in protected or len(kept) < limit:
                kept[artifact_id] = record
            else:
                inline = getattr(record, "inline_content", None)
                if inline is not None and artifact_store is not None:
                    dropped_inline.append((artifact_id, record, inline))
                elif inline is not None:
                    kept[artifact_id] = record
                else:
                    pass
        for artifact_id, record, inline in dropped_inline:
            try:
                tool_name_raw = getattr(record, "tool_name", None)
                tool_name = str(tool_name_raw) if tool_name_raw else None
                artifact_store.persist_generated_text(
                    kind=str(getattr(record, "kind", "evicted") or "evicted"),
                    source=str(getattr(record, "source", "") or "state_trim_artifacts"),
                    content=str(inline),
                    summary=str(getattr(record, "summary", "") or f"Evicted artifact {artifact_id}"),
                    preview_text=getattr(record, "preview_text", None),
                    metadata=dict(getattr(record, "metadata", {}) or {}),
                    tool_name=tool_name,
                    session_id=str(getattr(record, "session_id", "") or ""),
                    tool_call_id=str(getattr(record, "tool_call_id", "") or ""),
                )
            except Exception:
                kept[artifact_id] = record
        self.artifacts = kept

    def to_dict(self, artifact_store: Any = None) -> dict[str, Any]:
        started = time.perf_counter()
        self.trim_transcript_messages()
        self.reasoning_graph.trim_records(self.reasoning_graph_max_records_per_lane)
        self.trim_artifacts(artifact_store=artifact_store)
        self.trim_tool_execution_records()
        serialized_messages = [
            json_safe_value(m.to_dict(include_retrieval_safe_text=True))
            for m in self.recent_messages
        ]
        transcript_source = self.transcript_messages or self.recent_messages
        serialized_transcript_messages = [
            json_safe_value(m.to_dict(include_retrieval_safe_text=True))
            for m in transcript_source
        ]
        payload = {
            "schema_version": LOOP_STATE_SCHEMA_VERSION,
            "current_phase": self.current_phase,
            "thread_id": self.thread_id,
            "step_count": self.step_count,
            "token_usage": self.token_usage,
            "elapsed_seconds": self.elapsed_seconds,
            "inactive_steps": self.inactive_steps,
            "recent_errors": json_safe_value(self.recent_errors),
            "strategy": json_safe_value(self.strategy),
            "scratchpad": redact_sensitive_data(json_safe_value(self.scratchpad)),
            "recent_messages": serialized_messages,
            "transcript_messages": serialized_transcript_messages,
            "run_brief": json_safe_value(self.run_brief),
            "working_memory": json_safe_value(self.working_memory),
            "reasoning_graph": json_safe_value(self.reasoning_graph),
            "acceptance_ledger": json_safe_value(self.acceptance_ledger),
            "acceptance_waivers": json_safe_value(self.acceptance_waivers),
            "acceptance_waived": self.acceptance_waived,
            "last_verifier_verdict": json_safe_value(self.last_verifier_verdict),
            "challenge_progress": json_safe_value(self.challenge_progress),
            "last_failure_class": self.last_failure_class,
            "failure_events": json_safe_value(self.failure_events),
            "reflexion_memory": json_safe_value(self.reflexion_memory),
            "subtask_ledger": json_safe_value(self.subtask_ledger),
            "files_changed_this_cycle": json_safe_value(self.files_changed_this_cycle),
            "repair_cycle_id": self.repair_cycle_id,
            "stagnation_counters": json_safe_value(self.stagnation_counters),
            "draft_plan": json_safe_value(self.draft_plan),
            "active_plan": json_safe_value(self.active_plan),
            "plan_resolved": self.plan_resolved,
            "plan_artifact_id": self.plan_artifact_id,
            "planning_mode_enabled": self.planning_mode_enabled,
            "planner_requested_output_path": self.planner_requested_output_path,
            "planner_requested_output_format": self.planner_requested_output_format,
            "planner_resume_target_mode": self.planner_resume_target_mode,
            "planner_interrupt": json_safe_value(self.planner_interrupt),
            "plan_execution_mode": self.plan_execution_mode,
            "active_step_id": self.active_step_id,
            "active_step_run_id": self.active_step_run_id,
            "step_sandbox_history": [
                json_safe_value(m.to_dict(include_retrieval_safe_text=True))
                for m in self.step_sandbox_history
            ],
            "step_evidence": json_safe_value(self.step_evidence),
            "step_verification_result": json_safe_value(self.step_verification_result),
            "artifacts": json_safe_value(self.artifacts),
            "episodic_summaries": json_safe_value(self.episodic_summaries),
            "context_briefs": json_safe_value(self.context_briefs),
            "turn_bundles": json_safe_value(self.turn_bundles),
            "prompt_budget": json_safe_value(self.prompt_budget),
            "retrieval_cache": json_safe_value(self.retrieval_cache),
            "task_mode": self.task_mode,
            "active_intent": self.active_intent,
            "secondary_intents": json_safe_value(self.secondary_intents),
            "intent_tags": json_safe_value(self.intent_tags),
            "warm_experiences": json_safe_value(self.warm_experiences),
            "retrieved_experience_ids": json_safe_value(self.retrieved_experience_ids),
            "tool_execution_records": redact_sensitive_data({
                str(key): _coerce_tool_execution_record(value, operation_id=str(key))
                for key, value in self.tool_execution_records.items()
            }),
            "pending_interrupt": _coerce_pending_interrupt_payload(self.pending_interrupt),
            "background_processes": {
                str(key): _coerce_background_process_record(value, job_id=str(key))
                for key, value in self.background_processes.items()
            },
            "cwd": self.cwd,
            "active_tool_profiles": json_safe_value(self.active_tool_profiles),
            "task_exposed_tools": json_safe_value(sorted(self.task_exposed_tools)),
            "tool_history": json_safe_value(self.tool_history),
            "write_session": self.write_session.to_dict() if self.write_session else None,
            "active_write_sessions_by_path": {
                key: (session.to_dict() if session else None)
                for key, session in (self.active_write_sessions_by_path or {}).items()
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "recent_message_limit": self.recent_message_limit,
            "transcript_message_limit": self.transcript_message_limit,
            "reasoning_graph_max_records_per_lane": self.reasoning_graph_max_records_per_lane,
            "artifact_limit": self.artifact_limit,
            "tool_execution_records_limit": self.tool_execution_records_limit,
             "last_completion_tokens": self.last_completion_tokens,
        }
        # L19: the legacy "conversation_history" key is no longer written; the
        # transcript is serialized once under "transcript_messages". from_dict
        # still accepts old payloads that only carry "conversation_history".
        _log_loop_state_serialization(
            self.log,
            payload,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoopState":
        incoming_version = _coerce_int(data.get("schema_version"), default=0)
        migrated = _migrate_loop_state_payload(dict(data), incoming_version=incoming_version)
        raw_recent_messages = migrated.get("recent_messages")
        raw_transcript_messages = migrated.get("transcript_messages")
        if raw_transcript_messages is None:
            raw_transcript_messages = _coerce_list_payload(migrated.get("conversation_history"))
        else:
            raw_transcript_messages = _coerce_list_payload(raw_transcript_messages)
        transcript_messages = [
            message
            for msg in raw_transcript_messages
            if (message := _coerce_conversation_message(msg)) is not None
        ]
        if raw_recent_messages is None:
            raw_recent_messages = raw_transcript_messages
        else:
            raw_recent_messages = _coerce_list_payload(raw_recent_messages)
        recent_messages = [
            message
            for msg in raw_recent_messages
            if (message := _coerce_conversation_message(msg)) is not None
        ]
        if not transcript_messages:
            transcript_messages = list(recent_messages)
        raw = dict(migrated)
        raw["schema_version"] = LOOP_STATE_SCHEMA_VERSION
        raw.pop("conversation_history", None)
        stored_recent_limit = migrated.get("recent_message_limit")
        default_recent_limit = max(len(recent_messages), 6)
        recent_limit = max(_coerce_int(stored_recent_limit, default=default_recent_limit), 1)
        recent_messages = _trim_recent_messages(recent_messages, limit=recent_limit)
        raw["current_phase"] = str(migrated.get("current_phase", "explore") or "explore")
        raw["thread_id"] = str(migrated.get("thread_id", "") or "")
        raw["step_count"] = _coerce_int(migrated.get("step_count"), default=0)
        raw["token_usage"] = _coerce_int(migrated.get("token_usage"), default=0)
        raw["elapsed_seconds"] = _coerce_float(migrated.get("elapsed_seconds"), default=0.0)
        raw["inactive_steps"] = _coerce_int(migrated.get("inactive_steps"), default=0)
        raw["recent_errors"] = _coerce_string_list(migrated.get("recent_errors"))
        raw["strategy"] = _coerce_json_dict_payload(migrated.get("strategy"))
        raw["recent_messages"] = recent_messages
        raw["transcript_messages"] = transcript_messages
        raw["recent_message_limit"] = recent_limit
        raw["transcript_message_limit"] = max(_coerce_int(migrated.get("transcript_message_limit"), default=5000), 1)
        raw["reasoning_graph_max_records_per_lane"] = max(
            _coerce_int(migrated.get("reasoning_graph_max_records_per_lane"), default=5000), 1
        )
        raw["artifact_limit"] = max(_coerce_int(migrated.get("artifact_limit"), default=5000), 1)
        raw["tool_execution_records_limit"] = max(
            _coerce_int(migrated.get("tool_execution_records_limit"), default=2000), 1
        )
        raw["run_brief"] = _coerce_run_brief(migrated.get("run_brief"))
        raw["working_memory"] = _coerce_working_memory(
            migrated.get("working_memory"),
            current_step=raw["step_count"],
            current_phase=raw["current_phase"],
        )
        raw["reasoning_graph"] = _coerce_reasoning_graph(migrated.get("reasoning_graph"))
        raw["acceptance_ledger"] = _coerce_string_map(migrated.get("acceptance_ledger"))
        raw["acceptance_waivers"] = _coerce_string_list(migrated.get("acceptance_waivers"))
        raw["acceptance_waived"] = bool(migrated.get("acceptance_waived", False))
        raw["last_verifier_verdict"] = _coerce_json_dict_payload(migrated.get("last_verifier_verdict"))
        raw["challenge_progress"] = _coerce_challenge_progress_state(migrated.get("challenge_progress"))
        raw["last_failure_class"] = str(migrated.get("last_failure_class", "") or "")
        raw["failure_events"] = [
            event
            for item in _coerce_list_payload(migrated.get("failure_events"))
            if (event := _coerce_failure_event(item)) is not None
        ]
        raw["reflexion_memory"] = [
            reflection
            for item in _coerce_list_payload(migrated.get("reflexion_memory"))
            if (reflection := _coerce_reflection_memory(item)) is not None
        ]
        raw["subtask_ledger"] = _coerce_subtask_ledger(migrated.get("subtask_ledger"))
        raw["files_changed_this_cycle"] = _coerce_string_list(migrated.get("files_changed_this_cycle"))
        raw["repair_cycle_id"] = str(migrated.get("repair_cycle_id", "") or "")
        raw["stagnation_counters"] = _coerce_int_map(migrated.get("stagnation_counters"))
        raw["draft_plan"] = _coerce_execution_plan(migrated.get("draft_plan"))
        raw["active_plan"] = _coerce_execution_plan(migrated.get("active_plan"))
        raw["plan_resolved"] = bool(migrated.get("plan_resolved", False))
        raw["plan_artifact_id"] = str(migrated.get("plan_artifact_id", "") or "")
        raw["planning_mode_enabled"] = bool(migrated.get("planning_mode_enabled", False))
        raw["planner_requested_output_path"] = str(migrated.get("planner_requested_output_path", "") or "")
        raw["planner_requested_output_format"] = str(migrated.get("planner_requested_output_format", "") or "")
        raw["planner_resume_target_mode"] = str(migrated.get("planner_resume_target_mode", "loop") or "loop")
        raw["planner_interrupt"] = _coerce_plan_interrupt(migrated.get("planner_interrupt"))
        raw["plan_execution_mode"] = _coerce_bool(migrated.get("plan_execution_mode"), default=False)
        raw["active_step_id"] = str(migrated.get("active_step_id", "") or "")
        raw["active_step_run_id"] = str(migrated.get("active_step_run_id", "") or "")
        raw["step_sandbox_history"] = [
            message
            for msg in _coerce_list_payload(migrated.get("step_sandbox_history"))
            if (message := _coerce_conversation_message(msg)) is not None
        ]
        raw["step_evidence"] = {
            str(key): evidence
            for key, value in _coerce_dict_payload(migrated.get("step_evidence")).items()
            if (evidence := _coerce_step_evidence_artifact(value, step_id=str(key))) is not None
        }
        raw["step_verification_result"] = _coerce_step_verification_result(
            migrated.get("step_verification_result")
        )
        raw["artifacts"] = {
            key: _coerce_artifact_record(value, artifact_id=key)
            for key, value in _coerce_dict_payload(migrated.get("artifacts")).items()
        }
        raw["episodic_summaries"] = [
            _coerce_episodic_summary(item)
            for item in _coerce_list_payload(migrated.get("episodic_summaries"))
        ]
        raw["context_briefs"] = [
            _coerce_context_brief(item)
            for item in _coerce_list_payload(migrated.get("context_briefs"))
        ]
        raw["turn_bundles"] = [
            _coerce_turn_bundle(item)
            for item in _coerce_list_payload(migrated.get("turn_bundles"))
        ]
        raw["prompt_budget"] = _coerce_prompt_budget(migrated.get("prompt_budget"))
        raw["retrieval_cache"] = _coerce_string_list(migrated.get("retrieval_cache"))
        raw["task_mode"] = str(migrated.get("task_mode", "") or "")
        raw["active_intent"] = str(migrated.get("active_intent", "") or "")
        raw["secondary_intents"] = _coerce_string_list(migrated.get("secondary_intents"))
        raw["intent_tags"] = _coerce_string_list(migrated.get("intent_tags"))
        raw["warm_experiences"] = [
            _coerce_experience_memory(item)
            for item in _coerce_list_payload(migrated.get("warm_experiences"))
        ]
        raw["retrieved_experience_ids"] = _coerce_string_list(migrated.get("retrieved_experience_ids"))
        raw["tool_execution_records"] = {
            str(key): _coerce_tool_execution_record(value, operation_id=str(key))
            for key, value in _coerce_dict_payload(migrated.get("tool_execution_records")).items()
        }
        raw["pending_interrupt"] = _coerce_pending_interrupt_payload(migrated.get("pending_interrupt"))
        raw["scratchpad"] = _coerce_json_dict_payload(migrated.get("scratchpad"))
        raw["background_processes"] = {
            str(key): _coerce_background_process_record(value, job_id=str(key))
            for key, value in _coerce_dict_payload(migrated.get("background_processes")).items()
        }
        raw["cwd"] = str(migrated.get("cwd", Path.cwd()) or Path.cwd())
        raw["active_tool_profiles"] = _coerce_string_list(migrated.get("active_tool_profiles")) or ["core"]
        raw["task_exposed_tools"] = set(_coerce_string_list(migrated.get("task_exposed_tools")))
        raw["tool_history"] = _coerce_string_list(migrated.get("tool_history"))
        raw["write_session"] = _coerce_write_session(migrated.get("write_session"))
        raw["active_write_sessions_by_path"] = _coerce_active_write_sessions_by_path(
            migrated.get("active_write_sessions_by_path")
        )
        raw["created_at"] = _coerce_timestamp_string(migrated.get("created_at"))
        raw["updated_at"] = _coerce_timestamp_string(migrated.get("updated_at"))
        raw["last_completion_tokens"] = _coerce_int(migrated.get("last_completion_tokens"), default=0)
        state = cls(**_filter_dataclass_payload(cls, raw))
        state.sync_plan_mirror()
        return state


def _log_loop_state_serialization(
    log: logging.Logger,
    payload: dict[str, Any],
    *,
    elapsed_ms: float,
) -> None:
    try:
        tool_records = payload.get("tool_execution_records")
        tool_records_bytes = _json_size_bytes(tool_records)
        recent_messages = payload.get("recent_messages") if isinstance(payload.get("recent_messages"), list) else []
        transcript_messages = payload.get("transcript_messages") if isinstance(payload.get("transcript_messages"), list) else []
        log.debug(
            "loop_state_serialization %s",
            {
                "elapsed_ms": round(elapsed_ms, 2),
                "tool_execution_records_bytes": tool_records_bytes,
                "recent_messages_chars": _message_content_chars(recent_messages),
                "transcript_messages_chars": _message_content_chars(transcript_messages),
                "tool_execution_record_count": len(tool_records) if isinstance(tool_records, dict) else 0,
            },
        )
    except Exception:
        return


def _json_size_bytes(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8"))
    except Exception:
        return len(str(value).encode("utf-8", errors="replace"))


def _message_content_chars(messages: list[Any]) -> int:
    total = 0
    for item in messages:
        if isinstance(item, dict):
            total += len(str(item.get("content") or ""))
    return total
