from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models.conversation import ConversationMessage
from .state_flow import LoopStateFlowMixin
from .state_schema import (
    ArtifactRecord,
    ArtifactSnippet,
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
    _coerce_artifact_record,
    _coerce_episodic_summary,
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
    run_brief: RunBrief = field(default_factory=RunBrief)
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    reasoning_graph: ReasoningGraph = field(default_factory=ReasoningGraph)
    acceptance_ledger: dict[str, str] = field(default_factory=dict)
    acceptance_waivers: list[str] = field(default_factory=list)
    acceptance_waived: bool = False
    last_verifier_verdict: dict[str, Any] | None = None
    last_failure_class: str = ""
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
    pending_interrupt: dict[str, Any] | None = None
    background_processes: dict[str, dict[str, Any]] = field(default_factory=dict)
    cwd: str = field(default_factory=lambda: str(Path.cwd()))
    active_tool_profiles: list[str] = field(default_factory=lambda: ["core"])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    recent_message_limit: int = 6
    last_completion_tokens: int = 0
    tool_history: list[str] = field(default_factory=list)
    write_session: WriteSession | None = None
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("smallctl.state"))
    
    @property
    def state(self) -> "LoopState":
        return self

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def append_message(self, message: ConversationMessage) -> None:
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

    def to_dict(self) -> dict[str, Any]:
        self.reasoning_graph.touch_ids()
        serialized_messages = [
            json_safe_value(m.to_dict(include_retrieval_safe_text=True))
            for m in self.recent_messages
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
            "scratchpad": json_safe_value(self.scratchpad),
            "recent_messages": serialized_messages,
            "run_brief": json_safe_value(self.run_brief),
            "working_memory": json_safe_value(self.working_memory),
            "reasoning_graph": json_safe_value(self.reasoning_graph),
            "acceptance_ledger": json_safe_value(self.acceptance_ledger),
            "acceptance_waivers": json_safe_value(self.acceptance_waivers),
            "acceptance_waived": self.acceptance_waived,
            "last_verifier_verdict": json_safe_value(self.last_verifier_verdict),
            "last_failure_class": self.last_failure_class,
            "files_changed_this_cycle": json_safe_value(self.files_changed_this_cycle),
            "repair_cycle_id": self.repair_cycle_id,
            "stagnation_counters": json_safe_value(self.stagnation_counters),
            "draft_plan": json_safe_value(self.draft_plan),
            "active_plan": json_safe_value(self.active_plan),
            "planning_mode_enabled": self.planning_mode_enabled,
            "planner_requested_output_path": self.planner_requested_output_path,
            "planner_requested_output_format": self.planner_requested_output_format,
            "planner_resume_target_mode": self.planner_resume_target_mode,
            "planner_interrupt": json_safe_value(self.planner_interrupt),
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
            "tool_execution_records": {
                str(key): _coerce_tool_execution_record(value, operation_id=str(key))
                for key, value in self.tool_execution_records.items()
            },
            "pending_interrupt": _coerce_pending_interrupt_payload(self.pending_interrupt),
            "background_processes": {
                str(key): _coerce_background_process_record(value, job_id=str(key))
                for key, value in self.background_processes.items()
            },
            "cwd": self.cwd,
            "active_tool_profiles": json_safe_value(self.active_tool_profiles),
            "tool_history": json_safe_value(self.tool_history),
            "write_session": self.write_session.to_dict() if self.write_session else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "recent_message_limit": self.recent_message_limit,
            "last_completion_tokens": self.last_completion_tokens,
        }
        payload["conversation_history"] = list(serialized_messages)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoopState":
        incoming_version = _coerce_int(data.get("schema_version"), default=0)
        migrated = _migrate_loop_state_payload(dict(data), incoming_version=incoming_version)
        raw_recent_messages = migrated.get("recent_messages")
        if raw_recent_messages is None:
            raw_recent_messages = _coerce_list_payload(migrated.get("conversation_history"))[-6:]
        else:
            raw_recent_messages = _coerce_list_payload(raw_recent_messages)
        recent_messages = [
            message
            for msg in raw_recent_messages
            if (message := _coerce_conversation_message(msg)) is not None
        ]
        raw = dict(migrated)
        raw["schema_version"] = LOOP_STATE_SCHEMA_VERSION
        raw.pop("conversation_history", None)
        recent_limit = max(_coerce_int(migrated.get("recent_message_limit"), default=6), 1)
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
        raw["recent_message_limit"] = recent_limit
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
        raw["last_failure_class"] = str(migrated.get("last_failure_class", "") or "")
        raw["files_changed_this_cycle"] = _coerce_string_list(migrated.get("files_changed_this_cycle"))
        raw["repair_cycle_id"] = str(migrated.get("repair_cycle_id", "") or "")
        raw["stagnation_counters"] = _coerce_int_map(migrated.get("stagnation_counters"))
        raw["draft_plan"] = _coerce_execution_plan(migrated.get("draft_plan"))
        raw["active_plan"] = _coerce_execution_plan(migrated.get("active_plan"))
        raw["planning_mode_enabled"] = bool(migrated.get("planning_mode_enabled", False))
        raw["planner_requested_output_path"] = str(migrated.get("planner_requested_output_path", "") or "")
        raw["planner_requested_output_format"] = str(migrated.get("planner_requested_output_format", "") or "")
        raw["planner_resume_target_mode"] = str(migrated.get("planner_resume_target_mode", "loop") or "loop")
        raw["planner_interrupt"] = _coerce_plan_interrupt(migrated.get("planner_interrupt"))
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
        raw["tool_history"] = _coerce_string_list(migrated.get("tool_history"))
        raw["write_session"] = _coerce_write_session(migrated.get("write_session"))
        raw["created_at"] = _coerce_timestamp_string(migrated.get("created_at"))
        raw["updated_at"] = _coerce_timestamp_string(migrated.get("updated_at"))
        raw["last_completion_tokens"] = _coerce_int(migrated.get("last_completion_tokens"), default=0)
        state = cls(**_filter_dataclass_payload(cls, raw))
        state.sync_plan_mirror()
        return state
