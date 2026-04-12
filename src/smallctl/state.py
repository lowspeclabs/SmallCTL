from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

from .models.conversation import ConversationMessage
from .normalization import (
    coerce_datetime as _coerce_datetime,
    coerce_dict_payload as _coerce_dict_payload,
    coerce_float as _coerce_float,
    coerce_int,
    coerce_json_dict_payload,
    coerce_list_payload as _coerce_list_payload,
    coerce_string_list,
    coerce_timestamp_string as _coerce_timestamp_string,
)

LOOP_STATE_SCHEMA_VERSION = 2


def _coerce_int(value: Any, *, default: int = 0) -> int:
    return coerce_int(value, default=default)


def _coerce_string_list(value: Any) -> list[str]:
    return coerce_string_list(value)


def _coerce_json_dict_payload(value: Any) -> dict[str, Any]:
    return coerce_json_dict_payload(value, json_safe_func=json_safe_value)


def _coerce_write_section_ranges(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, int]] = {}
    for key, item in value.items():
        if not isinstance(item, dict):
            continue
        start = _coerce_int(item.get("start"), default=-1)
        end = _coerce_int(item.get("end"), default=-1)
        if start < 0 or end < start:
            continue
        normalized[str(key)] = {"start": start, "end": end}
    return normalized


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_intent_label(value: Any) -> str:
    normalized = str(value or "").strip()
    if normalized.startswith("use_") and len(normalized) > 4:
        return f"requested_{normalized[4:]}"
    return normalized


@dataclass
class RunBrief:
    original_task: str = ""
    task_contract: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    implementation_plan: list[str] = field(default_factory=list)
    current_phase_objective: str = ""


@dataclass
class WorkingMemory:
    current_goal: str = ""
    plan: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    known_facts: list[str] = field(default_factory=list)
    known_fact_meta: list["MemoryEntry"] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    failure_meta: list["MemoryEntry"] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    next_action_meta: list["MemoryEntry"] = field(default_factory=list)


@dataclass
class PlanStep:
    step_id: str
    title: str
    description: str = ""
    status: str = "pending"
    notes: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    substeps: list["PlanStep"] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    claim_refs: list[str] = field(default_factory=list)

    def iter_steps(self) -> list["PlanStep"]:
        steps = [self]
        for substep in self.substeps:
            steps.extend(substep.iter_steps())
        return steps

    def find_step(self, step_id: str) -> "PlanStep" | None:
        if self.step_id == step_id:
            return self
        for substep in self.substeps:
            found = substep.find_step(step_id)
            if found is not None:
                return found
        return None

    def compact_label(self) -> str:
        return f"[{self.status}] {self.step_id} {self.title}".strip()


@dataclass
class ExecutionPlan:
    plan_id: str
    goal: str
    summary: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    implementation_plan: list[str] = field(default_factory=list)
    claim_refs: list[str] = field(default_factory=list)
    steps: list[PlanStep] = field(default_factory=list)
    status: str = "draft"
    requested_output_path: str | None = None
    requested_output_format: str | None = None
    approved: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def iter_steps(self) -> list[PlanStep]:
        steps: list[PlanStep] = []
        for step in self.steps:
            steps.extend(step.iter_steps())
        return steps

    def find_step(self, step_id: str) -> PlanStep | None:
        for step in self.steps:
            found = step.find_step(step_id)
            if found is not None:
                return found
        return None

    def active_step(self) -> PlanStep | None:
        for step in self.iter_steps():
            if step.status == "in_progress":
                return step
        for step in self.iter_steps():
            if step.status == "pending":
                return step
        return None

    def compact_lines(self) -> list[str]:
        lines = [f"{self.plan_id}: {self.goal}".strip()]
        if self.summary:
            lines.append(f"summary: {self.summary}")
        spec = self.spec_summary()
        if spec:
            lines.append(f"spec: {spec}")
        for step in self.steps:
            lines.extend(_compact_plan_step_lines(step))
        if self.requested_output_path:
            lines.append(f"export: {self.requested_output_path}")
        return lines

    def spec_summary(self) -> str:
        parts: list[str] = []
        if self.inputs:
            parts.append("inputs=" + ", ".join(self.inputs))
        if self.outputs:
            parts.append("outputs=" + ", ".join(self.outputs))
        if self.constraints:
            parts.append("constraints=" + ", ".join(self.constraints))
        if self.acceptance_criteria:
            parts.append("acceptance=" + "; ".join(self.acceptance_criteria))
        if self.implementation_plan:
            parts.append("implementation=" + " | ".join(self.implementation_plan))
        if self.claim_refs:
            parts.append("claims=" + ", ".join(self.claim_refs))
        return " ; ".join(parts)


@dataclass
class EvidenceRecord:
    evidence_id: str
    kind: str = "observation"
    statement: str = ""
    phase: str = ""
    tool_name: str = ""
    operation_id: str = ""
    artifact_id: str = ""
    source: str = ""
    evidence_type: str = "direct_observation"
    confidence: float = 0.0
    negative: bool = False
    replayed: bool = False
    claim_ids: list[str] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionRecord:
    decision_id: str
    phase: str = ""
    intent_label: str = ""
    requested_tool: str = ""
    argument_fingerprint: str = ""
    plan_step_id: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    rationale_summary: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimRecord:
    claim_id: str
    kind: str = "hypothesis"
    statement: str = ""
    supporting_evidence_ids: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    alternative_explanations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: str = "candidate"
    decision_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningGraph:
    graph_version: int = 1
    evidence_records: list[EvidenceRecord] = field(default_factory=list)
    decision_records: list[DecisionRecord] = field(default_factory=list)
    claim_records: list[ClaimRecord] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.touch_ids()

    def touch_ids(self) -> None:
        self.evidence_ids = [record.evidence_id for record in self.evidence_records if record.evidence_id]
        self.decision_ids = [record.decision_id for record in self.decision_records if record.decision_id]
        self.claim_ids = [record.claim_id for record in self.claim_records if record.claim_id]

    def to_dict(self) -> dict[str, Any]:
        self.touch_ids()
        return {
            "graph_version": self.graph_version,
            "evidence_records": [json_safe_value(record) for record in self.evidence_records],
            "decision_records": [json_safe_value(record) for record in self.decision_records],
            "claim_records": [json_safe_value(record) for record in self.claim_records],
            "evidence_ids": list(self.evidence_ids),
            "decision_ids": list(self.decision_ids),
            "claim_ids": list(self.claim_ids),
        }


@dataclass
class PlanInterrupt:
    kind: str = "plan_execute_approval"
    question: str = "Plan ready. Execute it now?"
    plan_id: str = ""
    approved: bool = False
    response_mode: str = "yes/no/revise"


@dataclass
class MemoryEntry:
    content: str
    created_at_step: int = 0
    created_phase: str = ""
    freshness: str = "current"
    confidence: float | None = None


@dataclass
class ExperienceMemory:
    memory_id: str
    tier: str = "warm"
    source: str = "observed"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    last_reinforced_at: str | None = None
    run_id: str = ""
    phase: str = ""
    intent: str = ""
    intent_tags: list[str] = field(default_factory=list)
    environment_tags: list[str] = field(default_factory=list)
    entity_tags: list[str] = field(default_factory=list)
    action_type: str = ""
    tool_name: str = ""
    arguments_fingerprint: str = ""
    outcome: str = "partial"
    failure_mode: str = ""
    confidence: float = 0.0
    reuse_count: int = 0
    notes: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    pinned: bool = False
    expires_at: str | None = None


@dataclass
class ArtifactSnippet:
    artifact_id: str
    text: str
    score: float = 0.0


@dataclass
class ArtifactRecord:
    artifact_id: str
    kind: str
    source: str
    created_at: str
    size_bytes: int
    summary: str
    keywords: list[str] = field(default_factory=list)
    path_tags: list[str] = field(default_factory=list)
    tool_name: str = ""
    content_path: str | None = None
    inline_content: str | None = None
    preview_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""       # thread_id of the run that created this artifact
    tool_call_id: str = ""     # tool_call_id from the model turn that triggered this artifact


@dataclass
class EpisodicSummary:
    summary_id: str
    created_at: str
    decisions: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    failed_approaches: list[str] = field(default_factory=list)
    remaining_plan: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    full_summary_artifact_id: str | None = None


@dataclass
class ContextBrief:
    brief_id: str
    created_at: str
    tier: str                          # "warm" | "cold"
    step_range: tuple[int, int]        # (start_step, end_step) this covers
    task_goal: str                     # original_task at time of compression
    current_phase: str                 # explore | plan | execute | verify
    key_discoveries: list[str]         # things learned (max 5)
    tools_tried: list[str]             # tool names used
    blockers: list[str]                # errors / failures encountered
    files_touched: list[str]           # paths seen
    artifact_ids: list[str]            # artifact IDs referenced
    next_action_hint: str              # what the model was about to do
    staleness_step: int                # step at which this was created
    facts_confirmed: list[str] = field(default_factory=list)
    facts_unconfirmed: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    candidate_causes: list[str] = field(default_factory=list)
    disproven_causes: list[str] = field(default_factory=list)
    next_observations_needed: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    claim_refs: list[str] = field(default_factory=list)
    full_artifact_id: str | None = None # link to full-text artifact


@dataclass
class WriteSession:
    write_session_id: str = ""
    write_target_path: str = ""
    write_session_intent: str = "replace_file"  # "replace_file" | "patch_existing"
    write_session_mode: str = "chunked_author"  # "single_write" | "chunked_author" | "local_repair" | "stub_and_fill"
    write_session_started_at: float = 0.0
    write_first_chunk_at: float = 0.0
    write_staging_path: str = ""
    write_original_snapshot_path: str = ""
    write_target_existed_at_start: bool = False
    write_section_ranges: dict[str, dict[str, int]] = field(default_factory=dict)
    write_last_attempt_snapshot_path: str = ""
    write_last_attempt_sections: list[str] = field(default_factory=list)
    write_last_attempt_ranges: dict[str, dict[str, int]] = field(default_factory=dict)
    write_last_staged_hash: str = ""
    write_sections_completed: list[str] = field(default_factory=list)
    write_current_section: str = ""
    write_next_section: str = ""
    write_failed_local_patches: int = 0
    write_empty_payload_retries: int = 0
    write_salvage_count: int = 0
    write_last_verifier: dict[str, Any] | None = None
    write_session_fallback_mode: str = "stub_and_fill"
    write_pending_finalize: bool = False
    suggested_sections: list[str] = field(default_factory=list)
    status: str = "open"  # "open" | "local_repair" | "fallback" | "complete"

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class PromptBudgetSnapshot:
    estimated_prompt_tokens: int = 0
    sections: dict[str, int] = field(default_factory=dict)
    section_limits: dict[str, int] = field(default_factory=dict)
    trimmed_sections: dict[str, int] = field(default_factory=dict)
    pruned_memory_sections: dict[str, int] = field(default_factory=dict)
    pruned_memory_reasons: dict[str, list[str]] = field(default_factory=dict)
    dropped_summary_ids: list[str] = field(default_factory=list)
    dropped_artifact_ids: list[str] = field(default_factory=list)
    pressure_level: str = ""
    message_count: int = 0
    max_prompt_tokens: int | None = None
    reserve_completion_tokens: int = 0
    reserve_tool_tokens: int = 0
    compaction_estimated_prompt_tokens_before: int = 0
    compaction_estimated_prompt_tokens_after: int = 0
    compaction_threshold: int = 0
    compaction_recent_messages_before: int = 0
    compaction_recent_messages_after: int = 0
    compaction_keep_recent_initial: int = 0
    compaction_keep_recent_final: int = 0
    compaction_messages_compacted: int = 0
    compaction_attempt_count: int = 0
    compaction_stopped_reason: str = ""


@dataclass
class LoopState:
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
    prompt_budget: PromptBudgetSnapshot = field(default_factory=PromptBudgetSnapshot)
    retrieval_cache: list[str] = field(default_factory=list)
    active_intent: str = ""
    secondary_intents: list[str] = field(default_factory=list)
    intent_tags: list[str] = field(default_factory=list)
    warm_experiences: list[ExperienceMemory] = field(default_factory=list)
    retrieved_experience_ids: list[str] = field(default_factory=list)
    tool_execution_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_interrupt: dict[str, Any] | None = None
    background_processes: dict[str, dict[str, Any]] = field(default_factory=dict)
    cwd: str = field(default_factory=lambda: str(Path.cwd()))
    inventory_state: dict[str, Any] = field(default_factory=dict)
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

    def sync_plan_mirror(self) -> None:
        plan = self.active_plan or self.draft_plan
        if plan is None:
            self.plan_resolved = bool(self.working_memory.plan)
            if not self.plan_resolved:
                self.working_memory.plan = []
            return
        self.working_memory.plan = clip_string_list(
            plan.compact_lines(),
            limit=12,
            item_char_limit=220,
        )[0]
        self.working_memory.current_goal = plan.goal
        self.run_brief.current_phase_objective = plan.goal
        self.plan_resolved = True
        self.run_brief.task_contract = plan.spec_summary()
        self.run_brief.inputs = list(plan.inputs)
        self.run_brief.outputs = list(plan.outputs)
        self.run_brief.constraints = list(plan.constraints)
        self.run_brief.acceptance_criteria = list(plan.acceptance_criteria)
        self.run_brief.implementation_plan = list(plan.implementation_plan)

    def active_acceptance_criteria(self) -> list[str]:
        plan = self.active_plan or self.draft_plan
        if plan is not None and plan.acceptance_criteria:
            return list(plan.acceptance_criteria)
        if self.run_brief.acceptance_criteria:
            return list(self.run_brief.acceptance_criteria)
        return []

    def acceptance_checklist(self) -> list[dict[str, Any]]:
        criteria = self.active_acceptance_criteria()
        if not criteria:
            return []
        ledger = self.acceptance_ledger.copy()
        scratch_ledger = self.scratchpad.get("_acceptance_ledger")
        if isinstance(scratch_ledger, dict):
            for key, value in scratch_ledger.items():
                ledger[str(key)] = str(value)
        waived_items = {item for item in self.acceptance_waivers if item}
        if self.acceptance_waived:
            waived_items.update(criteria)
        checklist: list[dict[str, Any]] = []
        for criterion in criteria:
            status = str(ledger.get(criterion, "pending") or "pending")
            if criterion in waived_items:
                status = "waived"
            checklist.append(
                {
                    "criterion": criterion,
                    "status": status,
                    "satisfied": status in {"done", "passed", "complete", "completed", "waived"},
                }
            )
        return checklist

    def acceptance_ready(self) -> bool:
        checklist = self.acceptance_checklist()
        return not checklist or all(item["satisfied"] for item in checklist)

    def current_verifier_verdict(self) -> dict[str, Any] | None:
        verdict = self.last_verifier_verdict
        if isinstance(verdict, dict):
            return verdict
        scratch_verdict = self.scratchpad.get("_last_verifier_verdict")
        return scratch_verdict if isinstance(scratch_verdict, dict) else None

    def contract_flow_active(self) -> bool:
        if self.repair_cycle_id or self.current_verifier_verdict() is not None:
            return True
        if self.active_plan is not None or self.draft_plan is not None:
            return True
        if self.run_brief.acceptance_criteria or self.run_brief.implementation_plan:
            return True
        normalized_intent = normalize_intent_label(self.active_intent)
        if normalized_intent in {
            "write_file",
            "requested_write_file",
            "requested_file_write",
            "requested_file_append",
            "requested_file_patch",
            "requested_file_delete",
        }:
            return True

        intent_tags = {
            str(tag).strip().lower()
            for tag in (self.intent_tags or [])
            if str(tag).strip()
        }
        if intent_tags & {"write_file", "file_write", "file_patch", "mutate_repo"}:
            return True

        target_paths = self.scratchpad.get("_task_target_paths")
        has_targets = isinstance(target_paths, list) and any(str(path).strip() for path in target_paths)
        task_bits = [
            str(self.run_brief.original_task or "").strip(),
            str(self.working_memory.current_goal or "").strip(),
        ]
        task_text = " ".join(bit for bit in task_bits if bit).lower()
        if not task_text:
            return False

        write_markers = (
            "write ",
            "edit ",
            "patch ",
            "create ",
            "build ",
            "implement ",
            "update ",
            "append ",
            "replace ",
            "refactor ",
            "fix ",
        )
        artifact_markers = (
            "script",
            "file",
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".json",
            ".md",
            "unittest",
            "test suite",
            "tests",
        )
        return any(marker in task_text for marker in write_markers) and (
            has_targets or any(marker in task_text for marker in artifact_markers)
        )

    def contract_phase(self) -> str:
        scratch_phase = str(self.scratchpad.get("_contract_phase") or "").strip()
        if scratch_phase:
            return scratch_phase
        plan = self.active_plan or self.draft_plan
        if self.planning_mode_enabled and not (plan and plan.approved):
            return self.current_phase
        if self.repair_cycle_id:
            return "repair"
        if not self.contract_flow_active():
            return self.current_phase
        if self.current_verifier_verdict() is None:
            return "author"
        return "verify" if not self.acceptance_ready() else "execute"
        return self.current_phase

    def upsert_experience(self, memory: ExperienceMemory) -> ExperienceMemory:
        for i, existing in enumerate(self.warm_experiences):
            if existing.memory_id == memory.memory_id:
                self.warm_experiences[i] = memory
                self.touch()
                return memory
        self.warm_experiences.append(memory)
        self.touch()
        return memory

    def reinforce_experience(self, memory_id: str, *, success: bool) -> None:
        for memory in self.warm_experiences:
            if memory.memory_id == memory_id:
                memory.reuse_count += 1
                modifier = 0.05 if success else -0.15
                memory.confidence = max(0.0, min(1.0, memory.confidence + modifier))
                memory.last_reinforced_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self.touch()
                break

    def decay_experiences(self, *, rate: float = 0.005) -> None:
        remaining = []
        now = datetime.now(timezone.utc)
        for memory in self.warm_experiences:
            if memory.pinned:
                remaining.append(memory)
                continue

            expires_at = _coerce_datetime(memory.expires_at)
            if expires_at is not None and expires_at <= now:
                continue

            last_reinforced_at = _coerce_datetime(memory.last_reinforced_at)
            age_penalty = rate
            if last_reinforced_at is not None:
                age_days = max(0.0, (now - last_reinforced_at).total_seconds() / 86400.0)
                age_penalty += min(0.03, age_days * rate)
            else:
                created_at = _coerce_datetime(memory.created_at)
                if created_at is not None:
                    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
                    age_penalty += min(0.02, age_days * rate * 0.5)

            memory.confidence = max(0.0, memory.confidence - age_penalty)
            if memory.confidence > 0.1:
                remaining.append(memory)
        self.warm_experiences = remaining
        self.touch()

    def prune_stale_meta(self, *, limit: int = 15) -> None:
        self.working_memory.known_fact_meta = [
            m for m in self.working_memory.known_fact_meta 
            if not memory_entry_is_stale(
                m, 
                current_step=self.step_count, 
                current_phase=self.current_phase, 
                staleness_step_limit=limit
            )
        ]
        self.working_memory.failure_meta = [
            m for m in self.working_memory.failure_meta 
            if not memory_entry_is_stale(
                m, 
                current_step=self.step_count, 
                current_phase=self.current_phase, 
                staleness_step_limit=limit
            )
        ]
        self.working_memory.next_action_meta = [
            m for m in self.working_memory.next_action_meta 
            if not memory_entry_is_stale(
                m, 
                current_step=self.step_count, 
                current_phase=self.current_phase, 
                staleness_step_limit=limit
            )
        ]
        self.align_meta_to_content()

    def align_meta_to_content(self) -> None:
        self.working_memory.known_fact_meta = align_memory_entries(
            self.working_memory.known_facts, 
            self.working_memory.known_fact_meta, 
            current_step=self.step_count,
            current_phase=self.current_phase
        )
        self.working_memory.failure_meta = align_memory_entries(
            self.working_memory.failures, 
            self.working_memory.failure_meta, 
            current_step=self.step_count,
            current_phase=self.current_phase
        )
        self.working_memory.next_action_meta = align_memory_entries(
            self.working_memory.next_actions, 
            self.working_memory.next_action_meta, 
            current_step=self.step_count,
            current_phase=self.current_phase
        )

    def memory_entries(self, section: str) -> list[MemoryEntry]:
        if section == "known_facts":
            return self.working_memory.known_fact_meta
        if section == "failures":
            return self.working_memory.failure_meta
        if section == "next_actions":
            return self.working_memory.next_action_meta
        return []

    def set_memory_entries(self, section: str, entries: list[MemoryEntry]) -> None:
        if section == "known_facts":
            self.working_memory.known_facts = [e.content for e in entries]
            self.working_memory.known_fact_meta = entries
        elif section == "failures":
            self.working_memory.failures = [e.content for e in entries]
            self.working_memory.failure_meta = entries
        elif section == "next_actions":
            self.working_memory.next_actions = [e.content for e in entries]
            self.working_memory.next_action_meta = entries
        self.touch()

    def append_memory_entry(self, section: str, entry: MemoryEntry) -> None:
        entries = self.memory_entries(section)
        entries.append(entry)
        self.set_memory_entries(section, entries)

    def remove_memory_entry(self, section: str, marker: str) -> bool:
        entries = self.memory_entries(section)
        initial_len = len(entries)
        new_entries = [e for e in entries if marker not in e.content]
        if len(new_entries) < initial_len:
            self.set_memory_entries(section, new_entries)
            return True
        return False

    @property
    def conversation_history(self) -> list[ConversationMessage]:
        # Compatibility alias for older code and checkpoint consumers.
        return list(self.recent_messages)

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
            "prompt_budget": json_safe_value(self.prompt_budget),
            "retrieval_cache": json_safe_value(self.retrieval_cache),
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
            "inventory_state": json_safe_value(self.inventory_state),
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
        raw["prompt_budget"] = _coerce_prompt_budget(migrated.get("prompt_budget"))
        raw["retrieval_cache"] = _coerce_string_list(migrated.get("retrieval_cache"))
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
        raw["inventory_state"] = _coerce_json_dict_payload(migrated.get("inventory_state"))
        raw["active_tool_profiles"] = _coerce_string_list(migrated.get("active_tool_profiles")) or ["core"]
        raw["tool_history"] = _coerce_string_list(migrated.get("tool_history"))
        raw["write_session"] = _coerce_write_session(migrated.get("write_session"))
        raw["created_at"] = _coerce_timestamp_string(migrated.get("created_at"))
        raw["updated_at"] = _coerce_timestamp_string(migrated.get("updated_at"))
        raw["last_completion_tokens"] = _coerce_int(migrated.get("last_completion_tokens"), default=0)
        state = cls(**_filter_dataclass_payload(cls, raw))
        state.sync_plan_mirror()
        return state


def _migrate_loop_state_payload(payload: dict[str, Any], *, incoming_version: int) -> dict[str, Any]:
    migrated = dict(payload)
    if incoming_version >= LOOP_STATE_SCHEMA_VERSION:
        return migrated

    write_session = migrated.get("write_session")
    if isinstance(write_session, dict):
        ws = dict(write_session)
        if ws.get("session_id") and not ws.get("write_session_id"):
            ws["write_session_id"] = ws.get("session_id")
        if ws.get("mode") and not ws.get("write_session_mode"):
            ws["write_session_mode"] = ws.get("mode")
        if ws.get("lifecycle_status") and not ws.get("status"):
            ws["status"] = ws.get("lifecycle_status")
        migrated["write_session"] = ws
    if incoming_version < LOOP_STATE_SCHEMA_VERSION and "reasoning_graph" not in migrated:
        migrated["reasoning_graph"] = {}
    return migrated


def json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        sanitized = [json_safe_value(item) for item in value]
        return sorted(sanitized, key=str)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_safe_value(getattr(value, field.name)) for field in fields(value)}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return json_safe_value(to_dict())
        except Exception:
            pass
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _coerce_run_brief(value: Any) -> RunBrief:
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


def _coerce_conversation_message(value: Any) -> ConversationMessage | None:
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


def _coerce_working_memory(value: Any, *, current_step: int = 0, current_phase: str = "explore") -> WorkingMemory:
    if isinstance(value, WorkingMemory):
        return value
    if isinstance(value, dict):
        payload = _filter_dataclass_payload(WorkingMemory, value)
        payload["current_goal"] = str(payload.get("current_goal") or "")
        for key in (
            "plan",
            "decisions",
            "open_questions",
            "known_facts",
            "failures",
            "next_actions",
        ):
            payload[key] = _coerce_string_list(payload.get(key))
        
        payload["known_fact_meta"] = _coerce_memory_entry_list(
            payload.get("known_fact_meta"),
            fallback_texts=payload["known_facts"],
            current_step=current_step,
            current_phase=current_phase,
        )
        payload["failure_meta"] = _coerce_memory_entry_list(
            payload.get("failure_meta"),
            fallback_texts=payload["failures"],
            current_step=current_step,
            current_phase=current_phase,
        )
        payload["next_action_meta"] = _coerce_memory_entry_list(
            payload.get("next_action_meta"),
            fallback_texts=payload["next_actions"],
            current_step=current_step,
            current_phase=current_phase,
        )
        return WorkingMemory(**payload)
    return WorkingMemory()


def _coerce_string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if str(key).strip()}


def _coerce_int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, item in value.items():
        try:
            normalized[str(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return normalized


def _compact_plan_step_lines(step: PlanStep, *, depth: int = 0) -> list[str]:
    prefix = "  " * depth
    lines = [f"{prefix}{step.compact_label()}"]
    if step.description:
        lines.append(f"{prefix}  {step.description}")
    if step.notes:
        for note in step.notes:
            lines.append(f"{prefix}  note: {note}")
    if step.evidence_refs:
        lines.append(f"{prefix}  evidence: {', '.join(step.evidence_refs)}")
    if step.claim_refs:
        lines.append(f"{prefix}  claims: {', '.join(step.claim_refs)}")
    for substep in step.substeps:
        lines.extend(_compact_plan_step_lines(substep, depth=depth + 1))
    return lines


def _coerce_plan_step(value: Any) -> PlanStep | None:
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
    payload["substeps"] = [
        substep
        for item in _coerce_list_payload(payload.get("substeps"))
        if (substep := _coerce_plan_step(item)) is not None
    ]
    return PlanStep(**_filter_dataclass_payload(PlanStep, payload))


def _coerce_execution_plan(value: Any) -> ExecutionPlan | None:
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


def _coerce_plan_interrupt(value: Any) -> PlanInterrupt | None:
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


def _coerce_artifact_record(value: Any, *, artifact_id: str) -> ArtifactRecord:
    if isinstance(value, ArtifactRecord):
        return value
    if isinstance(value, dict):
        payload = dict(value)
        payload.setdefault("artifact_id", artifact_id)
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        payload.setdefault("kind", "tool_result")
        payload.setdefault("source", "")
        payload.setdefault("size_bytes", 0)
        payload.setdefault("summary", "")
        payload["artifact_id"] = str(payload.get("artifact_id", artifact_id) or artifact_id)
        payload["kind"] = str(payload.get("kind", "tool_result") or "tool_result")
        payload["source"] = str(payload.get("source", "") or "")
        payload["created_at"] = _coerce_timestamp_string(payload.get("created_at"))
        payload["size_bytes"] = _coerce_int(payload.get("size_bytes"), default=0)
        payload["summary"] = str(payload.get("summary", "") or "")
        payload["keywords"] = _coerce_string_list(payload.get("keywords"))
        payload["path_tags"] = _coerce_string_list(payload.get("path_tags"))
        payload["tool_name"] = str(payload.get("tool_name", "") or "")
        for key in ("content_path", "inline_content", "preview_text"):
            field_value = payload.get(key)
            payload[key] = None if field_value in (None, "") else str(field_value)
        metadata = json_safe_value(payload.get("metadata") or {})
        payload["metadata"] = metadata if isinstance(metadata, dict) else {}
        return ArtifactRecord(**_filter_dataclass_payload(ArtifactRecord, payload))
    return ArtifactRecord(
        artifact_id=artifact_id,
        kind="tool_result",
        source="",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        size_bytes=0,
        summary="",
    )


def _coerce_episodic_summary(value: Any) -> EpisodicSummary:
    if isinstance(value, EpisodicSummary):
        return value
    if isinstance(value, dict):
        payload = dict(value)
        payload.setdefault("summary_id", "")
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        payload["summary_id"] = str(payload.get("summary_id", "") or "")
        payload["created_at"] = _coerce_timestamp_string(payload.get("created_at"))
        for key in (
            "decisions",
            "files_touched",
            "failed_approaches",
            "remaining_plan",
            "artifact_ids",
            "notes",
        ):
            payload[key] = _coerce_string_list(payload.get(key))
        return EpisodicSummary(**_filter_dataclass_payload(EpisodicSummary, payload))
    return EpisodicSummary(
        summary_id="",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _coerce_context_brief(value: Any) -> ContextBrief:
    if isinstance(value, ContextBrief):
        return value
    if isinstance(value, dict):
        payload = dict(value)
        payload.setdefault("brief_id", "")
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        payload.setdefault("tier", "warm")
        payload.setdefault("step_range", (0, 0))
        payload["brief_id"] = str(payload.get("brief_id", "") or "")
        payload["created_at"] = _coerce_timestamp_string(payload.get("created_at"))
        
        for key in (
            "key_discoveries",
            "tools_tried",
            "blockers",
            "files_touched",
            "artifact_ids",
            "facts_confirmed",
            "facts_unconfirmed",
            "open_questions",
            "candidate_causes",
            "disproven_causes",
            "next_observations_needed",
            "evidence_refs",
            "claim_refs",
        ):
            payload[key] = _coerce_string_list(payload.get(key))
            
        return ContextBrief(**_filter_dataclass_payload(ContextBrief, payload))
    
    return ContextBrief(
        brief_id="",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        tier="warm",
        step_range=(0, 0),
        task_goal="",
        current_phase="",
        key_discoveries=[],
        tools_tried=[],
        blockers=[],
        files_touched=[],
        artifact_ids=[],
        next_action_hint="",
        staleness_step=0,
        facts_confirmed=[],
        facts_unconfirmed=[],
        open_questions=[],
        candidate_causes=[],
        disproven_causes=[],
        next_observations_needed=[],
        evidence_refs=[],
        claim_refs=[],
    )


def _coerce_evidence_record(value: Any) -> EvidenceRecord | None:
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


def _coerce_decision_record(value: Any) -> DecisionRecord | None:
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


def _coerce_claim_record(value: Any) -> ClaimRecord | None:
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


def _coerce_reasoning_graph(value: Any) -> ReasoningGraph:
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


def _coerce_prompt_budget(value: Any) -> PromptBudgetSnapshot:
    if isinstance(value, PromptBudgetSnapshot):
        return value
    if isinstance(value, dict):
        payload = _filter_dataclass_payload(PromptBudgetSnapshot, value)
        payload["estimated_prompt_tokens"] = _coerce_int(
            payload.get("estimated_prompt_tokens"), default=0
        )
        sections = json_safe_value(payload.get("sections") or {})
        if not isinstance(sections, dict):
            sections = {}
        payload["sections"] = {
            str(key): _coerce_int(item, default=0) for key, item in sections.items()
        }
        payload["message_count"] = _coerce_int(payload.get("message_count"), default=0)
        max_prompt_tokens = payload.get("max_prompt_tokens")
        payload["max_prompt_tokens"] = (
            None if max_prompt_tokens in (None, "") else _coerce_int(max_prompt_tokens, default=0)
        )
        payload["reserve_completion_tokens"] = _coerce_int(
            payload.get("reserve_completion_tokens"), default=0
        )
        payload["reserve_tool_tokens"] = _coerce_int(
            payload.get("reserve_tool_tokens"), default=0
        )
        payload["compaction_estimated_prompt_tokens_before"] = _coerce_int(
            payload.get("compaction_estimated_prompt_tokens_before"), default=0
        )
        payload["compaction_estimated_prompt_tokens_after"] = _coerce_int(
            payload.get("compaction_estimated_prompt_tokens_after"), default=0
        )
        payload["compaction_threshold"] = _coerce_int(payload.get("compaction_threshold"), default=0)
        payload["compaction_recent_messages_before"] = _coerce_int(
            payload.get("compaction_recent_messages_before"), default=0
        )
        payload["compaction_recent_messages_after"] = _coerce_int(
            payload.get("compaction_recent_messages_after"), default=0
        )
        payload["compaction_keep_recent_initial"] = _coerce_int(
            payload.get("compaction_keep_recent_initial"), default=0
        )
        payload["compaction_keep_recent_final"] = _coerce_int(
            payload.get("compaction_keep_recent_final"), default=0
        )
        payload["compaction_messages_compacted"] = _coerce_int(
            payload.get("compaction_messages_compacted"), default=0
        )
        payload["compaction_attempt_count"] = _coerce_int(
            payload.get("compaction_attempt_count"), default=0
        )
        payload["compaction_stopped_reason"] = str(
            payload.get("compaction_stopped_reason", "") or ""
        )
        return PromptBudgetSnapshot(**payload)
    return PromptBudgetSnapshot()


def _coerce_write_session(value: Any) -> WriteSession | None:
    if isinstance(value, WriteSession):
        return value
    if not isinstance(value, dict):
        return None
    payload = _filter_dataclass_payload(WriteSession, value)
    
    # Handle legacy field names if they exist in the payload
    if "session_id" in value and "write_session_id" not in payload:
        payload["write_session_id"] = str(value["session_id"])
    if "target_path" in value and "write_target_path" not in payload:
        payload["write_target_path"] = str(value["target_path"])
    if "completed_sections" in value and "write_sections_completed" not in payload:
        payload["write_sections_completed"] = _coerce_string_list(value["completed_sections"])
    if "current_section" in value and "write_current_section" not in payload:
        payload["write_current_section"] = str(value["current_section"])
    if "next_section" in value and "write_next_section" not in payload:
        payload["write_next_section"] = str(value["next_section"])
    if "verdict" in value and "write_last_verifier" not in payload:
        payload["write_last_verifier"] = _coerce_json_dict_payload(value["verdict"])
    
    payload["write_session_id"] = str(payload.get("write_session_id", "") or "")
    payload["write_target_path"] = str(payload.get("write_target_path", "") or "")
    intent = str(payload.get("write_session_intent", "replace_file") or "replace_file").strip().lower()
    payload["write_session_intent"] = intent if intent in {"replace_file", "patch_existing"} else "replace_file"
    mode = str(payload.get("write_session_mode", "chunked_author") or "chunked_author").strip().lower()
    if mode not in {"single_write", "chunked_author", "local_repair", "stub_and_fill"}:
        mode = "chunked_author"
    payload["write_session_mode"] = mode
    payload["write_session_started_at"] = _coerce_float(payload.get("write_session_started_at"), default=0.0)
    payload["write_first_chunk_at"] = _coerce_float(payload.get("write_first_chunk_at"), default=0.0)
    payload["write_staging_path"] = str(payload.get("write_staging_path", "") or "")
    payload["write_original_snapshot_path"] = str(payload.get("write_original_snapshot_path", "") or "")
    existed_at_start = payload.get("write_target_existed_at_start")
    payload["write_target_existed_at_start"] = (
        bool(existed_at_start)
        if isinstance(existed_at_start, bool)
        else str(existed_at_start or "").strip().lower() in {"1", "true", "yes", "on"}
    )
    payload["write_section_ranges"] = _coerce_write_section_ranges(payload.get("write_section_ranges"))
    payload["write_last_attempt_snapshot_path"] = str(payload.get("write_last_attempt_snapshot_path", "") or "")
    payload["write_last_attempt_sections"] = _coerce_string_list(payload.get("write_last_attempt_sections"))
    payload["write_last_attempt_ranges"] = _coerce_write_section_ranges(payload.get("write_last_attempt_ranges"))
    payload["write_last_staged_hash"] = str(payload.get("write_last_staged_hash", "") or "")
    payload["write_sections_completed"] = _coerce_string_list(payload.get("write_sections_completed"))
    payload["write_current_section"] = str(payload.get("write_current_section", "") or "")
    payload["write_next_section"] = str(payload.get("write_next_section", "") or "")
    payload["write_failed_local_patches"] = _coerce_int(payload.get("write_failed_local_patches"), default=0)
    payload["write_empty_payload_retries"] = _coerce_int(payload.get("write_empty_payload_retries"), default=0)
    payload["write_salvage_count"] = _coerce_int(payload.get("write_salvage_count"), default=0)
    payload["write_last_verifier"] = _coerce_json_dict_payload(payload.get("write_last_verifier"))
    payload["write_session_fallback_mode"] = str(payload.get("write_session_fallback_mode", "stub_and_fill") or "stub_and_fill")
    pending_finalize = payload.get("write_pending_finalize")
    payload["write_pending_finalize"] = bool(pending_finalize) if isinstance(pending_finalize, bool) else str(pending_finalize or "").strip().lower() in {"1", "true", "yes", "on"}
    payload["suggested_sections"] = _coerce_string_list(payload.get("suggested_sections"))
    status = str(payload.get("status", "open") or "open").strip().lower()
    if status not in {"open", "local_repair", "fallback", "complete"}:
        status = "open"
    payload["status"] = status
    
    return WriteSession(**payload)




def _coerce_background_process_record(value: Any, *, job_id: str) -> dict[str, Any]:
    payload = dict(value) if isinstance(value, dict) else {}
    pid = payload.get("pid")
    try:
        normalized_pid = int(pid)
    except (TypeError, ValueError):
        normalized_pid = 0
    command = str(payload.get("command", ""))
    cwd = str(payload.get("cwd", ""))
    started_at = payload.get("started_at")
    if not isinstance(started_at, str) or not started_at.strip():
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status = str(payload.get("status", "running" if normalized_pid else "unknown"))
    normalized = json_safe_value(payload)
    if not isinstance(normalized, dict):
        normalized = {}
    normalized["job_id"] = job_id
    normalized["pid"] = normalized_pid
    normalized["command"] = command
    normalized["cwd"] = cwd
    normalized["started_at"] = str(started_at)
    normalized["status"] = status
    return normalized


def _coerce_tool_execution_record(value: Any, *, operation_id: str) -> dict[str, Any]:
    payload = dict(value) if isinstance(value, dict) else {}
    normalized = json_safe_value(payload)
    if not isinstance(normalized, dict):
        normalized = {}
    normalized["operation_id"] = str(payload.get("operation_id", operation_id) or operation_id)
    if "thread_id" in payload:
        normalized["thread_id"] = str(payload.get("thread_id", ""))
    if "step_count" in payload:
        step_count = payload.get("step_count", 0)
        try:
            normalized["step_count"] = int(step_count)
        except (TypeError, ValueError):
            normalized["step_count"] = 0
    if "tool_name" in payload:
        normalized["tool_name"] = str(payload.get("tool_name", ""))
    if "tool_call_id" in payload:
        tool_call_id = payload.get("tool_call_id")
        normalized["tool_call_id"] = None if tool_call_id is None else str(tool_call_id)
    if "args" in payload:
        args = json_safe_value(payload.get("args") or {})
        normalized["args"] = args if isinstance(args, dict) else {}
    if "result" in payload:
        normalized["result"] = _coerce_tool_envelope_payload(payload.get("result"))
    tool_message = _coerce_conversation_message_payload(payload.get("tool_message"))
    if tool_message is not None:
        normalized["tool_message"] = tool_message
    else:
        normalized.pop("tool_message", None)
    artifact_id = payload.get("artifact_id")
    if artifact_id is None:
        normalized.pop("artifact_id", None)
    else:
        normalized["artifact_id"] = str(artifact_id)
    return normalized


def _coerce_tool_envelope_payload(value: Any) -> dict[str, Any]:
    payload = dict(value) if isinstance(value, dict) else {}
    normalized = json_safe_value(payload)
    if not isinstance(normalized, dict):
        normalized = {}
    metadata = normalized.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    normalized["success"] = bool(payload.get("success"))
    normalized["output"] = json_safe_value(payload.get("output"))
    error = payload.get("error")
    normalized["error"] = None if error is None else str(error)
    normalized["metadata"] = metadata
    return normalized


def _coerce_pending_interrupt_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    normalized = json_safe_value(value)
    if not isinstance(normalized, dict):
        return None
    kind = value.get("kind")
    normalized["kind"] = str(kind) if kind is not None else "ask_human"
    for key in ("question", "current_phase", "thread_id", "operation_id", "plan_id", "response_mode"):
        if key in value:
            normalized[key] = str(value.get(key, ""))
    if "approved" in value:
        normalized["approved"] = bool(value.get("approved"))
    active_profiles = normalized.get("active_profiles")
    if active_profiles is not None:
        if isinstance(active_profiles, list):
            normalized["active_profiles"] = [str(item) for item in active_profiles]
        else:
            normalized["active_profiles"] = []
    recent_tool_outcomes = normalized.get("recent_tool_outcomes")
    if recent_tool_outcomes is not None:
        if isinstance(recent_tool_outcomes, list):
            normalized["recent_tool_outcomes"] = [
                json_safe_value(item) for item in recent_tool_outcomes if isinstance(item, dict)
            ]
        else:
            normalized["recent_tool_outcomes"] = []
    return normalized




def _coerce_list_payload(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _coerce_dict_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}






def _coerce_conversation_message_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    normalized = json_safe_value(value)
    if not isinstance(normalized, dict):
        return None
    filtered = _filter_dataclass_payload(ConversationMessage, normalized)
    filtered["role"] = str(filtered.get("role", "tool"))
    for key in ("content", "name", "tool_call_id", "retrieval_safe_text"):
        if key in filtered and filtered[key] is not None:
            filtered[key] = str(filtered[key])
    if "tool_calls" in filtered:
        tool_calls = filtered.get("tool_calls")
        filtered["tool_calls"] = tool_calls if isinstance(tool_calls, list) else []
    if "metadata" in filtered:
        metadata = filtered.get("metadata")
        filtered["metadata"] = metadata if isinstance(metadata, dict) else {}
    return filtered


def _filter_dataclass_payload(dataclass_type: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    allowed_fields = {field.name for field in fields(dataclass_type)}
    return {key: value for key, value in payload.items() if key in allowed_fields}


def _coerce_experience_memory(value: Any) -> ExperienceMemory:
    if isinstance(value, ExperienceMemory):
        return value
    if not isinstance(value, dict):
        return ExperienceMemory(memory_id="unknown")
    payload = dict(value)
    memory_id = str(payload.get("memory_id", "") or "unknown")
    tier = str(payload.get("tier", "warm") or "warm").strip().lower()
    if tier not in {"warm", "cold"}:
        tier = "warm"
    source = str(payload.get("source", "observed") or "observed").strip().lower()
    if source not in {"manual", "observed", "summarized", "imported"}:
        source = "observed"
    outcome = str(payload.get("outcome", "partial") or "partial").strip().lower()
    if outcome not in {"success", "failure", "partial"}:
        outcome = "partial"
    confidence = _coerce_float(payload.get("confidence"), default=0.0)
    confidence = max(0.0, min(1.0, confidence))
    last_reinforced_at = payload.get("last_reinforced_at")
    expires_at = payload.get("expires_at")
    return ExperienceMemory(
        memory_id=memory_id,
        tier=tier,
        source=source,
        created_at=_coerce_timestamp_string(payload.get("created_at")),
        last_reinforced_at=None if last_reinforced_at in (None, "") else str(last_reinforced_at),
        run_id=str(payload.get("run_id", "") or ""),
        phase=str(payload.get("phase", "") or ""),
        intent=str(payload.get("intent", "") or ""),
        intent_tags=_coerce_string_list(payload.get("intent_tags")),
        environment_tags=_coerce_string_list(payload.get("environment_tags")),
        entity_tags=_coerce_string_list(payload.get("entity_tags")),
        action_type=str(payload.get("action_type", "") or ""),
        tool_name=str(payload.get("tool_name", "") or ""),
        arguments_fingerprint=str(payload.get("arguments_fingerprint", "") or ""),
        outcome=outcome,
        failure_mode=str(payload.get("failure_mode", "") or ""),
        confidence=confidence,
        reuse_count=max(0, _coerce_int(payload.get("reuse_count"), default=0)),
        notes=str(payload.get("notes", "") or ""),
        evidence_refs=_coerce_string_list(payload.get("evidence_refs")),
        supersedes=_coerce_string_list(payload.get("supersedes")),
        pinned=bool(payload.get("pinned")),
        expires_at=None if expires_at in (None, "") else str(expires_at),
    )


def _coerce_memory_entry(value: Any, *, current_step: int, current_phase: str) -> MemoryEntry | None:
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


def _coerce_memory_entry_list(
    value: Any,
    *,
    fallback_texts: list[str],
    current_step: int,
    current_phase: str,
) -> list[MemoryEntry]:
    if isinstance(value, list):
        entries = [
            entry
            for item in value
            if (entry := _coerce_memory_entry(item, current_step=current_step, current_phase=current_phase)) is not None
        ]
        if entries:
            return entries
    return [
        MemoryEntry(
            content=text,
            created_at_step=current_step,
            created_phase=current_phase,
        )
        for text in fallback_texts
        if text
    ]


def memory_entry_is_stale(
    entry: MemoryEntry,
    *,
    current_step: int,
    current_phase: str,
    staleness_step_limit: int = 12,
) -> bool:
    if entry.freshness == "pinned":
        return False
    if entry.freshness == "phase" and entry.created_phase != current_phase:
        return True
    if staleness_step_limit > 0:
        age = current_step - entry.created_at_step
        if age >= staleness_step_limit:
            return True
    return False


def _trim_recent_messages(
    messages: list[ConversationMessage],
    *,
    limit: int,
) -> list[ConversationMessage]:
    """ Trims history while preserving the original task and ensuring tool call integrity. """
    if len(messages) <= limit:
        return list(messages)
    
    # Identify indices of all user messages
    user_indices = [i for i, m in enumerate(messages) if m.role == "user"]
    if not user_indices:
        return list(messages[-limit:])
    
    # TASK ANCHOR: The very first user message (turn 0/1) should ideally be preserved
    task_anchor_index = user_indices[0]
    
    # RECENT WINDOW: We want to keep the latest messages
    suffix_size = max(limit - 1, 1)
    
    # ADJUSTMENT: Ensure we don't start at a 'tool' role, which would be invalid for the API.
    # Pull in the parent 'assistant' call if needed.
    start_idx = len(messages) - suffix_size
    while start_idx > 0 and messages[start_idx].role == "tool":
        start_idx -= 1
        
    suffix = messages[start_idx:]
    
    # If the anchor is already in the suffix, just return the whole suffix (truncated to limit)
    anchor = messages[task_anchor_index]
    if any(m is anchor for m in suffix):
        return list(suffix)
        
    # [Anchor] + [Suffix]
    result = [anchor] + list(suffix)
    if len(result) > limit:
        # If still over limit due to tool alignment, drop the start of the suffix
        # but ensure we dont break tool-call integrity again.
        to_drop = len(result) - limit
        trimmed_suffix = list(suffix)
        while to_drop > 0 and trimmed_suffix:
            trimmed_suffix.pop(0)
            to_drop -= 1
        # Re-check integrity of the new start
        while trimmed_suffix and trimmed_suffix[0].role == "tool":
            trimmed_suffix.pop(0)
        return [anchor] + trimmed_suffix

    return result


def clip_text_value(
    value: Any,
    *,
    limit: int | None,
    marker: str = " [truncated]",
) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    if limit is None or limit <= 0 or len(text) <= limit:
        return text, False
    trimmed_limit = max(1, limit - len(marker))
    clipped = text[:trimmed_limit].rstrip()
    if clipped:
        clipped = f"{clipped}{marker}"
    else:
        clipped = marker.strip()
    return clipped, True


def clip_string_list(
    values: Any,
    *,
    limit: int | None,
    item_char_limit: int | None = None,
    keep_tail: bool = True,
    marker: str = " [truncated]",
) -> tuple[list[str], bool]:
    items = _coerce_string_list(values)
    normalized: list[str] = []
    clipped = False
    for item in items:
        clipped_item, item_was_clipped = clip_text_value(item, limit=item_char_limit, marker=marker)
        if item_was_clipped:
            clipped = True
        if clipped_item and clipped_item not in normalized:
            normalized.append(clipped_item)
    if limit is not None and limit > 0 and len(normalized) > limit:
        clipped = True
        normalized = normalized[-limit:] if keep_tail else normalized[:limit]
    return normalized, clipped


def align_memory_entries(
    contents: list[str],
    previous_meta: list[MemoryEntry],
    *,
    current_step: int,
    current_phase: str,
    confidence: float = 0.7,
) -> list[MemoryEntry]:
    meta: list[MemoryEntry] = []
    lookup = {m.content: m for m in previous_meta}
    for text in contents:
        if text in lookup:
            meta.append(lookup[text])
        else:
            meta.append(MemoryEntry(
                content=text,
                created_at_step=current_step,
                created_phase=current_phase,
                confidence=confidence,
            ))
    return meta
