from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
class StepVerifierSpec:
    kind: str
    args: dict[str, Any] = field(default_factory=dict)
    required: bool = True
    timeout_sec: int = 30


@dataclass
class StepOutputSpec:
    kind: str = "artifact"
    ref: str = ""
    description: str = ""
    required: bool = True


@dataclass
class StepVerificationResult:
    step_id: str
    step_run_id: str = ""
    passed: bool = False
    failed_criteria: list[str] = field(default_factory=list)
    verifier_results: list[dict[str, Any]] = field(default_factory=list)
    evidence_artifact_id: str = ""


@dataclass
class StepEvidenceArtifact:
    step_id: str
    step_run_id: str = ""
    attempt: int = 1
    summary: str = ""
    artifact_ids: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    verifier_results: list[dict[str, Any]] = field(default_factory=list)
    tool_operation_ids: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


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
    task: str = ""
    tool_allowlist: list[str] = field(default_factory=list)
    prompt_token_budget: int = 0
    acceptance: list[str] = field(default_factory=list)
    verifiers: list[StepVerifierSpec] = field(default_factory=list)
    outputs_expected: list[StepOutputSpec] = field(default_factory=list)
    max_retries: int = 3
    retry_count: int = 0
    failure_reasons: list[str] = field(default_factory=list)

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
        from .state_records import _compact_plan_step_lines

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
        from .state_support import json_safe_value

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
    namespace: str = ""
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
    session_id: str = ""
    tool_call_id: str = ""


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
class TurnBundle:
    bundle_id: str
    created_at: str
    tier: str = "l1"
    step_range: tuple[int, int] = (0, 0)
    phase: str = ""
    intent: str = ""
    summary_lines: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    observation_refs: list[str] = field(default_factory=list)
    observation_summaries: list[str] = field(default_factory=list)
    observation_kinds: list[str] = field(default_factory=list)
    compaction_strategy: str = ""
    transcript_fallback_used: bool = False
    source_message_count: int = 0


@dataclass
class ContextBrief:
    brief_id: str
    created_at: str
    tier: str
    step_range: tuple[int, int]
    task_goal: str
    current_phase: str
    key_discoveries: list[str]
    tools_tried: list[str]
    blockers: list[str]
    files_touched: list[str]
    artifact_ids: list[str]
    next_action_hint: str
    staleness_step: int
    facts_confirmed: list[str] = field(default_factory=list)
    facts_unconfirmed: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    candidate_causes: list[str] = field(default_factory=list)
    disproven_causes: list[str] = field(default_factory=list)
    next_observations_needed: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    observation_refs: list[str] = field(default_factory=list)
    claim_refs: list[str] = field(default_factory=list)
    new_facts: list[str] = field(default_factory=list)
    invalidated_facts: list[str] = field(default_factory=list)
    state_changes: list[str] = field(default_factory=list)
    decision_deltas: list[str] = field(default_factory=list)
    full_artifact_id: str | None = None


@dataclass
class WriteSession:
    write_session_id: str = ""
    write_target_path: str = ""
    write_session_intent: str = "replace_file"
    write_session_mode: str = "chunked_author"
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
    status: str = "open"

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
    included_compaction_levels: list[str] = field(default_factory=list)
    dropped_compaction_levels: list[str] = field(default_factory=list)
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
