from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Literal, TYPE_CHECKING, TypeVar

from ..state import DecisionRecord, ExperienceMemory
from .frame import PromptStateDrop
from .observations import ObservationPacket, build_observation_packets
from .policy import ContextPolicy, estimate_text_tokens

if TYPE_CHECKING:  # pragma: no cover
    from ..graph.tool_plan_observations import ToolPlanObservation
    from ..state import LoopState


ReWOORole = Literal["planner", "solver", "refiner"]
T = TypeVar("T")


@dataclass(slots=True)
class ReWOOPlanLane:
    goal: str = ""
    task_contract: str = ""
    constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    active_plan: list[str] = field(default_factory=list)
    active_step: str = ""
    pending_dependencies: list[str] = field(default_factory=list)
    prior_failures: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    repair_nudges: list[str] = field(default_factory=list)
    subtask_context: list[str] = field(default_factory=list)
    hard_route_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReWOOEvidenceLane:
    packets: list[ObservationPacket] = field(default_factory=list)
    tool_plan_observations: list["ToolPlanObservation"] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReWOODecisionLane:
    records: list[DecisionRecord] = field(default_factory=list)
    memory_decisions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReWOOExperienceLane:
    memories: list[ExperienceMemory] = field(default_factory=list)


@dataclass(slots=True)
class ReWOOLaneFrame:
    role: ReWOORole
    plan: ReWOOPlanLane = field(default_factory=ReWOOPlanLane)
    evidence: ReWOOEvidenceLane = field(default_factory=ReWOOEvidenceLane)
    decisions: ReWOODecisionLane = field(default_factory=ReWOODecisionLane)
    experiences: ReWOOExperienceLane = field(default_factory=ReWOOExperienceLane)
    drop_log: list[PromptStateDrop] = field(default_factory=list)

    def add_drop(self, *, lane: str, reason: str, dropped_count: int, dropped_ids: list[str] | None = None) -> None:
        if dropped_count <= 0 and not dropped_ids:
            return
        self.drop_log.append(
            PromptStateDrop(
                lane=lane,
                reason=reason,
                dropped_count=max(0, int(dropped_count)),
                dropped_ids=list(dropped_ids or []),
            )
        )


class ReWOOLaneCompiler:
    def __init__(self, policy: ContextPolicy | None = None) -> None:
        self.policy = policy or ContextPolicy()

    def compile(
        self,
        *,
        state: "LoopState",
        role: ReWOORole,
        retrieved_experiences: Iterable[ExperienceMemory] = (),
        tool_plan_observations: Iterable["ToolPlanObservation"] = (),
        token_budget: int | None = None,
    ) -> ReWOOLaneFrame:
        frame = ReWOOLaneFrame(role=role)
        frame.plan = self._compile_plan_lane(state, frame)
        frame.evidence = self._compile_evidence_lane(state, role, frame, tool_plan_observations)
        frame.decisions = self._compile_decision_lane(state, role, frame)
        frame.experiences = self._compile_experience_lane(state, role, frame, retrieved_experiences)
        self._fit_role_budgets(frame, token_budget=token_budget)
        self._fit_frame(frame, token_budget=token_budget)
        return frame

    def render(self, frame: ReWOOLaneFrame, *, token_budget: int | None = None) -> str:
        rendered = self._render_frame(frame)
        budget = token_budget if token_budget is not None else None
        if not budget or estimate_text_tokens(rendered) <= budget:
            return rendered
        working = replace(
            frame,
            plan=replace(frame.plan),
            evidence=replace(frame.evidence, packets=list(frame.evidence.packets), tool_plan_observations=list(frame.evidence.tool_plan_observations)),
            decisions=replace(frame.decisions, records=list(frame.decisions.records), memory_decisions=list(frame.decisions.memory_decisions)),
            experiences=replace(frame.experiences, memories=list(frame.experiences.memories)),
            drop_log=list(frame.drop_log),
        )
        self._fit_frame(working, token_budget=budget)
        return self._render_frame(working)

    def _compile_plan_lane(self, state: "LoopState", frame: ReWOOLaneFrame) -> ReWOOPlanLane:
        run_brief = state.run_brief
        memory = state.working_memory
        lane = ReWOOPlanLane(
            goal=_first_nonempty(memory.current_goal, run_brief.current_phase_objective, run_brief.original_task),
            task_contract=str(run_brief.task_contract or ""),
            constraints=_dedupe_strings([*run_brief.constraints]),
            acceptance_criteria=_dedupe_strings([*run_brief.acceptance_criteria]),
            active_plan=_dedupe_strings([*memory.plan, *run_brief.implementation_plan]),
            prior_failures=_dedupe_strings([*memory.failures]),
            open_questions=_dedupe_strings([*memory.open_questions]),
        )
        active_plan = state.active_plan
        if active_plan is not None:
            lane.active_plan = _dedupe_strings([*lane.active_plan, *active_plan.compact_lines()])
            active_step = active_plan.find_step(state.active_step_id) if state.active_step_id else active_plan.active_step()
            if active_step is not None:
                lane.active_step = active_step.compact_label()
                lane.pending_dependencies = _dedupe_strings(active_step.depends_on)
                lane.acceptance_criteria = _dedupe_strings([*lane.acceptance_criteria, *active_step.acceptance])
        for event in list(getattr(state, "failure_events", []) or [])[-6:]:
            message = str(getattr(event, "message", "") or "").strip()
            if message:
                lane.prior_failures.append(message[:260])
        nudge = str(state.scratchpad.get("_tool_plan_repair_nudge") or "").strip()
        if nudge:
            lane.repair_nudges.append(nudge[:600])
        lane.subtask_context = _extract_subtask_context(state)
        lane.hard_route_reasons = _extract_hard_route_reasons(state)
        lane.active_plan, dropped = _limit_strings(lane.active_plan, 10)
        frame.add_drop(lane=f"rewoo.{frame.role}.plan", reason="active plan item limit", dropped_count=dropped)
        lane.prior_failures, dropped = _limit_strings(lane.prior_failures, 6)
        frame.add_drop(lane=f"rewoo.{frame.role}.plan", reason="prior failure item limit", dropped_count=dropped)
        lane.subtask_context, dropped = _limit_strings(lane.subtask_context, 4)
        frame.add_drop(lane=f"rewoo.{frame.role}.plan", reason="subtask context item limit", dropped_count=dropped)
        lane.hard_route_reasons, dropped = _limit_strings(lane.hard_route_reasons, 5)
        frame.add_drop(lane=f"rewoo.{frame.role}.plan", reason="hard-route reason item limit", dropped_count=dropped)
        return lane

    def _compile_evidence_lane(
        self,
        state: "LoopState",
        role: ReWOORole,
        frame: ReWOOLaneFrame,
        tool_plan_observations: Iterable["ToolPlanObservation"],
    ) -> ReWOOEvidenceLane:
        packet_limit = 8 if role == "planner" else 12
        packets = build_observation_packets(state, limit=packet_limit)
        if role == "planner":
            packets = _filter_planner_packets(state, packets)
        observations = _coerce_tool_plan_observations(tool_plan_observations)
        if not observations:
            raw = state.scratchpad.get("_tool_plan_observations")
            if isinstance(raw, list):
                observations = _coerce_tool_plan_observations(raw)
        contradictions: list[str] = []
        gaps: list[str] = []
        for claim in list(getattr(state.reasoning_graph, "claim_records", []) or [])[-8:]:
            for gap in getattr(claim, "missing_evidence", []) or []:
                gaps.append(str(gap))
            for alt in getattr(claim, "alternative_explanations", []) or []:
                contradictions.append(str(alt))
        for packet in packets:
            if packet.kind in {"negative_observation", "tool_plan_negative_observation"} or packet.failure_mode:
                gaps.append(packet.summary)
        max_observations = 4 if role == "planner" else 16
        if role == "planner":
            observations = []
        observations, dropped_obs = _limit_items(observations, max_observations)
        frame.add_drop(lane=f"rewoo.{role}.evidence", reason="tool plan observation item limit", dropped_count=dropped_obs)
        packets, dropped_packets = _limit_items(packets, packet_limit)
        frame.add_drop(lane=f"rewoo.{role}.evidence", reason="observation packet item limit", dropped_count=dropped_packets)
        return ReWOOEvidenceLane(
            packets=packets,
            tool_plan_observations=observations,
            contradictions=_dedupe_strings(contradictions)[:6],
            gaps=_dedupe_strings(gaps)[:8],
        )

    def _compile_decision_lane(self, state: "LoopState", role: ReWOORole, frame: ReWOOLaneFrame) -> ReWOODecisionLane:
        records = list(getattr(state.reasoning_graph, "decision_records", []) or [])
        records.sort(key=lambda record: (record.status != "active", record.created_at), reverse=False)
        records = [record for record in records if role != "planner" or record.status == "active"]
        records, dropped_records = _limit_items(records[-8:], 8)
        frame.add_drop(lane=f"rewoo.{role}.decisions", reason="decision record item limit", dropped_count=dropped_records)
        memory_decisions, dropped_memory = _limit_strings(state.working_memory.decisions, 6)
        frame.add_drop(lane=f"rewoo.{role}.decisions", reason="memory decision item limit", dropped_count=dropped_memory)
        return ReWOODecisionLane(records=records, memory_decisions=memory_decisions)

    def _compile_experience_lane(
        self,
        state: "LoopState",
        role: ReWOORole,
        frame: ReWOOLaneFrame,
        retrieved_experiences: Iterable[ExperienceMemory],
    ) -> ReWOOExperienceLane:
        if role == "solver":
            return ReWOOExperienceLane()
        memories = _dedupe_memories([*list(retrieved_experiences), *list(state.warm_experiences or [])])
        memories.sort(key=lambda memory: (bool(memory.pinned), float(memory.confidence or 0.0), int(memory.reuse_count or 0)), reverse=True)
        limit = 4 if role == "planner" else 3
        memories, dropped = _limit_items(memories, limit)
        frame.add_drop(lane=f"rewoo.{role}.experiences", reason="experience item limit", dropped_count=dropped)
        return ReWOOExperienceLane(memories=memories)

    def _fit_frame(self, frame: ReWOOLaneFrame, *, token_budget: int | None) -> None:
        if not token_budget:
            return
        while estimate_text_tokens(self._render_frame(frame)) > token_budget:
            if frame.experiences.memories:
                memory = frame.experiences.memories.pop()
                frame.add_drop(lane=f"rewoo.{frame.role}.experiences", reason="token budget", dropped_count=1, dropped_ids=[memory.memory_id])
                continue
            if frame.evidence.packets:
                packet = frame.evidence.packets.pop(0)
                frame.add_drop(lane=f"rewoo.{frame.role}.evidence", reason="token budget", dropped_count=1, dropped_ids=[packet.observation_id])
                continue
            if frame.decisions.records:
                record = frame.decisions.records.pop(0)
                frame.add_drop(lane=f"rewoo.{frame.role}.decisions", reason="token budget", dropped_count=1, dropped_ids=[record.decision_id])
                continue
            if frame.evidence.tool_plan_observations:
                observation = frame.evidence.tool_plan_observations.pop(0)
                frame.add_drop(lane=f"rewoo.{frame.role}.evidence", reason="token budget", dropped_count=1, dropped_ids=[observation.step_id])
                continue
            if frame.plan.active_plan:
                frame.plan.active_plan.pop()
                frame.add_drop(lane=f"rewoo.{frame.role}.plan", reason="token budget", dropped_count=1)
                continue
            break

    def _fit_role_budgets(self, frame: ReWOOLaneFrame, *, token_budget: int | None) -> None:
        if not token_budget:
            return
        budget = max(1, int(token_budget))
        if frame.role == "planner":
            self._trim_evidence_to(frame, max_tokens=max(1, int(budget * 0.15)))
            self._trim_experiences_to(frame, max_tokens=max(1, int(budget * 0.20)))
            self._trim_planner_failures_to(frame, max_tokens=max(1, int(budget * 0.25)))
            self._trim_plan_to(frame, max_tokens=max(1, int(budget * 0.65)))
        elif frame.role == "solver":
            self._trim_evidence_to(frame, max_tokens=max(1, int(budget * 0.55)))
            self._trim_solver_gaps_to(frame, max_tokens=max(1, int(budget * 0.20)))
            self._trim_decisions_to(frame, max_tokens=max(1, int(budget * 0.15)))
            self._trim_plan_to(frame, max_tokens=max(1, int(budget * 0.25)))
        else:
            self._trim_evidence_to(frame, max_tokens=max(1, int(budget * 0.25)))
            self._trim_solver_gaps_to(frame, max_tokens=max(1, int(budget * 0.15)))
            self._trim_plan_to(frame, max_tokens=max(1, int(budget * 0.60)))

    def _trim_evidence_to(self, frame: ReWOOLaneFrame, *, max_tokens: int) -> None:
        while estimate_text_tokens(self._render_evidence(frame.evidence)) > max_tokens:
            if frame.evidence.packets:
                packet = frame.evidence.packets.pop(0)
                frame.add_drop(lane=f"rewoo.{frame.role}.evidence", reason="role evidence budget", dropped_count=1, dropped_ids=[packet.observation_id])
                continue
            if frame.evidence.tool_plan_observations:
                observation = frame.evidence.tool_plan_observations.pop(0)
                frame.add_drop(lane=f"rewoo.{frame.role}.evidence", reason="role evidence budget", dropped_count=1, dropped_ids=[observation.step_id])
                continue
            break

    def _trim_solver_gaps_to(self, frame: ReWOOLaneFrame, *, max_tokens: int) -> None:
        while estimate_text_tokens("\n".join([*frame.evidence.contradictions, *frame.evidence.gaps])) > max_tokens:
            if frame.evidence.contradictions:
                frame.evidence.contradictions.pop(0)
                frame.add_drop(lane=f"rewoo.{frame.role}.evidence", reason="role contradiction/gap budget", dropped_count=1)
                continue
            if frame.evidence.gaps:
                frame.evidence.gaps.pop(0)
                frame.add_drop(lane=f"rewoo.{frame.role}.evidence", reason="role contradiction/gap budget", dropped_count=1)
                continue
            break

    def _trim_decisions_to(self, frame: ReWOOLaneFrame, *, max_tokens: int) -> None:
        while estimate_text_tokens(self._render_decisions(frame.decisions)) > max_tokens:
            if frame.decisions.records:
                record = frame.decisions.records.pop(0)
                frame.add_drop(lane=f"rewoo.{frame.role}.decisions", reason="role decision budget", dropped_count=1, dropped_ids=[record.decision_id])
                continue
            if frame.decisions.memory_decisions:
                frame.decisions.memory_decisions.pop(0)
                frame.add_drop(lane=f"rewoo.{frame.role}.decisions", reason="role decision budget", dropped_count=1)
                continue
            break

    def _trim_experiences_to(self, frame: ReWOOLaneFrame, *, max_tokens: int) -> None:
        while estimate_text_tokens(self._render_experiences(frame.experiences)) > max_tokens and frame.experiences.memories:
            memory = frame.experiences.memories.pop()
            frame.add_drop(lane=f"rewoo.{frame.role}.experiences", reason="role experience budget", dropped_count=1, dropped_ids=[memory.memory_id])

    def _trim_planner_failures_to(self, frame: ReWOOLaneFrame, *, max_tokens: int) -> None:
        while estimate_text_tokens("\n".join([*frame.plan.prior_failures, *frame.plan.open_questions, *frame.plan.repair_nudges, *frame.plan.hard_route_reasons])) > max_tokens:
            if frame.plan.prior_failures:
                frame.plan.prior_failures.pop(0)
            elif frame.plan.open_questions:
                frame.plan.open_questions.pop(0)
            elif frame.plan.hard_route_reasons:
                frame.plan.hard_route_reasons.pop(0)
            elif frame.plan.repair_nudges:
                frame.plan.repair_nudges.pop(0)
            else:
                break
            frame.add_drop(lane=f"rewoo.{frame.role}.plan", reason="role failure/question budget", dropped_count=1)

    def _trim_plan_to(self, frame: ReWOOLaneFrame, *, max_tokens: int) -> None:
        while estimate_text_tokens(self._render_plan(frame.plan)) > max_tokens:
            if frame.plan.active_plan:
                frame.plan.active_plan.pop()
            elif frame.plan.subtask_context:
                frame.plan.subtask_context.pop()
            elif frame.plan.constraints:
                frame.plan.constraints.pop()
            else:
                break
            frame.add_drop(lane=f"rewoo.{frame.role}.plan", reason="role plan budget", dropped_count=1)

    def _render_frame(self, frame: ReWOOLaneFrame) -> str:
        sections = [
            self._render_plan(frame.plan),
            self._render_evidence(frame.evidence),
            self._render_decisions(frame.decisions),
            self._render_experiences(frame.experiences),
            self._render_role_rules(frame.role),
        ]
        return "\n\n".join(section for section in sections if section.strip()).rstrip()

    def _render_plan(self, lane: ReWOOPlanLane) -> str:
        lines = ["REWOO PLAN STATE"]
        _append_field(lines, "Goal", lane.goal)
        _append_field(lines, "Task contract", lane.task_contract)
        _append_list(lines, "Constraints", lane.constraints)
        _append_list(lines, "Acceptance criteria", lane.acceptance_criteria)
        _append_list(lines, "Active plan", lane.active_plan)
        _append_field(lines, "Active step", lane.active_step)
        _append_list(lines, "Pending dependencies", lane.pending_dependencies)
        _append_list(lines, "Prior failures", lane.prior_failures)
        _append_list(lines, "Open questions", lane.open_questions)
        _append_list(lines, "Repair nudges", lane.repair_nudges)
        _append_list(lines, "Subtask context", lane.subtask_context)
        _append_list(lines, "Hard-route reasons", lane.hard_route_reasons)
        return "\n".join(lines)

    def _render_evidence(self, lane: ReWOOEvidenceLane) -> str:
        lines = ["REWOO EVIDENCE"]
        for observation in lane.tool_plan_observations:
            status = "success" if observation.success else "failed"
            source = observation.path or observation.query or f"tool_plan:{observation.step_id}"
            evidence_id = f"TP:{observation.step_id}"
            bits = [f"- {evidence_id} [{status}] {observation.tool}"]
            if source:
                bits.append(f"source={source}")
            if observation.artifact_id:
                bits.append(f"artifact={observation.artifact_id}")
            if observation.operation_id:
                bits.append(f"operation={observation.operation_id}")
            if observation.duplicate_of:
                bits.append(f"duplicate_of={observation.duplicate_of}")
            summary = _lane_safe_text(observation.error or observation.summary)
            bits.append(f"summary={summary}")
            lines.append("; ".join(bits))
        for packet in lane.packets:
            source = packet.path or packet.query or packet.command or packet.artifact_id
            parts = [f"- {packet.observation_id or '(no-id)'} [{packet.kind}] {_lane_safe_text(packet.summary)}"]
            if packet.tool_name:
                parts.append(f"tool={packet.tool_name}")
            if source:
                parts.append(f"source={source}")
            if packet.operation_id:
                parts.append(f"operation={packet.operation_id}")
            if packet.artifact_id:
                parts.append(f"artifact={packet.artifact_id}")
            lines.append("; ".join(parts))
        _append_list(lines, "Contradictions", lane.contradictions)
        _append_list(lines, "Gaps", lane.gaps)
        if len(lines) == 1:
            lines.append("- none")
        return "\n".join(lines)

    def _render_decisions(self, lane: ReWOODecisionLane) -> str:
        lines = ["REWOO DECISIONS"]
        for record in lane.records:
            summary = record.rationale_summary or record.intent_label or record.requested_tool
            refs = ",".join(record.evidence_refs)
            status = record.status or "active"
            suffix = f"; evidence={refs}" if refs else ""
            lines.append(f"- {record.decision_id} [{status}] {summary}{suffix}")
        _append_list(lines, "Memory decisions", lane.memory_decisions)
        if len(lines) == 1:
            lines.append("- none")
        return "\n".join(lines)

    def _render_experiences(self, lane: ReWOOExperienceLane) -> str:
        lines = ["REWOO EXPERIENCES"]
        for memory in lane.memories:
            tags = ", ".join([*memory.intent_tags, *memory.environment_tags, *memory.entity_tags])
            guidance = memory.notes or memory.failure_mode or memory.outcome
            suffix = f"; tags={tags}" if tags else ""
            refs = f"; evidence={','.join(memory.evidence_refs)}" if memory.evidence_refs else ""
            lines.append(f"- {memory.memory_id} confidence={memory.confidence:.2f} reuse={memory.reuse_count}{suffix}{refs}: {guidance}")
        if len(lines) == 1:
            lines.append("- none")
        return "\n".join(lines)

    def _render_role_rules(self, role: ReWOORole) -> str:
        lines = ["REWOO ROLE RULES"]
        if role == "planner":
            lines.append("- Use this frame only to choose bounded read-only evidence steps.")
            lines.append("- Return ONLY ToolPlan JSON.")
            lines.append("- Do not include raw transcript or tool-output commentary.")
        elif role == "solver":
            lines.append("- Ground claims in REWOO EVIDENCE and active decisions.")
            lines.append("- Ask for a missing read when evidence is insufficient.")
            lines.append("- Prefer smaller next actions when failures or gaps remain.")
        else:
            lines.append("- Critique the solver draft against acceptance criteria and evidence refs.")
            lines.append("- Flag failed observations, contradictions, and unsupported claims.")
        return "\n".join(lines)


def _append_field(lines: list[str], label: str, value: str) -> None:
    value = str(value or "").strip()
    if value:
        lines.append(f"{label}: {value}")


def _append_list(lines: list[str], label: str, values: Iterable[str]) -> None:
    items = [_lane_safe_text(str(value).strip()) for value in values if str(value).strip()]
    if not items:
        return
    lines.append(f"{label}:")
    lines.extend(f"- {item}" for item in items)


def _first_nonempty(*values: str) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _limit_strings(values: Iterable[str], limit: int) -> tuple[list[str], int]:
    items = _dedupe_strings(values)
    if len(items) <= limit:
        return items, 0
    return items[:limit], len(items) - limit


def _limit_items(values: Iterable[T], limit: int) -> tuple[list[T], int]:
    items = list(values)
    if len(items) <= limit:
        return items, 0
    return items[-limit:], len(items) - limit


def _filter_planner_packets(state: "LoopState", packets: list[ObservationPacket]) -> list[ObservationPacket]:
    if not packets:
        return []
    text_scope = " ".join(
        _dedupe_strings(
            [
                state.working_memory.current_goal,
                state.run_brief.current_phase_objective,
                state.run_brief.original_task,
                *state.working_memory.plan,
                *state.working_memory.failures,
                *state.working_memory.open_questions,
                *state.run_brief.implementation_plan,
            ]
        )
    ).lower()
    relevant: list[ObservationPacket] = []
    fallback_failures: list[ObservationPacket] = []
    for packet in packets:
        if packet.kind in {"negative_observation", "tool_plan_negative_observation"} or packet.failure_mode:
            fallback_failures.append(packet)
            relevant.append(packet)
            continue
        source = " ".join(
            part for part in (packet.path, packet.command, packet.query, packet.artifact_id, packet.summary) if part
        ).lower()
        if _packet_mentions_scope(source, text_scope):
            relevant.append(packet)
    if relevant:
        return relevant
    return fallback_failures


def _packet_mentions_scope(source: str, text_scope: str) -> bool:
    if not source or not text_scope:
        return False
    source_terms = _scope_terms(source)
    scope_terms = _scope_terms(text_scope)
    return bool(source_terms & scope_terms)


def _scope_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw in text.replace("\\", "/").replace(":", " ").replace(",", " ").split():
        token = raw.strip().strip("()[]{}'\"`")
        if not token or len(token) < 3:
            continue
        if "/" in token or "." in token:
            terms.add(token.lower())
            terms.add(token.rsplit("/", 1)[-1].lower())
            continue
        if len(token) >= 5:
            terms.add(token.lower())
    return terms


def _extract_subtask_context(state: "LoopState") -> list[str]:
    ledger = getattr(state, "subtask_ledger", None)
    if ledger is None:
        return []
    active = None
    active_fn = getattr(ledger, "active", None)
    if callable(active_fn):
        try:
            active = active_fn()
        except Exception:
            active = None
    if active is None:
        active_id = str(getattr(ledger, "active_subtask_id", "") or "")
        for task in list(getattr(ledger, "subtasks", []) or []):
            if str(getattr(task, "subtask_id", "") or "") == active_id:
                active = task
                break
    tasks = [active] if active is not None else []
    if not tasks:
        tasks = [task for task in list(getattr(ledger, "subtasks", []) or []) if getattr(task, "status", "") == "active"][:1]
    lines: list[str] = []
    for task in tasks:
        title = str(getattr(task, "title", "") or getattr(task, "goal", "") or "").strip()
        status = str(getattr(task, "status", "") or "").strip()
        next_action = str(getattr(task, "next_action", "") or "").strip()
        blockers = _dedupe_strings(getattr(task, "blockers", []) or [])[:2]
        evidence = [_lane_safe_text(item) for item in _dedupe_strings(getattr(task, "evidence", []) or [])[:3]]
        parts = [part for part in (f"{status} subtask: {title}" if title else status, f"next={next_action}" if next_action else "") if part]
        if blockers:
            parts.append("blockers=" + ", ".join(blockers))
        if evidence:
            parts.append("evidence=" + ", ".join(evidence))
        if parts:
            lines.append("; ".join(parts)[:400])
    return _dedupe_strings(lines)


def _extract_hard_route_reasons(state: "LoopState") -> list[str]:
    lines: list[str] = []
    try:
        from ..fama.capsules import render_fama_capsules
        from ..fama.state import active_mitigations

        for mitigation in active_mitigations(state):
            name = str(getattr(mitigation, "name", "") or "").strip()
            reason = str(getattr(mitigation, "reason", "") or "").strip()
            if name or reason:
                lines.append(f"{name}: {reason}" if reason else name)
        lines.extend(render_fama_capsules(state, token_budget=180))
    except Exception:
        pass
    scratchpad = getattr(state, "scratchpad", {}) if isinstance(getattr(state, "scratchpad", {}), dict) else {}
    for key in ("_guard_trip_recovery_capsule", "_repair_continuity_capsule"):
        capsule = scratchpad.get(key)
        if not isinstance(capsule, dict):
            continue
        reason = str(capsule.get("reason") or capsule.get("failure_mode") or "").strip()
        action = str(capsule.get("next_suggested_action") or capsule.get("suggested_next_action") or "").strip()
        if reason and action:
            lines.append(f"{reason}; next={action}")
        elif reason:
            lines.append(reason)
        elif action:
            lines.append(f"next={action}")
    return _dedupe_strings(line[:400] for line in lines)


def _lane_safe_text(text: str) -> str:
    return (
        str(text or "")
        .replace("TOOL PLAN OBSERVATIONS", "ToolPlan evidence")
        .replace("ToolPlan observations", "ToolPlan evidence")
    )


def _dedupe_memories(memories: Iterable[ExperienceMemory]) -> list[ExperienceMemory]:
    seen: set[str] = set()
    result: list[ExperienceMemory] = []
    for memory in memories:
        memory_id = str(memory.memory_id or "").strip()
        key = memory_id or memory.notes
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(memory)
    return result


def _coerce_tool_plan_observations(values: Iterable[object]) -> list["ToolPlanObservation"]:
    from ..graph.tool_plan_observations import ToolPlanObservation

    observations: list[ToolPlanObservation] = []
    for value in values:
        if isinstance(value, ToolPlanObservation):
            observations.append(value)
            continue
        if not isinstance(value, dict):
            continue
        observations.append(
            ToolPlanObservation(
                step_id=str(value.get("step_id") or ""),
                tool=str(value.get("tool") or ""),
                success=bool(value.get("success", False)),
                summary=str(value.get("summary") or ""),
                artifact_id=str(value.get("artifact_id") or ""),
                operation_id=str(value.get("operation_id") or ""),
                path=str(value.get("path") or ""),
                query=str(value.get("query") or ""),
                error=str(value.get("error") or ""),
                duplicate_of=str(value.get("duplicate_of") or ""),
            )
        )
    return observations
