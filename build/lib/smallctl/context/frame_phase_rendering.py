from __future__ import annotations

from ..state import LoopState


def render_phase_context(state: LoopState) -> list[str]:
    phase = state.contract_phase()
    sections: list[str] = []
    reasoning = state.reasoning_graph
    if phase == "explore":
        sections.append("Phase handoff: explore -> plan")
        if state.context_briefs:
            brief = state.context_briefs[-1]
            brief_bits = [f"brief={brief.brief_id}", f"phase={brief.current_phase}"]
            if brief.key_discoveries:
                brief_bits.append("facts=" + "; ".join(brief.key_discoveries[:4]))
            if brief.open_questions:
                brief_bits.append("questions=" + "; ".join(brief.open_questions[:3]))
            if brief.evidence_refs:
                brief_bits.append("evidence=" + ", ".join(brief.evidence_refs[:5]))
            sections.append("Explore brief: " + " | ".join(brief_bits))
    elif phase == "plan":
        sections.append("Phase handoff: plan from compressed evidence")
        evidence_bits = []
        for record in reasoning.evidence_records[-5:]:
            statement = record.statement.strip()
            if statement:
                evidence_bits.append(f"{record.evidence_id}: {statement}")
        if evidence_bits:
            sections.append("Evidence packet: " + " | ".join(evidence_bits))
        if reasoning.claim_records:
            claim_bits = []
            for claim in reasoning.claim_records[-4:]:
                claim_bits.append(f"{claim.claim_id} [{claim.status}] {claim.statement}".strip())
            if claim_bits:
                sections.append("Claims: " + " | ".join(claim_bits))
        plan = state.active_plan or state.draft_plan
        if plan is not None and getattr(plan, "claim_refs", None):
            sections.append("Plan claim refs: " + ", ".join(plan.claim_refs[:5]))
    elif phase == "author":
        sections.append("Phase handoff: plan -> author")
        plan = state.active_plan or state.draft_plan
        if plan is not None:
            sections.append(f"Authoring plan: {plan.plan_id} | {plan.goal}")
            active_step = plan.active_step()
            if active_step is not None:
                sections.append(
                    f"Current step: {active_step.step_id} [{active_step.status}] {active_step.title}"
                )
                if active_step.claim_refs:
                    sections.append("Current step claims: " + ", ".join(active_step.claim_refs[:5]))
            if plan.acceptance_criteria:
                sections.append("Acceptance: " + "; ".join(plan.acceptance_criteria[:4]))
            if getattr(plan, "claim_refs", None):
                sections.append("Plan claims: " + ", ".join(plan.claim_refs[:5]))
            if plan.requested_output_path:
                sections.append(f"Target output: {plan.requested_output_path}")
    elif phase == "execute":
        sections.append("Phase handoff: author -> execute")
        plan = state.active_plan or state.draft_plan
        if plan is not None:
            sections.append(f"Execution plan: {plan.plan_id} [{plan.status}]")
            if getattr(plan, "claim_refs", None):
                sections.append("Plan claims: " + ", ".join(plan.claim_refs[:5]))
        if state.reasoning_graph.evidence_records:
            evidence_ids = [record.evidence_id for record in state.reasoning_graph.evidence_records[-5:]]
            sections.append("Evidence refs: " + ", ".join(evidence_ids))
    elif phase == "verify":
        sections.append("Phase handoff: execute -> verify")
        if state.current_verifier_verdict():
            sections.append("Verifier verdict present")
        if state.run_brief.acceptance_criteria:
            sections.append("Acceptance criteria: " + "; ".join(state.run_brief.acceptance_criteria[:4]))
    elif phase == "repair":
        sections.append("Phase handoff: verify -> repair")
        if state.last_failure_class:
            sections.append(f"Failure class: {state.last_failure_class}")
        if state.files_changed_this_cycle:
            sections.append("Files changed: " + ", ".join(state.files_changed_this_cycle[-5:]))
        if state.write_session:
            sections.append(f"Write session: {state.write_session.write_session_id}")
    return sections
