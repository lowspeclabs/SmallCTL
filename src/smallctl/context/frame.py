from __future__ import annotations

from dataclasses import dataclass, field

from ..state import ArtifactSnippet, ContextBrief, EpisodicSummary, ExperienceMemory, TurnBundle
from .observations import ObservationPacket


@dataclass(slots=True)
class PromptStateSpine:
    cwd: str = ""
    task_goal: str = ""
    task_contract: str = ""
    current_phase: str = ""
    phase_focus: str = ""
    active_step: str = ""
    active_intent: str = ""
    unmet_acceptance_criteria: list[str] = field(default_factory=list)
    known_good_facts: list[str] = field(default_factory=list)
    current_blockers: list[str] = field(default_factory=list)
    next_allowed_action: str = ""
    files_in_play: list[str] = field(default_factory=list)
    write_session_summary: str = ""
    coding_anchor_lines: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    run_brief_text: str = ""
    working_memory_text: str = ""
    fama_capsule_lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PromptPhasePacket:
    lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PromptEvidencePacket:
    observations: list[ObservationPacket] = field(default_factory=list)
    turn_bundles: list[TurnBundle] = field(default_factory=list)
    context_briefs: list[ContextBrief] = field(default_factory=list)
    summaries: list[EpisodicSummary] = field(default_factory=list)


@dataclass(slots=True)
class PromptExperiencePacket:
    memories: list[ExperienceMemory] = field(default_factory=list)


@dataclass(slots=True)
class PromptArtifactPacket:
    snippets: list[ArtifactSnippet] = field(default_factory=list)


@dataclass(slots=True)
class PromptStateDrop:
    lane: str
    reason: str
    dropped_count: int = 0
    dropped_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PromptStateFrame:
    spine: PromptStateSpine = field(default_factory=PromptStateSpine)
    phase_packet: PromptPhasePacket = field(default_factory=PromptPhasePacket)
    evidence_packet: PromptEvidencePacket = field(default_factory=PromptEvidencePacket)
    experience_packet: PromptExperiencePacket = field(default_factory=PromptExperiencePacket)
    artifact_packet: PromptArtifactPacket = field(default_factory=PromptArtifactPacket)
    drop_log: list[PromptStateDrop] = field(default_factory=list)

    def add_drop(
        self,
        *,
        lane: str,
        reason: str,
        dropped_count: int = 0,
        dropped_ids: list[str] | None = None,
    ) -> None:
        self.drop_log.append(
            PromptStateDrop(
                lane=lane,
                reason=reason,
                dropped_count=max(0, int(dropped_count)),
                dropped_ids=list(dropped_ids or []),
            )
        )

    def included_lane_counts(self) -> dict[str, int]:
        return {
            "phase_packet": len(self.phase_packet.lines),
            "normalized_observations": len(self.evidence_packet.observations),
            "turn_bundles": len(self.evidence_packet.turn_bundles),
            "context_briefs": len(self.evidence_packet.context_briefs),
            "episodic_summaries": len(self.evidence_packet.summaries),
            "artifact_snippets": len(self.artifact_packet.snippets),
            "experience_memories": len(self.experience_packet.memories),
        }

    def dropped_lane_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for drop in self.drop_log:
            counts[drop.lane] = counts.get(drop.lane, 0) + max(drop.dropped_count, len(drop.dropped_ids))
        return counts
