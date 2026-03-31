from __future__ import annotations

from dataclasses import dataclass


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    # Conservative estimation for agentic prompts (special chars, code, JSON)
    # Using 0.4 tokens per character (approx // 2.5) to avoid overfilling tight limits.
    return max(1, int(len(text) * 0.4) + 1)


@dataclass
class ContextPolicy:
    max_prompt_tokens: int | None = None
    reserve_completion_tokens: int = 1024
    reserve_tool_tokens: int = 512
    summarize_at_ratio: float = 0.8
    recent_message_limit: int = 20
    transcript_token_limit: int = 1400
    run_brief_token_limit: int = 240
    working_memory_token_limit: int = 360
    episodic_summary_token_limit: int = 320
    artifact_snippet_section_token_limit: int = 520
    max_summary_items: int = 3
    max_artifact_snippets: int = 4
    artifact_snippet_token_limit: int = 400
    artifact_summarization_threshold: int = 1200
    prior_outcome_token_limit: int = 220
    max_prior_outcomes: int = 4
    tool_result_inline_token_limit: int = 250
    memory_staleness_step_limit: int = 8
    memory_low_confidence_threshold: float = 0.6
    
    # Tier-aware limits (#1 + #4)
    hot_message_limit: int = 8
    warm_brief_limit: int = 3
    warm_tier_token_budget: int = 400
    cold_fact_limit: int = 12
    compaction_step_interval: int = 8

    # Ratio-based scaling fields (Phase I)
    hot_history_ratio: float = 0.45     # share of soft_limit for verbatim transcript
    retrieval_ratio: float = 0.20       # share for cold/warm retrieved content
    warm_tier_ratio: float = 0.10       # share for warm briefs + summaries overlay
    reserve_ratio: float = 0.25         # kept for completion (already in soft_limit calc)
    
    # Scaling constants
    messages_per_k_tokens: float = 1.0  # hot window growth: 1 msg per 1k tokens
    min_hot_messages: int = 8
    max_hot_messages: int = 256
    compaction_strategy: str = "lazy"   # "lazy" | "aggressive"

    @property
    def section_token_limits(self) -> dict[str, int]:
        return {
            "run_brief": self.run_brief_token_limit,
            "working_memory": self.working_memory_token_limit,
            "episodic_summaries": self.episodic_summary_token_limit,
            "prior_outcomes": self.prior_outcome_token_limit,
            "artifact_snippets": self.artifact_snippet_section_token_limit,
            "recent_messages": self.transcript_token_limit,
        }

    @property
    def soft_prompt_token_limit(self) -> int | None:
        limit = self.max_prompt_tokens
        if limit is None:
            return None
        
        # Scale reserves if the limit is tight
        reserve_comp = min(self.reserve_completion_tokens, limit // 3)
        reserve_tool = min(self.reserve_tool_tokens, limit // 6)
        
        return max(
            limit // 4,  # Always leave at least 25% for prompt
            int(limit - reserve_comp - reserve_tool),
        )

    def recalculate_quotas(self, total_context: int) -> None:
        """Called once after server context discovery. Recalculates all section
        token budgets from ratios so the policy is internally consistent."""
        soft = self.soft_prompt_token_limit or total_context
        if soft is None:
            return

        # Token budgets per section
        self.transcript_token_limit = int(soft * self.hot_history_ratio)
        combined_retrieval = int(soft * self.retrieval_ratio)
        self.artifact_snippet_section_token_limit = int(combined_retrieval * 0.6)
        self.episodic_summary_token_limit = int(combined_retrieval * 0.25)
        self.working_memory_token_limit = int(combined_retrieval * 0.15)
        self.warm_tier_token_budget = int(soft * self.warm_tier_ratio)

        # Hot message window — linear scaling
        self.hot_message_limit = max(
            self.min_hot_messages,
            min(self.max_hot_messages, int(total_context / 1000 * self.messages_per_k_tokens)),
        )
        self.recent_message_limit = self.hot_message_limit

        # Compaction interval — scale with window size
        self.compaction_step_interval = max(4, self.hot_message_limit // 4)
