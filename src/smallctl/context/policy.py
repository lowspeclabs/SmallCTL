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
    multi_file_artifact_snippet_limit: int = 8
    multi_file_primary_file_limit: int = 3
    remote_task_artifact_snippet_limit: int = 8
    remote_task_primary_file_limit: int = 2
    artifact_summarization_threshold: int = 1560
    prior_outcome_token_limit: int = 220
    max_prior_outcomes: int = 4
    tool_result_inline_token_limit: int = 325
    artifact_read_inline_token_limit: int = 1024
    memory_staleness_step_limit: int = 8
    memory_low_confidence_threshold: float = 0.6
    
    # Tier-aware limits (#1 + #4)
    hot_message_limit: int = 8
    warm_brief_limit: int = 3
    warm_tier_token_budget: int = 400
    observation_token_limit: int = 768
    observation_token_floor: int = 2048
    min_observation_items: int = 3
    max_observation_items: int = 8
    fresh_tool_output_token_limit: int = 1200
    fresh_tool_output_items: int = 4
    turn_bundle_limit: int = 6
    coding_profile_enabled: bool = True
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
    backend_profile: str = "generic"

    def scaled_inline_limit(
        self,
        *,
        soft_limit: int | None,
        ratio: float,
        minimum: int,
        maximum: int,
    ) -> int:
        if soft_limit is None or soft_limit <= 0:
            return minimum
        scaled = int(soft_limit * ratio)
        return max(minimum, min(maximum, scaled))

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

    def apply_backend_profile(self, backend_profile: str | None) -> None:
        profile = str(backend_profile or "generic").strip().lower()
        self.backend_profile = profile

        if profile in {"lmstudio", "ollama", "llamacpp"}:
            self.messages_per_k_tokens = 0.8
            self.hot_history_ratio = 0.40
            self.retrieval_ratio = 0.18
            self.warm_tier_ratio = 0.08
            self.min_hot_messages = 6
        elif profile in {"openai", "openrouter", "vllm"}:
            self.messages_per_k_tokens = 1.2
            self.hot_history_ratio = 0.48
            self.retrieval_ratio = 0.22
            self.warm_tier_ratio = 0.12
            self.min_hot_messages = 8
        else:
            self.messages_per_k_tokens = 1.0
            self.hot_history_ratio = 0.45
            self.retrieval_ratio = 0.20
            self.warm_tier_ratio = 0.10
            self.min_hot_messages = 8

    def apply_model_profile(self, model_name: str | None) -> None:
        """Tune context pressure for small local models after provider defaults."""
        from ..guards import is_four_b_or_under_model_name

        if not is_four_b_or_under_model_name(model_name):
            return
        if self.backend_profile not in {"lmstudio", "ollama", "llamacpp"}:
            return

        self.messages_per_k_tokens = min(self.messages_per_k_tokens, 0.60)
        self.hot_history_ratio = min(self.hot_history_ratio, 0.34)
        self.retrieval_ratio = min(self.retrieval_ratio, 0.16)
        self.warm_tier_ratio = min(self.warm_tier_ratio, 0.08)
        self.min_hot_messages = min(self.min_hot_messages, 6)
        self.max_hot_messages = min(self.max_hot_messages, 24)
        self.compaction_strategy = "aggressive"

    def recalculate_quotas(self, total_context: int, *, backend_profile: str | None = None) -> None:
        """Called once after server context discovery. Recalculates all section
        token budgets from ratios so the policy is internally consistent."""
        if backend_profile is not None:
            self.apply_backend_profile(backend_profile)

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
        self.tool_result_inline_token_limit = self.scaled_inline_limit(
            soft_limit=soft,
            ratio=0.02,
            minimum=325,
            maximum=8000,
        )
        self.artifact_read_inline_token_limit = self.scaled_inline_limit(
            soft_limit=soft,
            ratio=0.04,
            minimum=1024,
            maximum=16000,
        )

        # Hot message window — linear scaling
        self.hot_message_limit = max(
            self.min_hot_messages,
            min(self.max_hot_messages, int(total_context / 1000 * self.messages_per_k_tokens)),
        )
        self.recent_message_limit = self.hot_message_limit

        # Compaction interval — scale with window size
        self.compaction_step_interval = max(4, self.hot_message_limit // 4)
