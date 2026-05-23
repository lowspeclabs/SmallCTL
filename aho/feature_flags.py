"""Feature flag system for gradual rollout of inline tool results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InlineResultsConfig:
    """
    Configuration for inline tool results feature.
    
    This controls how tool results are returned to the LLM:
    - Traditional: Full document stored as artifact, requires artifact_read
    - Inline: Key facts extracted and returned directly
    """
    
    enabled: bool = True
    mode: str = "smart"  # "always", "smart", "never"
    
    # Feature toggles
    distilled_by_default: bool = True
    dynamic_fact_extraction: bool = True
    fact_validation: bool = True
    confidence_threshold: float = 0.8
    
    # Tool-specific overrides
    tool_overrides: dict[str, dict] = field(default_factory=dict)
    
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "InlineResultsConfig":
        """Create config from harness configuration dict."""
        inline_cfg = cfg.get("inline_tool_results", {})
        
        return cls(
            enabled=inline_cfg.get("enabled", True),
            mode=inline_cfg.get("mode", "smart"),
            distilled_by_default=inline_cfg.get("features", {}).get(
                "distilled_by_default", True
            ),
            dynamic_fact_extraction=inline_cfg.get("features", {}).get(
                "dynamic_fact_extraction", True
            ),
            fact_validation=inline_cfg.get("features", {}).get(
                "fact_validation", True
            ),
            confidence_threshold=inline_cfg.get("features", {}).get(
                "confidence_threshold", 0.8
            ),
            tool_overrides=inline_cfg.get("tool_specific", {}),
        )
    
    def should_use_inline(self, tool_name: str, context: dict | None = None) -> bool:
        """
        Determine if a tool call should use inline results.
        
        Args:
            tool_name: Name of the tool being called
            context: Optional context about the call
            
        Returns:
            True if inline results should be used
        """
        if not self.enabled:
            return False
        
        # Check tool-specific override
        tool_config = self.tool_overrides.get(tool_name, {})
        if "distilled_default" in tool_config:
            return tool_config["distilled_default"]
        
        # Apply mode logic
        if self.mode == "always":
            return True
        elif self.mode == "never":
            return False
        elif self.mode == "smart":
            return self._smart_decision(tool_name, context)
        
        return self.distilled_by_default
    
    def _smart_decision(self, tool_name: str, context: dict | None = None) -> bool:
        """
        Smart decision based on tool type and context.
        
        Use inline for:
        - Retrieval tools (lookup, summarize, search)
        - When context suggests need for facts only
        
        Use full document for:
        - Code/file reading
        - Complex analysis
        - When user explicitly requests full text
        """
        # Tools that benefit from inline results
        inline_friendly_tools = {
            "long_context_lookup",
            "summarize_report",
            "web_search",
            "kbase_query",
        }
        
        # Tools that need full content
        full_content_tools = {
            "file_read",
            "code_read",
            "document_parse",
        }
        
        if tool_name in inline_friendly_tools:
            return True
        if tool_name in full_content_tools:
            return False
        
        # Check context hints
        if context:
            user_request = context.get("user_request", "").lower()
            if any(word in user_request for word in ["full", "complete", "entire"]):
                return False
            if any(word in user_request for word in ["fact", "key", "summary"]):
                return True
        
        # Default to inline for unknown tools
        return self.distilled_by_default
    
    def get_max_facts(self, tool_name: str) -> int:
        """Get maximum facts to extract for a tool."""
        tool_config = self.tool_overrides.get(tool_name, {})
        return tool_config.get("max_facts", 5)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "features": {
                "distilled_by_default": self.distilled_by_default,
                "dynamic_fact_extraction": self.dynamic_fact_extraction,
                "fact_validation": self.fact_validation,
                "confidence_threshold": self.confidence_threshold,
            },
            "tool_specific": self.tool_overrides,
        }


@dataclass
class FeatureFlags:
    """Global feature flags for AHO harness."""
    
    inline_results: InlineResultsConfig
    progress_guard_v2: bool = False
    context_compression_v2: bool = False
    
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "FeatureFlags":
        """Create feature flags from configuration."""
        return cls(
            inline_results=InlineResultsConfig.from_config(cfg),
            progress_guard_v2=cfg.get("progress_guard_v2", False),
            context_compression_v2=cfg.get("context_compression_v2", False),
        )


# Convenience function
def should_inline_results(tool_name: str, cfg: dict[str, Any]) -> bool:
    """Quick check if inline results should be used."""
    config = InlineResultsConfig.from_config(cfg)
    return config.should_use_inline(tool_name)
