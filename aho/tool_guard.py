"""Tool call guard to validate tool names and prevent hallucinations."""
import difflib
from typing import Any, Awaitable, Callable
from src.smallctl.models.tool_result import ToolEnvelope

VALID_TOOLS = {
    "long_context_lookup", 
    "summarize_report", 
    "artifact_read",
    "memory_update", 
    "task_complete",
    "task_fail",
    "loop_status",
    "review_draft",
    "weather_lookup",
    "clothing_suggest"
}

class ToolCallGuard:
    def __init__(self, valid_tools: set = None, fuzzy_threshold: float = 0.6):
        self.valid_tools = valid_tools or VALID_TOOLS
        self.fuzzy_threshold = fuzzy_threshold
        self.correction_count = 0
        self.hallucination_count = 0
        
    def validate(self, tool_name: str) -> dict:
        """
        Validate a tool name.
        Returns: {"valid": True} or {"valid": False, "action": "correct", "suggestion": "..."}
        """
        if tool_name in self.valid_tools:
            return {"valid": True}
        
        self.hallucination_count += 1
        
        # Try fuzzy matching
        suggestion = self._fuzzy_match(tool_name)
        if suggestion:
            self.correction_count += 1
            return {
                "valid": False,
                "action": "correct",
                "message": f"Tool '{tool_name}' not found. Did you mean '{suggestion}'?",
                "suggestion": suggestion
            }
        
        return {
            "valid": False,
            "action": "reject",
            "message": f"Tool '{tool_name}' does not exist. Available tools: {', '.join(sorted(self.valid_tools))}"
        }
    
    def _fuzzy_match(self, tool_name: str) -> str | None:
        """Find the closest matching valid tool name."""
        matches = difflib.get_close_matches(
            tool_name, 
            self.valid_tools, 
            n=1, 
            cutoff=self.fuzzy_threshold
        )
        return matches[0] if matches else None
    
    def get_stats(self) -> dict:
        return {
            "corrections": self.correction_count,
            "hallucinations": self.hallucination_count
        }
    async def __call__(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        next_dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolEnvelope]],
    ) -> ToolEnvelope:
        """Interceptor entry point."""
        validation = self.validate(tool_name)
        if not validation["valid"]:
            if validation["action"] == "correct":
                actual_tool = validation["suggestion"]
                return await next_dispatch(actual_tool, arguments)
            
            # Reject
            return ToolEnvelope(
                success=False,
                error=validation["message"],
                metadata={"hallucinated_tool": tool_name}
            )
        
        return await next_dispatch(tool_name, arguments)
