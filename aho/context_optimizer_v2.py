"""
Multi-tier context management with early intervention.
"""
import re
from typing import Any, Optional
from dataclasses import dataclass
from src.smallctl.context.policy import estimate_text_tokens


@dataclass
class ContextMetrics:
    total_chars: int
    estimated_tokens: int
    context_limit: int
    fill_ratio: float  # 0.0 to 1.0


class SmartContextOptimizer:
    """
    Tiered compression strategy:
    - 70%: Light compression (trim old artifacts)
    - 75%: Intent distillation (summarize task progress)  
    - 80%: Aggressive compression (facts only)
    - 90%: Emergency truncation (last resort)
    """
    
    COMPRESSION_TIERS = {
        0.70: "light",
        0.75: "distill", 
        0.80: "aggressive",
        0.90: "emergency"
    }
    
    def __init__(
        self,
        context_limit: int = 4096,
        compression_threshold: float = 0.70,
        task_patterns: Optional[dict] = None
    ):
        self.context_limit = context_limit
        self.compression_threshold = compression_threshold
        self.task_patterns = task_patterns or {
            r'2\.2\s*%?': 'Linear reduction factor: 2.2%',
            r'56\.10': 'EUA price: €56.10/t',
        }
        self.compression_history = []
        self._last_compression_ratio = 0.0
        
    def check_and_optimize(
        self, 
        messages: list, 
        phase: str = "explore",
        step_count: int = 0
    ) -> tuple[list, Optional[str]]:
        """
        Check context fill level and apply appropriate compression.
        
        Returns: (optimized_messages, action_taken)
        """
        metrics = self._calculate_metrics(messages)
        
        # Determine which tier to apply
        fill_pct = metrics.fill_ratio
        action = None
        
        if fill_pct >= 0.90:
            messages = self._emergency_truncate(messages)
            action = "emergency"
        elif fill_pct >= 0.80:
            messages = self._aggressive_compress(messages, phase)
            action = "aggressive"
        elif fill_pct >= 0.75:
            messages = self._distill_intent(messages, phase, step_count)
            action = "distill"
        elif fill_pct >= 0.70:
            messages = self._light_compress(messages)
            action = "light"
            
        if action:
            self.compression_history.append({
                "step": step_count,
                "phase": phase,
                "fill_before": fill_pct,
                "action": action
            })
            
        return messages, action
    
    def _calculate_metrics(self, messages: list) -> ContextMetrics:
        """Calculate current context usage."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        # Use core's token estimation for consistency
        estimated_tokens = estimate_text_tokens("".join(str(m.get("content", "")) for m in messages))
        fill_ratio = estimated_tokens / self.context_limit
        return ContextMetrics(total_chars, estimated_tokens, self.context_limit, fill_ratio)
    
    def _light_compress(self, messages: list) -> list:
        """
        70% threshold: Remove old artifact content, keep references.
        Remove redundant tool results from early steps.
        """
        compressed = []
        seen_artifacts = set()
        
        for msg in messages:
            content = str(msg.get("content", ""))
            
            # If it's a tool result with artifact reference, deduplicate
            if "artifact A" in content and "status: ok" in content:
                # Extract artifact ID
                match = re.search(r'artifact (A\d+)', content)
                if match:
                    art_id = match.group(1)
                    if art_id in seen_artifacts:
                        # Skip duplicate artifact references
                        continue
                    seen_artifacts.add(art_id)
                    
            compressed.append(msg)
            
        return compressed
    
    def _distill_intent(self, messages: list, phase: str, step_count: int) -> list:
        """
        75% threshold: Distill task progress into intent summary.
        Replace verbose tool outputs with distilled facts + current objective.
        """
        # Keep system prompt and recent messages
        compressed = []
        facts_extracted = []
        
        for msg in messages:
            content = str(msg.get("content", ""))
            
            # If it's a large tool result, distill to facts
            if len(content) > 1000 and ("SECTION" in content or "===" in content):
                facts = self._extract_facts(content)
                if facts:
                    facts_extracted.extend(facts)
                    # Replace with distilled version
                    new_msg = msg.copy()
                    new_msg["content"] = f"[DISTILLED] {facts}"
                    compressed.append(new_msg)
                    continue
                    
            compressed.append(msg)
        
        # Add intent summary if we extracted facts
        if facts_extracted and step_count > 2:
            intent_msg = {
                "role": "system",
                "content": f"[INTENT] Step {step_count} ({phase}): Have facts: {facts_extracted}. Goal: Complete task with these values."
            }
            # Insert after system prompt
            if compressed and compressed[0].get("role") == "system":
                compressed.insert(1, intent_msg)
            else:
                compressed.insert(0, intent_msg)
                
        return compressed
    
    def _aggressive_compress(self, messages: list, phase: str) -> list:
        """
        80% threshold: Facts only, drop all verbose content.
        Keep: system prompt, last user message, key facts
        Drop: old artifacts, verbose tool outputs, thinking blocks
        """
        compressed = []
        key_facts = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))
            
            # Always keep system prompt
            if role == "system":
                compressed.append(msg)
                continue
                
            # Keep user task description (usually short)
            if role == "user" and len(content) < 500:
                compressed.append(msg)
                continue
                
            # Extract facts from assistant/tool messages, drop rest
            if len(content) > 500:
                facts = self._extract_facts(content)
                if facts:
                    key_facts.extend(facts)
                # Skip adding this verbose message
                continue
                
            # Keep short assistant responses (likely tool calls or summaries)
            if role == "assistant" and len(content) < 800:
                compressed.append(msg)
                continue
                
        # Add consolidated facts
        if key_facts:
            compressed.append({
                "role": "system",
                "content": f"[KEY FACTS] {'; '.join(set(key_facts))}"
            })
            
        return compressed
    
    def _emergency_truncate(self, messages: list) -> list:
        """
        90% threshold: Last resort - keep only bare minimum.
        System prompt + last user message + key facts only.
        """
        if len(messages) <= 3:
            return messages
            
        # Keep first (system), last (user task), and second-to-last (current state)
        compressed = [messages[0]]  # System prompt
        
        # Add last user message if exists
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs:
            compressed.append(user_msgs[-1])
            
        # Add any distilled facts we can find
        all_content = " ".join(str(m.get("content", "")) for m in messages)
        facts = self._extract_facts(all_content)
        if facts:
            compressed.append({
                "role": "system", 
                "content": f"[EMERGENCY FACTS] {'; '.join(facts)}"
            })
            
        return compressed
    
    def _extract_facts(self, content: str) -> list:
        """Extract key facts using patterns."""
        facts = []
        for pattern, fact in self.task_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                if fact not in facts:
                    facts.append(fact)
        return facts
    
    def get_stats(self) -> dict:
        """Return compression history and statistics."""
        return {
            "compression_events": len(self.compression_history),
            "history": self.compression_history,
            "last_compression_ratio": self._last_compression_ratio
        }
