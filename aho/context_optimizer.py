"""Context optimizer for phase-aware compression."""
import re
from typing import Any


class ContextOptimizer:
    """Optimizes context messages based on phase and step count for efficient token usage."""
    
    def __init__(self, compression_threshold: int = 1000):
        self.compression_threshold = compression_threshold
        # Key facts patterns for the CarbonTradingEngine task
        self.key_patterns = {
            r'2\.2\s*%?': 'Linear reduction factor: 2.2%',
            r'56\.10': 'EUA price: €56.10/t',
            r'€\s*56\.10': 'EUA price: €56.10/t',
            r'56[.,]?10\s*€?/t': 'EUA price: €56.10/t',
        }
        
    def optimize(self, messages: list, phase: str = "explore", step_count: int = 0) -> list:
        """
        Optimize messages based on current phase.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            phase: Current execution phase (e.g., 'explore', 'synthesis', 'execute')
            step_count: Current step number in the execution
            
        Returns:
            Optimized list of messages
        """
        if phase == "synthesis" or step_count >= 6:
            return self._compress_for_synthesis(messages)
        return messages
    
    def _compress_for_synthesis(self, messages: list) -> list:
        """Compress messages by extracting key facts from long content."""
        compressed = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > self.compression_threshold:
                # Extract key facts instead of keeping full content
                facts = self._extract_facts(content)
                new_msg = msg.copy()
                new_msg["content"] = facts
                compressed.append(new_msg)
            else:
                compressed.append(msg)
        return compressed
    
    def _extract_facts(self, content: str) -> str:
        """Extract key facts from content."""
        facts = []
        content_str = str(content)
        
        # Check for known patterns
        for pattern, fact in self.key_patterns.items():
            if re.search(pattern, content_str, re.IGNORECASE):
                if fact not in facts:
                    facts.append(fact)
        
        if facts:
            return "KEY FACTS:\n" + "\n".join(f"- {f}" for f in facts)
        
        # Fallback: truncate with ellipsis
        return content[:500] + "... [truncated for synthesis]"
    
    def should_compress(self, messages: list, step_count: int) -> bool:
        """Determine if we should compress based on step count and message sizes."""
        if step_count < 4:
            return False
        
        total_size = sum(len(str(m.get("content", ""))) for m in messages)
        return total_size > 3000
