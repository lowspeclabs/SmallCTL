"""Progress guard - Hybrid: Context Fingerprinting + Adaptive Recovery."""
import hashlib
from typing import Any, Awaitable, Callable
from src.smallctl.models.tool_result import ToolEnvelope


class ProgressGuard:
    """
    Hybrid approach combining:
    1. Context Fingerprinting - Only detect loops when context hasn't changed
    2. Adaptive Recovery - Escalating nudges instead of immediate failure
    
    This gives accurate detection without false positives, plus lets the LLM
    self-correct before we fail the task.
    """
    
    def __init__(self, message_provider: Callable[[], list] | None = None,
                 window_size: int = 6, max_loops: int = 2,
                 context_window: int = 3, max_nudges: int = 2,
                 nudge_escalation: bool = True):
        self.message_provider = message_provider
        self.window_size = window_size
        self.max_loops = max_loops
        self.context_window = context_window
        self.max_nudges = max_nudges
        self.nudge_escalation = nudge_escalation
        
        self.tool_history = []
        self.context_history = []
        self.suspicion_count = {}  # call_sig@context_sig -> suspicion count
        self.total_nudges = 0
        
    def _compute_context_fingerprint(self, conversation_history: list | None = None) -> str:
        """
        Create a fingerprint of recent conversation context.
        Uses message roles and a hash of content (first 100 chars).
        """
        if not conversation_history:
            return "empty"
        
        # Take last N messages for context
        recent = conversation_history[-self.context_window:]
        
        # Build context signature: role + content hash
        parts = []
        for msg in recent:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100]
            else:
                role = getattr(msg, "role", "unknown")
                content = (getattr(msg, "content", "") or "")[:100]
            
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            parts.append(f"{role}:{content_hash}")
        
        return "|".join(parts)
    
    def _create_signature(self, tool_name: str, args: dict) -> str:
        """Create a signature for the tool call."""
        return f"{tool_name}:{sorted(args.items()) if args else ''}"
    
    def _get_nudge_message(self, tool_name: str, suspicion_level: int, 
                           context_changed: bool = False) -> str:
        """Get appropriate nudge message based on suspicion level."""
        
        if context_changed:
            # Context changed but tool repeated - probably legitimate
            base = f"⚠️  You called {tool_name} again. If you already have the answer, proceed to complete the task."
        else:
            # Same context - more likely to be a loop
            base = f"⚠️  You called {tool_name} with the same arguments in the same context."
        
        nudges = [
            f"{base} If you're stuck, try a different approach or call task_complete.",
            f"🚨 WARNING: Repeated {tool_name} calls detected. You may be in a loop. "
            f"Take corrective action NOW or call task_complete to finalize.",
        ]
        
        if suspicion_level <= 0:
            return nudges[0]
        else:
            return nudges[1]
    
    def check_progress(self, tool_name: str, args: dict,
                       conversation_history: list | None = None) -> dict:
        """
        Check if this tool call represents progress or a loop.
        
        Uses context fingerprinting for accurate detection, and adaptive
        recovery for nudge escalation.
        
        Returns: {"status": "progress"} or {"status": "loop"} or {"status": "nudge"}
        """
        # Create signatures
        call_sig = self._create_signature(tool_name, args)
        context_sig = self._compute_context_fingerprint(conversation_history)
        combined_sig = f"{call_sig}@{context_sig}"
        
        # Store history
        self.tool_history.append(combined_sig)
        self.context_history.append(context_sig)
        
        if len(self.tool_history) > self.window_size * 2:
            self.tool_history = self.tool_history[-self.window_size * 2:]
            self.context_history = self.context_history[-self.window_size * 2:]
        
        # Check for exact repeat in same context
        if len(self.tool_history) >= 2:
            prev_combined = self.tool_history[-2]
            prev_context = self.context_history[-2]
            
            if combined_sig == prev_combined:
                # Same tool + same args + same context = likely loop
                current_suspicion = self.suspicion_count.get(combined_sig, 0) + 1
                self.suspicion_count[combined_sig] = current_suspicion
                
                if current_suspicion >= self.max_nudges:
                    return {
                        "status": "loop",
                        "action": "fail_fast",
                        "reason": "max_nudges_exceeded_same_context",
                        "suspicion_count": current_suspicion
                    }
                else:
                    nudge = self._get_nudge_message(tool_name, current_suspicion - 1, 
                                                    context_changed=False)
                    self.total_nudges += 1
                    return {
                        "status": "nudge",
                        "action": "inject_warning",
                        "reason": "suspected_loop_same_context",
                        "nudge_message": nudge,
                        "suspicion_count": current_suspicion,
                        "nudges_remaining": self.max_nudges - current_suspicion
                    }
            
            # Check if tool repeats but context changed (legitimate re-query)
            prev_call_sig = prev_combined.split("@")[0]
            if call_sig == prev_call_sig and context_sig != prev_context:
                # Context changed - allow but track with lower suspicion
                current_suspicion = self.suspicion_count.get(combined_sig, 0) + 1
                self.suspicion_count[combined_sig] = current_suspicion
                
                # Only nudge if it happens multiple times even with context changes
                if current_suspicion >= self.max_nudges + 1:  # +1 for leniency
                    return {
                        "status": "nudge",
                        "action": "inject_warning",
                        "reason": "repeated_despite_context_changes",
                        "nudge_message": self._get_nudge_message(tool_name, 1, context_changed=True),
                        "suspicion_count": current_suspicion
                    }
                
                return {
                    "status": "progress",
                    "note": "same_tool_different_context_allowed",
                    "context_changed": True
                }
        
        # Check for cycle pattern (A->B->A->B) with same contexts
        if len(self.tool_history) >= 4:
            last4_tools = [h.split("@")[0] for h in self.tool_history[-4:]]
            last4_contexts = self.context_history[-4:]
            
            if (last4_tools[0] == last4_tools[2] and 
                last4_tools[1] == last4_tools[3] and
                last4_contexts[0] == last4_contexts[2] and
                last4_contexts[1] == last4_contexts[3]):
                
                cycle_sig = f"cycle:{last4_tools[0]}<->{last4_tools[1]}@{last4_contexts[0]}"
                current_suspicion = self.suspicion_count.get(cycle_sig, 0) + 1
                self.suspicion_count[cycle_sig] = current_suspicion
                
                if current_suspicion >= self.max_nudges:
                    return {
                        "status": "loop",
                        "action": "fail_fast",
                        "reason": "cycle_max_nudges_exceeded",
                        "suspicion_count": current_suspicion
                    }
                else:
                    nudge = f"🔄 Cycle detected: alternating between patterns. "
                    f"Please break this pattern or complete the task."
                    self.total_nudges += 1
                    return {
                        "status": "nudge",
                        "action": "inject_warning",
                        "reason": "suspected_cycle_same_context",
                        "nudge_message": nudge,
                        "suspicion_count": current_suspicion
                    }
        
        return {"status": "progress"}
    
    def get_stats(self) -> dict:
        """Get statistics about guard operation."""
        return {
            "total_nudges": self.total_nudges,
            "suspicion_counts": dict(self.suspicion_count),
            "history_length": len(self.tool_history)
        }
    
    def reset(self):
        """Reset the guard for a new trial."""
        self.tool_history = []
        self.context_history = []
        self.suspicion_count = {}
        self.total_nudges = 0
    async def __call__(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        next_dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolEnvelope]],
    ) -> ToolEnvelope:
        """Interceptor entry point."""
        history = self.message_provider() if self.message_provider else []
        res = self.check_progress(tool_name, arguments, history)
        
        if res["status"] == "loop":
            return ToolEnvelope(
                success=False,
                error=f"ERROR: Infinite loop detected! {res['reason']}",
                metadata={"loop_detected": True, "reason": res["reason"]}
            )
        
        if res["status"] == "nudge":
            # Inject warning via metadata/error so the harness can handle it
            # In the current setup, we might want to still call the tool but add a warning
            result = await next_dispatch(tool_name, arguments)
            result.error = f"{result.error}\n\n{res['nudge_message']}" if result.error else res['nudge_message']
            return result
            
        return await next_dispatch(tool_name, arguments)
