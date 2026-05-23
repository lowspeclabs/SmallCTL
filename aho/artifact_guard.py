from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from src.smallctl.tools.dispatcher import ToolInterceptor
from src.smallctl.models.tool_result import ToolEnvelope

class ArtifactDedupeInterceptor(ToolInterceptor):
    """
    Prevents redundant tool execution that would result in duplicate artifacts.
    Tracks tools that return large content (persisted as artifacts) and warns
    the model if it requests the exact same data again.
    """
    def __init__(self) -> None:
        # Maps (tool_name, args_json) -> result_summary or True
        self._call_history: set[str] = set()
        # Track if any state-modifying tools were called
        self._last_state_change_step: int = -1

    async def __call__(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        next_dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolEnvelope]],
    ) -> ToolEnvelope:
        # Only deduplicate "read-only" or "discovery" tools that generate artifacts
        DEDUPE_TOOLS = {
            "long_context_lookup",
            "summarize_report",
            "artifact_read",
            "file_read",
            "dir_list",
            "grep",
            "find_by_name"
        }
        
        # Tools that definitely change the system state
        STATE_CHANGE_TOOLS = {
            "file_write",
            "shell_exec",
            "multi_replace_file_content",
            "replace_file_content",
            "write_to_file",
            "file_patch",
            "ast_patch",
            "file_delete"
        }

        if tool_name in STATE_CHANGE_TOOLS:
            self._call_history.clear()  # Reset history on state change
            return await next_dispatch(tool_name, arguments)

        if tool_name in DEDUPE_TOOLS:
            # Create a stable fingerprint of the call
            fingerprint = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
            
            if fingerprint in self._call_history:
                return ToolEnvelope(
                    success=True,
                    output=(
                        f"[REDUNDANCY GUARD]: You already called {tool_name} with these exact parameters. "
                        "The results are already in your history or pinned facts. "
                        "Do not repeat the call unless you believe the underlying data has changed "
                        "(e.g., after a file write or command execution)."
                    ),
                    metadata={"interceptor": "ArtifactDedupeInterceptor", "is_redundant": True}
                )
            
            self._call_history.add(fingerprint)

        return await next_dispatch(tool_name, arguments)
