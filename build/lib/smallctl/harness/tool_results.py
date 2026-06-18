from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..models.conversation import ConversationMessage
from ..models.tool_result import ToolEnvelope
from .tool_dispatch import maybe_reuse_file_read as _maybe_reuse_file_read
from .tool_message_compaction import compact_oversized_tool_messages as _compact_oversized_tool_messages
from .tool_result_flow import record_result as _record_result
from .tool_result_verification import _store_verifier_verdict

if TYPE_CHECKING:
    from ..harness import Harness


class ToolResultService:
    def __init__(self, harness: Harness):
        self.harness = harness

    async def record_result(
        self,
        *,
        tool_name: str,
        tool_call_id: str | None,
        result: ToolEnvelope,
        arguments: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> ConversationMessage:
        return await _record_result(
            self,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            result=result,
            arguments=arguments,
            operation_id=operation_id,
        )

    def maybe_reuse_file_read(self, *, tool_name: str, args: dict[str, Any]) -> ToolEnvelope | None:
        return _maybe_reuse_file_read(self.harness, tool_name=tool_name, args=args)

    def compact_oversized_tool_messages(self, *, soft_limit: int) -> bool:
        return _compact_oversized_tool_messages(self.harness, soft_limit=soft_limit)
