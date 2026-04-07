from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..retrieval_safety import build_retrieval_safe_text


@dataclass
class ConversationMessage:
    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_safe_text: str | None = None

    def __post_init__(self) -> None:
        self.role = str(self.role)
        if self.content is not None:
            self.content = str(self.content)
        if self.name is not None:
            self.name = str(self.name)
        if self.tool_call_id is not None:
            self.tool_call_id = str(self.tool_call_id)
        if not isinstance(self.tool_calls, list):
            self.tool_calls = []
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        if self.retrieval_safe_text is not None:
            normalized = str(self.retrieval_safe_text).strip()
            self.retrieval_safe_text = normalized or None
        else:
            derived = build_retrieval_safe_text(
                role=self.role,
                content=self.content,
                name=self.name,
                metadata=self.metadata,
            )
            self.retrieval_safe_text = derived or None

    def to_dict(
        self,
        *,
        include_metadata: bool = True,
        include_retrieval_safe_text: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if include_metadata and self.metadata:
            payload["metadata"] = self.metadata
        if include_retrieval_safe_text and self.retrieval_safe_text is not None:
            payload["retrieval_safe_text"] = self.retrieval_safe_text
        return payload
