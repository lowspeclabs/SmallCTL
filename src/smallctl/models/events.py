from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class UIEventType(str, Enum):
    USER = "user"
    THINKING = "thinking"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    SYSTEM = "system"
    METRICS = "metrics"
    SHELL_STREAM = "shell_stream"
    ALERT = "alert"


@dataclass
class UIEvent:
    event_type: UIEventType
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "content": self.content,
            "data": self.data,
            "timestamp": self.timestamp,
        }
