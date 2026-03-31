from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class GraphEventType(str, Enum):
    NODE_START = "node_start"
    NODE_END = "node_end"
    STATE_UPDATE = "state_update"


@dataclass
class GraphEvent:
    event_type: GraphEventType
    node: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
