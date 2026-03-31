from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolEnvelope:
    success: bool
    status: str | None = None
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }

