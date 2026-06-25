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

    @classmethod
    def make_error(
        cls,
        tool_name: str,
        error: str,
        *,
        status: str | None = None,
        reason: str | None = None,
        **metadata: Any,
    ) -> "ToolEnvelope":
        """Build a failure envelope with consistent metadata."""
        meta: dict[str, Any] = {"tool_name": tool_name}
        if reason:
            meta["reason"] = reason
        meta.update(metadata)
        return cls(success=False, status=status, error=error, metadata=meta)

    @classmethod
    def make_blocked(
        cls,
        tool_name: str,
        reason: str,
        **metadata: Any,
    ) -> "ToolEnvelope":
        """Build a policy/phase-blocked failure envelope."""
        return cls.make_error(tool_name, reason, reason="blocked", **metadata)

    @classmethod
    def make_success(
        cls,
        output: Any,
        *,
        status: str | None = None,
        **metadata: Any,
    ) -> "ToolEnvelope":
        """Build a success envelope with optional status and metadata."""
        return cls(success=True, status=status, output=output, metadata=metadata)

