from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

ToolHandler = Callable[..., dict[str, Any] | Awaitable[dict[str, Any]]]
ToolTier = Literal["tier1", "tier2"]
ToolRisk = Literal["low", "medium", "high"]
ToolMode = Literal["chat", "loop", "indexer", "planning"]
ToolProfile = Literal["core", "data", "network", "support", "mutate", "indexer"]


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    handler: ToolHandler
    tier: ToolTier = "tier1"
    category: str = "general"
    risk: ToolRisk = "medium"
    allowed_phases: set[str] | None = None
    allowed_modes: set[ToolMode] | None = None
    profiles: set[ToolProfile] | None = None

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }

    def phase_allowed(self, phase: str) -> bool:
        if not self.allowed_phases:
            return True
        return phase in self.allowed_phases

    def mode_allowed(self, mode: str) -> bool:
        if not self.allowed_modes:
            return True
        return mode in self.allowed_modes

    def profile_allowed(self, profiles: set[str] | None) -> bool:
        if not profiles:
            return True
        if not self.profiles:
            # If tool has no profiles assigned, it's considered universal
            return True
        return bool(self.profiles & profiles)

    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        result = self.handler(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def tool(
    *,
    name: str,
    description: str,
    schema: dict[str, Any],
    tier: ToolTier = "tier1",
    category: str = "general",
    risk: ToolRisk = "medium",
    allowed_phases: set[str] | None = None,
    allowed_modes: set[ToolMode] | None = None,
    profiles: set[ToolProfile] | None = None,
) -> Callable[[ToolHandler], ToolHandler]:
    def decorator(func: ToolHandler) -> ToolHandler:
        setattr(
            func,
            "__tool_spec__",
            ToolSpec(
                name=name,
                description=description,
                schema=schema,
                handler=func,
                tier=tier,
                category=category,
                risk=risk,
                allowed_phases=allowed_phases,
                allowed_modes=allowed_modes,
                profiles=profiles,
            ),
        )
        return func

    return decorator

def build_tool_schema(
    *,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }
