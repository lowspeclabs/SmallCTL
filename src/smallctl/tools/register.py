from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from . import artifact, control, data, http, indexer, indexer_query, memory, network, planning, search, shell
from .base import ToolMode, ToolRisk, ToolSpec, ToolTier, build_tool_schema
from .profiles import (
    CORE_PROFILE,
    DATA_PROFILE,
    INDEXER_PROFILE,
    MUTATE_PROFILE,
    NETWORK_PROFILE,
    SUPPORT_PROFILE,
)
from .registry import ToolRegistry
from .register_control_planning import register_control_planning_tools
from .register_filesystem import register_filesystem_tools
from .register_operational import register_operational_tools
from .register_content import register_content_tools

Handler = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolRegistration:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Handler
    category: str
    risk: ToolRisk
    allowed_phases: set[str] | None = None
    allowed_modes: set[ToolMode] | None = None
    profiles: set[str] | None = None
    tier: ToolTier = "tier1"


def build_registry(
    state_provider: Any,
    *,
    registry_profiles: set[str] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    state_provider.log.info("build_registry: starting registration")

    def _inject_cwd(func: Callable[..., Awaitable[dict[str, Any]]]) -> Handler:
        return lambda **kwargs: func(cwd=state_provider.state.cwd, **kwargs)

    def _inject_state_and_cwd(func: Callable[..., Awaitable[dict[str, Any]]]) -> Handler:
        return lambda **kwargs: func(cwd=state_provider.state.cwd, state=state_provider.state, **kwargs)

    def _inject_state(func: Callable[..., Awaitable[dict[str, Any]]]) -> Handler:
        return lambda **kwargs: func(state=state_provider.state, **kwargs)

    def _inject_state_and_harness(func: Callable[..., Awaitable[dict[str, Any]]]) -> Handler:
        return lambda **kwargs: func(state=state_provider.state, harness=state_provider, **kwargs)

    def _inject_harness(func: Callable[..., Awaitable[dict[str, Any]]]) -> Handler:
        return lambda **kwargs: func(harness=state_provider, **kwargs)

    def _register(tools: list[ToolRegistration]) -> None:
        for tool in tools:
            spec = ToolSpec(
                name=tool.name,
                description=tool.description,
                schema=tool.schema,
                handler=tool.handler,
                tier=tool.tier,
                category=tool.category,
                risk=tool.risk,
                allowed_phases=tool.allowed_phases,
                allowed_modes=tool.allowed_modes,
                profiles=tool.profiles,
            )
            if registry_profiles and tool.profiles and not (tool.profiles & registry_profiles):
                continue
            registry.register(spec)

    def _make_registration(**kwargs: Any) -> ToolRegistration:
        return ToolRegistration(**kwargs)

    register_filesystem_tools(
        register=_register,
        make_registration=_make_registration,
        inject_cwd=_inject_cwd,
        inject_state_and_cwd=_inject_state_and_cwd,
        core_profile=CORE_PROFILE,
        mutate_profile=MUTATE_PROFILE,
        support_profile=SUPPORT_PROFILE,
    )

    register_control_planning_tools(
        register=_register,
        make_registration=_make_registration,
        inject_state=_inject_state,
        inject_state_and_harness=_inject_state_and_harness,
        core_profile=CORE_PROFILE,
    )

    register_operational_tools(
        register=_register,
        make_registration=_make_registration,
        inject_state=_inject_state,
        inject_state_and_harness=_inject_state_and_harness,
        core_profile=CORE_PROFILE,
        support_profile=SUPPORT_PROFILE,
        network_profile=NETWORK_PROFILE,
    )

    register_content_tools(
        register=_register,
        make_registration=_make_registration,
        inject_state=_inject_state,
        inject_harness=_inject_harness,
        core_profile=CORE_PROFILE,
        data_profile=DATA_PROFILE,
        indexer_profile=INDEXER_PROFILE,
    )

    _register([
        ToolRegistration(
            name="memory_update",
            description="Update the pinned Working Memory (plan, decisions, known_facts, next_actions). Use this to persist critical facts (like numerical values or key findings) found in large artifacts so they aren't lost when history is truncated.",
            schema=build_tool_schema(
                required=["section", "content"],
                properties={
                    "section": {
                        "type": "string",
                        "enum": ["plan", "decisions", "open_questions", "known_facts", "failures", "next_actions"],
                    },
                    "content": {"type": "string"},
                    "action": {"type": "string", "enum": ["add", "remove"]},
                },
            ),
            handler=_inject_state(memory.memory_update),
            category="memory",
            risk="low",
            allowed_modes={"loop", "planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="log_note",
            description="Append a concise note to the session notepad. Notes persist across task-boundary resets within this session.",
            schema=build_tool_schema(
                required=["content"],
                properties={
                    "content": {"type": "string"},
                    "tag": {"type": "string"},
                },
            ),
            handler=_inject_state(memory.log_note),
            category="memory",
            risk="low",
            allowed_modes={"loop", "planning"},
            profiles={CORE_PROFILE},
        ),
    ])

    return registry


def register_mock_tool(
    registry: ToolRegistry,
    name: str,
    description: str,
    handler: Handler,
    required: list[str] | None = None,
    properties: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Helper for registering mock tools with common patterns.

    This function consolidates the schema building patterns used in AHO,
    reducing code duplication and ensuring consistency with core tool registration.
    """
    registry.register(
        ToolSpec(
            name=name,
            description=description,
            schema=build_tool_schema(required=required or [], properties=properties or {}),
            handler=handler,
            tier="tier1",
            category="mock",
            risk="low",
            allowed_phases={"explore", "plan", "execute", "verify"},
            allowed_modes={"loop"},
            profiles={CORE_PROFILE},
            **kwargs,
        )
    )
