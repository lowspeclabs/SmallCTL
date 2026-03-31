from __future__ import annotations

from typing import Any

from .base import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._specs[spec.name] = spec

    def register_from_handler(self, handler: Any) -> ToolSpec:
        spec = getattr(handler, "__tool_spec__", None)
        if spec is None:
            raise ValueError(f"Handler has no tool spec: {handler}")
        self.register(spec)
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return sorted(self._specs.keys())

    def all_specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in self.names()]

    def export_openai_tools(
        self,
        phase: str | None = None,
        *,
        mode: str | None = None,
        profiles: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        specs = self.all_specs()
        if phase:
            specs = [spec for spec in specs if spec.phase_allowed(phase)]
        if mode:
            specs = [spec for spec in specs if spec.mode_allowed(mode)]
        if profiles:
            specs = [spec for spec in specs if spec.profile_allowed(profiles)]
        return [spec.openai_schema() for spec in specs]
