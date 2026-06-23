from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from smallctl.state import LoopState
from smallctl.tools.registry import ToolRegistry
from smallctl.tools.register import build_registry


class _FakeStateProvider:
    def __init__(self) -> None:
        self.state = LoopState(cwd=".")
        self.log = logging.getLogger("test")


def _make_registry() -> ToolRegistry:
    return build_registry(_FakeStateProvider())


def _has_tool(registry: ToolRegistry, name: str) -> bool:
    return registry.get(name) is not None


def test_interactive_run_is_registered_with_network_profile() -> None:
    registry = _make_registry()
    spec = registry.get("interactive_run")
    assert spec is not None, "interactive_run should be registered"
    assert "interactive" in spec.description.lower()
    assert spec.risk == "high"
    assert "network" in (spec.profiles or set())


def test_interactive_run_is_exported_for_loop_and_chat_and_planning() -> None:
    registry = _make_registry()
    for mode in ("loop", "chat", "planning"):
        schemas = registry.export_openai_tools(mode=mode)
        names = {s["function"]["name"] for s in schemas}
        assert "interactive_run" in names, f"interactive_run should be visible in {mode} mode"


def test_interactive_run_schema_requires_command() -> None:
    registry = _make_registry()
    spec = registry.get("interactive_run")
    assert spec is not None
    schema = spec.schema
    assert schema.get("required") == ["command"]
    props = schema.get("properties", {})
    assert "command" in props
    assert "answers" in props
    assert "target" in props
    assert "host" in props


@pytest.mark.parametrize(
    "model_name,is_small",
    [
        ("qwen2.5-7b-instruct", True),
        ("qwen2.5-14b-instruct", False),
        ("qwen2.5-32b-instruct", False),
        ("gemma-4-4b-it", True),
        ("gemma-4-12b-it", False),
        ("llama-3.1-8b-instruct", False),
    ],
)
def test_interactive_run_visible_for_small_and_mid_size_models(model_name: str, is_small: bool) -> None:
    """interactive_run should be visible for small (<=7B) and mid-size models, not restricted to large models."""
    from smallctl.guards import is_seven_b_or_under_model_name

    assert is_seven_b_or_under_model_name(model_name) == is_small
    # The tool itself is model-agnostic; visibility is governed by profiles/modes.
    registry = _make_registry()
    schemas = registry.export_openai_tools(mode="loop")
    names = {s["function"]["name"] for s in schemas}
    assert "interactive_run" in names
