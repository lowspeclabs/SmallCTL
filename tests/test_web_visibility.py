from __future__ import annotations

from types import SimpleNamespace

from smallctl.harness.tool_dispatch import chat_mode_tools
from smallctl.harness.tool_visibility import resolve_turn_tool_exposure
from smallctl.state import LoopState
from smallctl.tools.profiles import classify_tool_profiles
from smallctl.tools.register import build_registry


def _tool_names(tools: list[dict[str, object]]) -> list[str]:
    return [
        str(entry["function"]["name"])
        for entry in tools
        if isinstance(entry, dict) and isinstance(entry.get("function"), dict)
    ]


def test_web_search_visible_in_chat_mode_for_current_info_tasks(tmp_path) -> None:
    task = "What is the latest pricing for ExampleCorp and what docs mention it?"
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = sorted(classify_tool_profiles(task))
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        _current_user_task=lambda: task,
        _runlog=lambda *args, **kwargs: None,
        registry=build_registry(SimpleNamespace(state=state, log=SimpleNamespace(info=lambda *args, **kwargs: None)), registry_profiles=None),
    )

    tools = chat_mode_tools(harness)
    names = _tool_names(tools)
    exposure = resolve_turn_tool_exposure(harness, "chat")

    assert "web_search" in names
    assert "web_fetch" in names
    assert "http_get" not in names
    assert "web_search" in exposure["names"]
    assert "web_fetch" in exposure["names"]
