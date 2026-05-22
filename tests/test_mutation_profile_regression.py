from __future__ import annotations

from types import SimpleNamespace

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


def test_fix_traceback_in_py_file_adds_mutate_profile() -> None:
    profiles = classify_tool_profiles("fix traceback in temp/pong.py")
    assert "mutate" in profiles


def test_debug_script_adds_mutate_profile() -> None:
    profiles = classify_tool_profiles("debug the login script")
    assert "mutate" in profiles


def test_repair_code_without_file_does_not_add_mutate_profile() -> None:
    profiles = classify_tool_profiles("repair the server")
    assert "mutate" not in profiles


def test_active_intent_requested_file_patch_adds_mutate_profile(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_intent = "requested_file_patch"
    state.active_tool_profiles = ["core"]
    harness = SimpleNamespace(
        state=state,
        registry=build_registry(
            SimpleNamespace(state=state, log=SimpleNamespace(info=lambda *a, **k: None)),
            registry_profiles=None,
        ),
        log=SimpleNamespace(info=lambda *a, **k: None),
    )

    exposure = resolve_turn_tool_exposure(harness, "loop")
    names = exposure["names"]

    assert "mutate" in state.active_tool_profiles
    assert "file_patch" in names
