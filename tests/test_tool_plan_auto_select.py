from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.graph.runtime_auto import AutoGraphRuntime
from smallctl.harness.task_classifier import (
    classify_runtime_intent,
    looks_like_tool_plan_candidate,
)
from smallctl.state import LoopState


def test_looks_like_tool_plan_candidate_accepts_evidence_markers() -> None:
    assert looks_like_tool_plan_candidate("read through the repo and find where dispatch happens")
    assert looks_like_tool_plan_candidate("look through the codebase")
    assert looks_like_tool_plan_candidate("summarize current implementation")
    assert looks_like_tool_plan_candidate("identify files to patch")
    assert looks_like_tool_plan_candidate("trace where X is called")
    assert looks_like_tool_plan_candidate("investigate the failure")
    assert looks_like_tool_plan_candidate("multi-file debug session")
    assert looks_like_tool_plan_candidate("web research on python asyncio")


def test_looks_like_tool_plan_candidate_rejects_shell_and_action() -> None:
    assert not looks_like_tool_plan_candidate("run pytest and fix src/app.py")
    assert not looks_like_tool_plan_candidate("execute shell command to restart server")
    assert not looks_like_tool_plan_candidate("install nginx on 192.168.1.1")


def test_looks_like_tool_plan_candidate_rejects_write_requests() -> None:
    assert not looks_like_tool_plan_candidate("write a new file src/app.py")
    assert not looks_like_tool_plan_candidate("patch the existing code")
    assert not looks_like_tool_plan_candidate("create a new module")


def test_looks_like_tool_plan_candidate_rejects_empty_and_plan_only() -> None:
    assert not looks_like_tool_plan_candidate("")
    assert not looks_like_tool_plan_candidate("   ")
    assert not looks_like_tool_plan_candidate("write me a plan for how to refactor")


def test_looks_like_tool_plan_candidate_allows_analysis_with_content_lookup() -> None:
    assert looks_like_tool_plan_candidate("inspect the codebase")
    assert looks_like_tool_plan_candidate("trace the error in src/app.py")


def test_classify_runtime_intent_for_tool_plan_like_tasks() -> None:
    intent = classify_runtime_intent("inspect the codebase", recent_messages=[], pending_interrupt=None)
    assert intent.task_mode in {"analysis", "debug_inspect", "content_lookup", "readonly_lookup"}


def test_auto_runtime_auto_selects_tool_plan_for_web_research_when_enabled(monkeypatch, tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    runs: list[str] = []

    class _StubToolPlanRuntime:
        async def run(self, task: str) -> dict[str, object]:
            runs.append(task)
            return {"status": "tool_plan"}

    monkeypatch.setattr(
        "smallctl.graph.runtime_specialized.ToolPlanRuntime.from_harness",
        staticmethod(lambda harness, event_handler=None: _StubToolPlanRuntime()),
    )
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(
            run_mode="auto",
            tool_plan_runtime_enabled=True,
            tool_plan_auto_select=True,
        ),
        has_pending_interrupt=lambda: False,
        decide_run_mode=lambda task: asyncio.Future(),
        _runlog=lambda *args, **kwargs: None,
    )
    harness.decide_run_mode = lambda task: (_ for _ in ()).throw(AssertionError("should not call decide_run_mode"))

    result = asyncio.run(AutoGraphRuntime.from_harness(harness).run("web research on asyncio best practices"))

    assert result == {"status": "tool_plan"}
    assert runs == ["web research on asyncio best practices"]


def test_auto_runtime_does_not_auto_select_when_disabled(monkeypatch, tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    loop_runs: list[str] = []

    class _StubLoopRuntime:
        async def run(self, task: str) -> dict[str, object]:
            loop_runs.append(task)
            return {"status": "loop"}

    monkeypatch.setattr(
        "smallctl.graph.runtime.LoopGraphRuntime.from_harness",
        staticmethod(lambda harness, event_handler=None: _StubLoopRuntime()),
    )
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(
            run_mode="auto",
            tool_plan_runtime_enabled=False,
            tool_plan_auto_select=False,
        ),
        has_pending_interrupt=lambda: False,
        decide_run_mode=lambda task: asyncio.Future(),
        _runlog=lambda *args, **kwargs: None,
    )

    async def _decide(task: str) -> str:
        return "loop"

    harness.decide_run_mode = _decide

    result = asyncio.run(AutoGraphRuntime.from_harness(harness).run("read through the repo and find dispatch"))

    assert result == {"status": "loop"}
    assert loop_runs == ["read through the repo and find dispatch"]
