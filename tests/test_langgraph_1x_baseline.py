"""Baseline tests for LangGraph 1.x runtime behavior."""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.types import Command

from smallctl.graph.runtime import ChatGraphRuntime, LoopGraphRuntime
from smallctl.graph.runtime_base import (
    checkpoint_config,
    load_runtime_state,
    serialize_runtime_state,
)
from smallctl.graph.runtime_payloads import build_runtime_payload
from smallctl.graph.runtime_specialized import (
    IndexerGraphRuntime,
    PlanningGraphRuntime,
    ToolPlanRuntime,
)
from smallctl.graph.runtime_staged import StagedExecutionRuntime
from smallctl.graph.state import GraphRunState, PendingToolCall
from smallctl.graph.subgraphs import ChildSubgraphRunner
from smallctl.harness import Harness


RUNTIME_GRAPH_CLASSES: list[Any] = [
    LoopGraphRuntime,
    ChatGraphRuntime,
    PlanningGraphRuntime,
    IndexerGraphRuntime,
    ToolPlanRuntime,
    StagedExecutionRuntime,
]


def _minimal_harness(*, backend: str, tmp_path: Path | None = None) -> SimpleNamespace:
    path = None
    if backend == "file" and tmp_path is not None:
        path = str(tmp_path / "checkpoints.json")
    return SimpleNamespace(
        config=SimpleNamespace(),
        graph_checkpointer=backend,
        graph_checkpoint_path=path,
        _runlog=lambda *args, **kwargs: None,
    )


def test_imports_without_langchain_pending_deprecation_warnings() -> None:
    import_code = (
        "import smallctl.graph.checkpoint; "
        "import smallctl.graph.runtime_base; "
        "import smallctl.graph.runtime; "
        "import smallctl.graph.subgraphs; "
        "print('imports_ok')"
    )
    python_exe = sys.executable
    specific_warning = "langchain_core._api.beta_decorator.LangChainPendingDeprecationWarning"

    result = subprocess.run(
        [python_exe, "-W", f"error::{specific_warning}", "-c", import_code],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and (
        "Unknown warning category" in result.stderr
        or "Invalid -W option" in result.stderr
    ):
        result = subprocess.run(
            [python_exe, "-W", "error", "-c", import_code],
            capture_output=True,
            text=True,
        )

    assert result.returncode == 0, result.stderr
    assert "imports_ok" in result.stdout


@pytest.mark.parametrize("backend", ["memory", "file"])
@pytest.mark.parametrize("runtime_cls", RUNTIME_GRAPH_CLASSES)
def test_runtime_graph_class_compiles(
    tmp_path: Path, runtime_cls: Any, backend: str
) -> None:
    harness = _minimal_harness(backend=backend, tmp_path=tmp_path)
    runtime = runtime_cls.from_harness(harness)
    compiled = runtime._build_compiled_graph()
    assert compiled is not None
    assert hasattr(compiled, "get_state")


@pytest.mark.parametrize("backend", ["memory", "file"])
def test_child_subgraph_compiles(tmp_path: Path, backend: str) -> None:
    harness = _minimal_harness(backend=backend, tmp_path=tmp_path)
    runner = ChildSubgraphRunner()
    request = SimpleNamespace(
        brief="test",
        parent_conversation_id="c-1",
        child_depth=0,
    )
    compiled = runner._build_compiled_subgraph(
        parent=harness,
        request=request,
        harness_factory=None,
    )
    assert compiled is not None
    assert hasattr(compiled, "get_state")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "file"])
async def test_loop_graph_interrupt_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    harness = Harness(
        endpoint="http://example.test/v1",
        model="test-model",
        provider_profile="generic",
        phase="explore",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer=backend,
        graph_checkpoint_path=(str(tmp_path / "checkpoints.json") if backend == "file" else None),
    )
    harness.config.graph_idle_watchdog_sec = None

    runtime = LoopGraphRuntime.from_harness(harness)
    prepare_step_calls = 0

    async def patched_prepare_step(
        self: LoopGraphRuntime, payload: dict[str, Any]
    ) -> dict[str, Any]:
        nonlocal prepare_step_calls
        prepare_step_calls += 1
        graph_state = load_runtime_state(self, payload)
        if prepare_step_calls == 1:
            graph_state.pending_tool_calls = [
                PendingToolCall(
                    tool_call_id="tc-1",
                    tool_name="ask_human",
                    args={"question": "test?"},
                    source="internal",
                )
            ]
        else:
            graph_state.final_result = {
                "status": "completed",
                "assistant": "resumed and completed",
            }
        return serialize_runtime_state(graph_state)

    async def patched_dispatch_tools(
        self: LoopGraphRuntime, payload: dict[str, Any]
    ) -> dict[str, Any]:
        graph_state = load_runtime_state(self, payload)
        graph_state.pending_tool_calls = []
        graph_state.interrupt_payload = {
            "kind": "ask_human",
            "question": "test?",
        }
        return serialize_runtime_state(graph_state)

    async def patched_persist_tool_results(
        self: LoopGraphRuntime, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return payload

    async def patched_apply_tool_outcomes(
        self: LoopGraphRuntime, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return payload

    async def patched_resume_loop_run(
        graph_state: GraphRunState, deps: Any, *, human_input: str
    ) -> None:
        graph_state.interrupt_payload = None
        graph_state.loop_state.pending_interrupt = None

    runtime._prepare_step_node = types.MethodType(patched_prepare_step, runtime)
    runtime._dispatch_tools_node = types.MethodType(patched_dispatch_tools, runtime)
    runtime._persist_tool_results_node = types.MethodType(
        patched_persist_tool_results, runtime
    )
    runtime._apply_tool_outcomes_node = types.MethodType(
        patched_apply_tool_outcomes, runtime
    )
    monkeypatch.setattr(
        "smallctl.graph.runtime.resume_loop_run", patched_resume_loop_run
    )

    compiled = runtime._build_compiled_graph()
    config = checkpoint_config(harness, recursion_limit=512)
    payload = build_runtime_payload(
        harness, run_mode="loop", input_task="interrupt test"
    )

    async for _ in compiled.astream(payload, config):
        pass

    snapshot = compiled.get_state(config)
    assert snapshot is not None
    values = snapshot.values if hasattr(snapshot, "values") else snapshot
    assert values.get("interrupt_payload") is not None
    assert values["interrupt_payload"].get("kind") == "ask_human"

    resume_chunks = 0
    async for _ in compiled.astream(Command(resume="ok"), config):
        resume_chunks += 1

    final_snapshot = compiled.get_state(config)
    final_values = (
        final_snapshot.values
        if hasattr(final_snapshot, "values")
        else final_snapshot
    )
    assert final_values.get("final_result") is not None
    assert final_values["final_result"]["status"] == "completed"
    assert prepare_step_calls == 2
    assert resume_chunks >= 1
