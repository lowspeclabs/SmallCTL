from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import time
from pathlib import Path

from smallctl import memory_cli
from smallctl.memory_cli import memory_cli as run_memory_cli
from smallctl.graph import model_stream_loop, model_stream_loop_rendering
from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.lifecycle_prompt import prepare_prompt, select_loop_tools
from smallctl.graph.model_stream_loop_rendering import (
    StreamTagState,
    handle_model_stream_chunk,
)
from smallctl.graph.state import GraphRunState
from smallctl.harness.prompt_builder import prompt_budget_overflow_error
from smallctl.memory_store import ExperienceStore
from smallctl.state import ConversationMessage, LoopState
from smallctl.tools.control_task_complete_gates import _is_remote_service_install_task


def test_stream_timing_uses_only_monotonic_clock(monkeypatch) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: 1_000.0)
    monkeypatch.setattr(time, "perf_counter", lambda: 900_000_000.0)

    async def _run() -> float | None:
        stream_state, first_token_time = await handle_model_stream_chunk(
            harness=None,
            deps=None,
            event={"type": "chunk", "data": {"choices": [{"delta": {}}]}},
            start_tag="<think>",
            end_tag="</think>",
            echo_to_stdout=False,
            chunks=[],
            stream_state=StreamTagState(),
            first_token_time=None,
        )
        return first_token_time

    first_token_time = asyncio.run(_run())

    assert first_token_time == 1_000.0
    for module in (model_stream_loop, model_stream_loop_rendering):
        assert "perf_counter" not in inspect.getsource(module)


def test_remote_install_gate_requires_valid_remote_anchors() -> None:
    local = LoopState()
    local.run_brief.original_task = (
        "Add a host mapping for myapp.test to /etc/hosts and run the app test suite."
    )
    assert _is_remote_service_install_task(local) is False

    version_like = LoopState()
    version_like.run_brief.original_task = (
        "Run the app install checks for release 3.10.2 locally and start nothing."
    )
    assert _is_remote_service_install_task(version_like) is False

    explicit_ssh = LoopState()
    explicit_ssh.run_brief.original_task = (
        "Use ssh to deploy the netbox app on 192.168.1.63 as a docker service."
    )
    assert _is_remote_service_install_task(explicit_ssh) is True

    user_at_host = LoopState()
    user_at_host.run_brief.original_task = (
        "Install the application container on root@appserver and start the service."
    )
    assert _is_remote_service_install_task(user_at_host) is True

    loopback_is_not_remote = LoopState()
    loopback_is_not_remote.run_brief.original_task = (
        "Install the app on 127.0.0.1 and run the container locally."
    )
    assert _is_remote_service_install_task(loopback_is_not_remote) is False


def _seed_warm_record(memory_id: str = "mem-1") -> None:
    memory_dir = Path(".smallctl") / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "warm-experiences.jsonl").write_text(
        json.dumps(
            {
                "memory_id": memory_id,
                "tier": "warm",
                "intent": "deployed_service",
                "tool_name": "ssh_exec",
                "outcome": "success",
                "notes": "deployment note",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _main_parser_promote_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    memory_cli.build_memory_parser(subparsers)
    return parser.parse_args(["memory", *argv])


def test_memory_promote_partial_failures_and_shared_parser(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_warm_record()

    original_upsert = ExperienceStore.upsert
    monkeypatch.setattr(ExperienceStore, "upsert", lambda self, memory: None)
    exit_code = run_memory_cli(["promote", "mem-1"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["recoverable_in_tier"] == "warm"
    assert "mem-1" in (tmp_path / ".smallctl" / "memory" / "warm-experiences.jsonl").read_text()
    monkeypatch.setattr(ExperienceStore, "upsert", original_upsert)

    original_delete = ExperienceStore.delete
    monkeypatch.setattr(ExperienceStore, "delete", lambda self, memory_id: False)
    exit_code = run_memory_cli(["promote", "--memory-id", "mem-1"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "partial"
    assert payload["recoverable_in_tier"] == "warm"
    assert "mem-1" in (tmp_path / ".smallctl" / "memory" / "warm-experiences.jsonl").read_text()
    assert "mem-1" in (tmp_path / ".smallctl" / "memory" / "cold-experiences.jsonl").read_text()
    monkeypatch.setattr(ExperienceStore, "delete", original_delete)

    cold_path = tmp_path / ".smallctl" / "memory" / "cold-experiences.jsonl"
    for argv in (["promote", "mem-1"], ["promote", "--memory-id", "mem-1"]):
        results = []
        for entry_point in ("memory_cli", "main_parser"):
            _seed_warm_record()
            if cold_path.exists():
                cold_path.unlink()
            if entry_point == "memory_cli":
                exit_code = run_memory_cli(argv)
            else:
                exit_code = memory_cli.handle_memory_command(_main_parser_promote_args(argv))
            results.append((exit_code, json.loads(capsys.readouterr().out)))
        assert results[0] == results[1]
        assert results[0][1]["status"] == "promoted"


def test_current_serialization_writes_single_transcript_copy() -> None:
    state = LoopState(cwd="/tmp")
    state.append_message(ConversationMessage(role="user", content="hello"))
    state.append_message(ConversationMessage(role="assistant", content="hi"))

    payload = state.to_dict()

    assert "conversation_history" not in payload
    assert [m["content"] for m in payload["transcript_messages"]] == ["hello", "hi"]

    legacy = LoopState.from_dict(
        {
            "cwd": "/tmp",
            "conversation_history": [
                {"role": "user", "content": "legacy question"},
                {"role": "assistant", "content": "legacy answer"},
            ],
        }
    )
    assert [m.content for m in legacy.transcript_messages] == ["legacy question", "legacy answer"]


class _BudgetHarness:
    def __init__(self, state: LoopState, *, limit: int, per_tool_tokens: int, always_overflow: bool) -> None:
        import logging

        self.state = state
        self.strategy_prompt = ""
        self._indexer = False
        self.log = logging.getLogger("smallctl.test.phase3")
        self.runlog_events: list[tuple[tuple, dict]] = []
        self._limit = limit
        self._per_tool_tokens = per_tool_tokens
        self._always_overflow = always_overflow

    def _runlog(self, *args, **kwargs) -> None:
        self.runlog_events.append((args, kwargs))

    async def _emit(self, event_handler, event) -> None:
        return None

    def _failure(self, message: str, error_type: str = "", details: dict | None = None) -> dict:
        return {"status": "failed", "error": message, "error_type": error_type}

    async def _build_prompt_messages(self, system_prompt: str, event_handler=None) -> list[dict]:
        tool_count = system_prompt.count("tool:")
        estimated = 10 + tool_count * self._per_tool_tokens
        if self._always_overflow or estimated > self._limit:
            raise prompt_budget_overflow_error(
                estimated_tokens=estimated,
                limit=self._limit,
                section_tokens={"system_prompt": estimated, "transcript": 5},
            )
        return [{"role": "system", "content": system_prompt}]


def _tool_schema(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": {}}},
    }


def test_prompt_budget_overflow_slimming_retry_and_diagnostics(monkeypatch, tmp_path) -> None:
    exposure = {
        "names": ["file_read", "task_complete", "web_search"],
        "schemas": [
            _tool_schema("file_read", "read a file " * 40),
            _tool_schema("task_complete", "complete the task " * 40),
            _tool_schema("web_search", "search the web " * 40),
        ],
    }
    monkeypatch.setattr(
        "smallctl.graph.lifecycle_prompt.resolve_turn_tool_exposure",
        lambda harness, mode: exposure,
    )
    monkeypatch.setattr(
        "smallctl.graph.lifecycle_prompt.build_system_prompt",
        lambda state, phase, available_tool_names=None, **kwargs: "sys " + " ".join(
            f"tool:{name}" for name in (available_tool_names or [])
        ),
    )

    state = LoopState(cwd=str(tmp_path))
    harness = _BudgetHarness(state, limit=30, per_tool_tokens=10, always_overflow=False)
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    deps = GraphRuntimeDeps(harness=harness)

    messages = asyncio.run(prepare_prompt(graph_state, deps))

    assert messages is not None
    assert state.scratchpad["_prompt_budget_tool_slimming_active"] is True
    slimmed = state.scratchpad["_prompt_budget_slimmed_tool_schemas"]
    assert [schema["function"]["name"] for schema in slimmed] == ["task_complete"]
    assert all(len(schema["function"]["description"]) <= 80 for schema in slimmed)
    assert any(
        args and args[0] == "prompt_budget_tool_slimming_retry" for args, _ in harness.runlog_events
    )
    selected = select_loop_tools(graph_state, deps)
    assert [schema["function"]["name"] for schema in selected] == ["task_complete"]

    full_state = LoopState(cwd=str(tmp_path))
    full_harness = _BudgetHarness(full_state, limit=30, per_tool_tokens=10, always_overflow=True)
    full_graph_state = GraphRunState(loop_state=full_state, thread_id="t2", run_mode="loop")
    full_deps = GraphRuntimeDeps(harness=full_harness)

    result = asyncio.run(prepare_prompt(full_graph_state, full_deps))

    assert result is None
    assert full_graph_state.final_result["error_type"] == "prompt_budget"
    error_text = full_graph_state.final_result["error"]
    assert "PROMPT BUDGET OVERFLOW" in error_text
    assert "Top contributors" in error_text
    assert "Remediation" in error_text
    assert "30" in error_text
