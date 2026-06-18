from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.state import GraphRunState, ToolExecutionRecord, serialize_graph_state
from smallctl.graph.tool_execution_support import _store_tool_execution_record, _tool_envelope_from_dict
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


def test_store_tool_execution_record_compacts_large_output() -> None:
    state = LoopState(thread_id="thread-1")
    harness = SimpleNamespace(state=state)
    pending = SimpleNamespace(
        tool_name="grep",
        tool_call_id="call-1",
        args={"pattern": "needle"},
        source="model",
    )
    result = ToolEnvelope(success=True, status="ok", output="x" * 70_000, metadata={})

    _store_tool_execution_record(
        harness,
        operation_id="op-1",
        thread_id="thread-1",
        step_count=1,
        pending=pending,
        result=result,
    )

    stored = state.tool_execution_records["op-1"]["result"]
    assert stored["status"] == "ok"
    assert isinstance(stored["output"], str)
    assert len(stored["output"]) < 10_000
    assert "output compacted" in stored["output"]
    assert stored["metadata"]["durable_output_compacted"]["truncated"] is True


def test_graph_state_serialization_compacts_large_shell_output() -> None:
    large_stdout = "line\n" * 20_000
    graph_state = GraphRunState(
        loop_state=LoopState(thread_id="thread-1"),
        thread_id="thread-1",
        run_mode="loop",
        last_tool_results=[
            ToolExecutionRecord(
                operation_id="op-1",
                tool_name="shell_exec",
                args={"command": "printf many-lines"},
                tool_call_id="call-1",
                result=ToolEnvelope(
                    success=True,
                    status="completed",
                    output={"exit_code": 0, "stdout": large_stdout, "stderr": "", "command": "printf many-lines"},
                    metadata={"artifact_id": "artifact-1"},
                ),
            )
        ],
    )

    payload = serialize_graph_state(graph_state)
    result = payload["last_tool_results"][0]["result"]

    assert result["status"] == "completed"
    assert isinstance(result["output"], str)
    assert "artifact artifact-1" in result["output"]
    assert "exit_code: 0" in result["output"]
    assert len(result["output"]) < 10_000
    assert large_stdout not in result["output"]


def test_loop_state_to_dict_compacts_legacy_large_record_output() -> None:
    state = LoopState(thread_id="thread-1")
    state.tool_execution_records["op-1"] = {
        "operation_id": "op-1",
        "tool_name": "file_read",
        "artifact_id": "artifact-1",
        "result": {
            "success": True,
            "status": "ok",
            "output": "x" * 70_000,
            "error": None,
            "metadata": {"artifact_id": "artifact-1"},
        },
    }

    payload = state.to_dict()
    result = payload["tool_execution_records"]["op-1"]["result"]

    assert result["status"] == "ok"
    assert isinstance(result["output"], str)
    assert "artifact artifact-1" in result["output"]
    assert len(result["output"]) < 10_000
    assert result["metadata"]["durable_output_compacted"]["original_chars"] == 70_000


def test_tool_envelope_rehydration_preserves_status() -> None:
    envelope = _tool_envelope_from_dict(
        {"success": True, "status": "completed", "output": "ok", "error": None, "metadata": {}}
    )

    assert envelope.status == "completed"
