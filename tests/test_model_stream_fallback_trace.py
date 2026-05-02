from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.model_stream_fallback_trace import (
    _build_text_write_fallback_trace,
    _record_text_write_fallback_state,
)


def test_text_write_fallback_trace_includes_key_sections() -> None:
    session = SimpleNamespace(write_target_path="notes.py")
    trace = _build_text_write_fallback_trace(
        session=session,
        current_section="imports",
        prompt="print('hi')",
        assistant_text="```python\nprint('hi')\n```",
        extracted_code="print('hi')",
        next_section_name="body",
        tool_names=["file_write"],
    )

    assert "Chat-mode fallback activated for a stalled write task." in trace
    assert "Target path" in trace
    assert "Observed tool calls" in trace


def test_text_write_fallback_state_records_clip_excerpts() -> None:
    harness = SimpleNamespace(state=SimpleNamespace(scratchpad={}))
    session = SimpleNamespace(write_session_id="ws-1", write_target_path="notes.py")

    _record_text_write_fallback_state(
        harness,
        status="attempting",
        reason="test",
        session=session,
        current_section="imports",
        remaining_sections=["body"],
        prompt="x" * 1000,
        tool_names=["file_write"],
        assistant_text="y" * 1000,
        extracted_code="z" * 1000,
        next_section_name="body",
    )

    record = harness.state.scratchpad["_last_text_write_fallback"]
    assert record["status"] == "attempting"
    assert record["write_session_id"] == "ws-1"
    assert record["prompt_excerpt"]
