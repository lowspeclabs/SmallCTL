from __future__ import annotations

from smallctl.graph.state import PendingToolCall
from smallctl.guards import apply_triple_answer_guard


def test_short_meta_commentary_is_stripped() -> None:
    assistant_text = "I'll call task_complete now."
    pending_calls = [
        PendingToolCall(
            tool_name="task_complete",
            args={"message": "done"},
            raw_arguments='{"message": "done"}',
        )
    ]

    assert apply_triple_answer_guard(assistant_text, pending_calls) == ""


def test_long_assistant_response_with_meta_commentary_is_preserved() -> None:
    assistant_text = (
        "I have reviewed the file tree and confirmed the issue is isolated to the build step. "
        "I'll call task_complete now, but the important part is that the failure only appears when "
        "the cache is stale and the dependency lockfile is missing."
    )
    pending_calls = [
        PendingToolCall(
            tool_name="task_complete",
            args={"message": "I will call task_complete now."},
            raw_arguments='{"message": "I will call task_complete now."}',
        )
    ]

    assert apply_triple_answer_guard(assistant_text, pending_calls) == assistant_text


def test_exact_match_with_task_complete_message_is_stripped() -> None:
    assistant_text = "Task complete: the build passes."
    pending_calls = [
        PendingToolCall(
            tool_name="task_complete",
            args={"message": "Task complete: the build passes."},
            raw_arguments='{"message": "Task complete: the build passes."}',
        )
    ]

    assert apply_triple_answer_guard(assistant_text, pending_calls) == ""


def test_substantial_fuzzy_match_with_extra_details_is_preserved() -> None:
    assistant_text = (
        "Task complete: the directory listing was created and saved to the report. "
        "I also confirmed the output path is /tmp/report.txt, and the parent directory still has "
        "enough disk space for the next step."
    )
    pending_calls = [
        PendingToolCall(
            tool_name="task_complete",
            args={"message": "Task complete: the directory listing was created and saved to the report."},
            raw_arguments='{"message": "Task complete: the directory listing was created and saved to the report."}',
        )
    ]

    assert apply_triple_answer_guard(assistant_text, pending_calls) == assistant_text
