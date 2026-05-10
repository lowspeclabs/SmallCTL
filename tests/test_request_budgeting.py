from __future__ import annotations

from smallctl.client.request_budget import RequestEstimator, build_request_budget
from smallctl.client.tool_budgeting import fit_tools_to_context_budget


def _tool(name: str, *, size: int = 32) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name + " " + ("x" * size),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "description": "y" * size},
                },
            },
        },
    }


def _payload(tools: list[dict[str, object]], *, message_size: int = 32) -> dict[str, object]:
    return {
        "model": "demo",
        "messages": [{"role": "user", "content": "inspect " + ("m" * message_size)}],
        "stream": True,
        "tools": tools,
    }


def test_payload_under_budget_returns_unchanged() -> None:
    tools = [_tool("ssh_exec"), _tool("task_complete")]
    payload = _payload(tools)

    result = fit_tools_to_context_budget(
        payload=payload,
        tools=tools,
        budget=build_request_budget(8192),
        estimator=RequestEstimator(),
    )

    assert result.action == "unchanged"
    assert result.payload is payload
    assert result.dropped_tool_names == ()
    assert result.kept_tool_names == ("ssh_exec", "task_complete")


def test_request_budget_keeps_tokenizer_slop_away_from_hard_context_edge() -> None:
    budget = build_request_budget(16384)

    assert budget.reserve_completion_tokens == 1024
    assert budget.safety_margin_tokens == 2048
    assert budget.tokenizer_slop_tokens == 512
    assert budget.effective_prompt_budget == 12800


def test_payload_over_budget_drops_optional_tools_before_required_tools() -> None:
    tools = [
        _tool("artifact_read", size=3000),
        _tool("web_search", size=3000),
        _tool("ssh_exec", size=100),
        _tool("ssh_file_read", size=100),
        _tool("task_complete", size=100),
        _tool("task_fail", size=100),
    ]
    payload = _payload(tools, message_size=300)

    result = fit_tools_to_context_budget(
        payload=payload,
        tools=tools,
        budget=build_request_budget(4096),
        estimator=RequestEstimator(),
    )

    assert result.action == "reduced_tools"
    assert set(result.dropped_tool_names) <= {"artifact_read", "web_search"}
    assert result.dropped_tool_names
    assert {"ssh_exec", "ssh_file_read", "task_complete", "task_fail"} <= set(result.kept_tool_names)


def test_ssh_inventory_preserves_remote_read_and_lightweight_control_tools() -> None:
    tools = [
        _tool("artifact_grep", size=900),
        _tool("dir_list", size=900),
        _tool("ssh_exec", size=900),
        _tool("ssh_file_read", size=900),
        _tool("loop_status", size=100),
        _tool("ask_human", size=100),
        _tool("log_note", size=100),
        _tool("memory_update", size=100),
        _tool("task_complete", size=100),
        _tool("task_fail", size=100),
    ]
    payload = _payload(tools, message_size=300)

    result = fit_tools_to_context_budget(
        payload=payload,
        tools=tools,
        budget=build_request_budget(4096),
        estimator=RequestEstimator(),
    )

    assert {"ssh_exec", "ssh_file_read", "loop_status", "ask_human", "log_note", "memory_update"} <= set(
        result.kept_tool_names
    )
    assert set(result.dropped_tool_names) <= {"artifact_grep", "dir_list"}
    assert result.dropped_tool_names


def test_remote_edit_intent_preserves_requested_mutation_tool_and_read_tool() -> None:
    tools = [
        _tool("artifact_read", size=900),
        _tool("ssh_exec", size=900),
        _tool("ssh_file_read", size=100),
        _tool("ssh_file_patch", size=100),
        _tool("task_complete", size=100),
        _tool("task_fail", size=100),
    ]
    payload = _payload(tools, message_size=300)

    result = fit_tools_to_context_budget(
        payload=payload,
        tools=tools,
        budget=build_request_budget(3072),
        requested_tool_name="ssh_file_patch",
        estimator=RequestEstimator(),
    )

    assert "ssh_file_patch" in result.kept_tool_names
    assert "ssh_file_read" in result.kept_tool_names


def test_staged_execution_preserves_step_completion_tools() -> None:
    tools = [
        _tool("artifact_print", size=1200),
        _tool("shell_exec", size=1200),
        _tool("step_complete", size=100),
        _tool("step_fail", size=100),
    ]
    payload = _payload(tools, message_size=300)

    result = fit_tools_to_context_budget(
        payload=payload,
        tools=tools,
        budget=build_request_budget(3072),
        mode="staged",
        estimator=RequestEstimator(),
    )

    assert {"step_complete", "step_fail"} <= set(result.kept_tool_names)


def test_messages_plus_required_tools_can_exceed_budget() -> None:
    tools = [_tool("task_complete", size=100), _tool("task_fail", size=100)]
    payload = _payload(tools, message_size=20000)

    result = fit_tools_to_context_budget(
        payload=payload,
        tools=tools,
        budget=build_request_budget(2048),
        estimator=RequestEstimator(),
    )

    assert result.action == "exceeded"
    assert result.footprint.over_budget_tokens > 0


def test_required_tool_descriptions_are_slimmed_before_declaring_budget_exceeded() -> None:
    tools = [
        _tool("file_read", size=1000),
        _tool("file_write", size=1000),
        _tool("task_complete", size=1000),
        _tool("task_fail", size=1000),
    ]
    payload = _payload(tools, message_size=80)

    result = fit_tools_to_context_budget(
        payload=payload,
        tools=tools,
        budget=build_request_budget(3072),
        requested_tool_name="file_write",
        estimator=RequestEstimator(),
    )

    assert result.action == "reduced_tools"
    assert result.footprint.over_budget_tokens == 0
    descriptions = [
        str(tool["function"]["description"])
        for tool in result.payload["tools"]
    ]
    assert descriptions
    assert all(len(description) <= 80 for description in descriptions)
