from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from smallctl.client import OpenAICompatClient
from smallctl.graph.tool_call_parser import ToolCallParseResult, parse_tool_calls
from smallctl.graph.tool_inline_parsing import _extract_inline_tool_calls
from smallctl.tools.base import ToolSpec, build_tool_schema
from smallctl.tools.registry import ToolRegistry


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="task_complete",
            description="finish the task",
            schema=build_tool_schema(
                properties={"message": {"type": "string"}},
                required=["message"],
            ),
            handler=lambda **kwargs: kwargs,
        )
    )
    registry.register(
        ToolSpec(
            name="file_read",
            description="read a file",
            schema=build_tool_schema(
                properties={"path": {"type": "string"}},
                required=["path"],
            ),
            handler=lambda **kwargs: kwargs,
        )
    )
    return registry


class _Harness:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or _make_registry()
        self.state = SimpleNamespace(scratchpad={}, messages=[])
        self.client = SimpleNamespace(model="google/gemma-4-e2b-it")
        self.thinking_start_tag = "<think>"
        self.thinking_end_tag = "</think>"

    def _runlog(self, *args: object, **kwargs: object) -> None:
        pass


def _parse(
    assistant_text: str,
    *,
    model_name: str = "google/gemma-4-e2b-it",
    registry: ToolRegistry | None = None,
) -> ToolCallParseResult:
    stream = SimpleNamespace(
        assistant_text=assistant_text, thinking_text="", tool_calls=[]
    )
    harness = _Harness(registry=registry)
    deps = SimpleNamespace(harness=harness)
    graph_state = SimpleNamespace(run_mode="loop")
    return parse_tool_calls(stream, [], graph_state, deps, model_name=model_name)


def test_extract_inline_raw_function_call_with_trailing_text_on_same_line() -> None:
    """Gemma-4-e2b-it often emits `task_complete(...)` and then repeats the
    answer on the same line. The parser must still extract the call."""
    answer = (
        "1. **File Path Created:** `./temp/catch-the-stars.html`\n\n"
        "2. **How to Run It:** Open the file in a browser."
    )
    raw_call = "task_complete(message='Done')"
    text = (
        f"{answer}\n\n{raw_call}1. **File Path Created:** `./temp/catch-the-stars.html`"
    )

    cleaned, calls = _extract_inline_tool_calls(
        text,
        model_name="google/gemma-4-e2b-it",
        allowed_raw_function_names={"task_complete"},
    )

    assert len(calls) == 1
    assert calls[0].tool_name == "task_complete"
    assert calls[0].args == {"message": "Done"}
    # Only the matched call portion is removed, not the trailing duplicate text.
    assert raw_call not in cleaned
    assert "File Path Created" in cleaned


def test_terminal_call_truncates_duplicated_trailing_assistant_text() -> None:
    """When a terminal call is followed by a repeat of the assistant's answer,
    the final assistant text should stop at the call boundary."""
    answer = (
        "1. **File Path Created:** `./temp/catch-the-stars.html`\n\n"
        "2. **How to Run It:** Open the file in a browser.\n\n"
        "3. **Controls:** Left/Right arrows."
    )
    raw_call = "task_complete(message='Successfully created the game.')"
    text = f"{answer}\n\n{raw_call}{answer}"

    result = _parse(text)

    assert len(result.pending_tool_calls) == 1
    assert result.pending_tool_calls[0].tool_name == "task_complete"
    assert result.final_assistant_text == answer


def test_terminal_call_at_start_leaves_empty_assistant_text_for_non_gemma() -> None:
    result = _parse("task_complete(message='Done')", model_name="openai/gpt-4o")

    assert len(result.pending_tool_calls) == 1
    assert result.pending_tool_calls[0].tool_name == "task_complete"
    assert result.final_assistant_text == ""


def test_terminal_call_at_start_recovers_message_for_gemma() -> None:
    result = _parse("task_complete(message='Done')")

    assert len(result.pending_tool_calls) == 1
    assert result.pending_tool_calls[0].tool_name == "task_complete"
    assert result.final_assistant_text == "Done"


def test_action_tool_call_does_not_truncate_following_text() -> None:
    """Non-terminal inline calls should keep surrounding explanation intact."""
    text = (
        "I will read the file first.\n"
        "file_read(path='src/app.py')\n"
        "Then I will inspect the result."
    )

    result = _parse(text)

    assert [c.tool_name for c in result.pending_tool_calls] == ["file_read"]
    assert "I will read the file first." in result.final_assistant_text
    assert "Then I will inspect the result." in result.final_assistant_text


def test_extract_inline_json_tool_call_with_tool_call_key() -> None:
    """Small models sometimes emit a flat JSON object keyed by `tool_call`
    with the remaining top-level fields as arguments."""
    text = '{"tool_call": "artifact_read", "artifact_id": "A0001"}'

    cleaned, calls = _extract_inline_tool_calls(
        text,
        allowed_raw_function_names={"artifact_read", "task_complete"},
    )

    assert len(calls) == 1
    assert calls[0].tool_name == "artifact_read"
    assert calls[0].args == {"artifact_id": "A0001"}
    assert cleaned == ""


def test_parse_tool_calls_recovers_tool_call_key_from_thinking_text() -> None:
    """When the assistant text is unhelpful prose but the thinking block
    contains a flat `tool_call` JSON object, the call should be recovered."""
    stream = SimpleNamespace(
        assistant_text="A0001",
        thinking_text=(
            "<think>My goal is to list the report.\n"
            "</think>\n"
            '{"tool_call": "artifact_read", "artifact_id": "A0001"}'
        ),
        tool_calls=[],
    )
    harness = _Harness()
    # Register artifact_read so the parser knows it is allowed.
    harness.registry.register(
        ToolSpec(
            name="artifact_read",
            description="read an artifact",
            schema=build_tool_schema(
                properties={"artifact_id": {"type": "string"}},
                required=["artifact_id"],
            ),
            handler=lambda **kwargs: kwargs,
        )
    )
    deps = SimpleNamespace(harness=harness)
    graph_state = SimpleNamespace(run_mode="chat")

    result = parse_tool_calls(stream, [], graph_state, deps)

    assert len(result.pending_tool_calls) == 1
    assert result.pending_tool_calls[0].tool_name == "artifact_read"
    assert result.pending_tool_calls[0].args["artifact_id"] == "A0001"


def test_real_gemma_run_step_assistant_text_is_deduplicated() -> None:
    """Regression guard for the exact step-3 assistant text from the
    4f5e7b89 Gemma-4-e2b-it run that produced double output."""
    answer = (
        "1. **File Path Created:** `./temp/catch-the-stars.html`\n\n"
        "2. **How to Run It:** Open the `catch-the-stars.html` file directly in any modern web browser (e.g., Chrome, Firefox). The game will run entirely client-side as it contains all necessary HTML, CSS, and JavaScript within the single file.\n\n"
        "3. **Controls:**\n"
        "    *   **Move Left:** Left arrow key or 'A'.\n"
        "    *   **Move Right:** Right arrow key or 'D'.\n\n"
        "4. **Known Limitations:**\n"
        "    *   The game relies on the browser's rendering capabilities and JavaScript execution.\n"
        "    *   The game uses simple DOM manipulation for the stars, which is not a high-performance canvas implementation.\n"
        "    *   The game logic is entirely contained within the single HTML file, limiting complexity compared to a multi-file project."
    )
    raw_call = (
        'task_complete(message=\'Successfully created the single-file HTML game "Catch the Stars" '
        "at ./temp/catch-the-stars.html and provided the required verification explanation.')"
    )
    text = f"{answer}\n\n{raw_call}{answer}"

    result = _parse(text)

    assert len(result.pending_tool_calls) == 1
    assert result.pending_tool_calls[0].tool_name == "task_complete"
    assert result.final_assistant_text == answer


def test_gemma_brace_tool_call_with_quote_tokens() -> None:
    """Gemma-4-e2b-it sometimes emits calls as `call:tool_name{key:<|\"|>value<|\"|>}<tool_call|>`."""
    q = '<|"|>'
    text = (
        f"<tool_call>call:ssh_exec{{command:{q}docker pull vikunja/vikunja && "
        f"docker run -d --name vikunja -p 80:80 vikunja/vikunja{q},"
        f"host:{q}192.168.1.89{q},"
        f"password:{q}secret{q},"
        f"target:{q}root@192.168.1.89{q},"
        f"user:{q}root{q}}}<tool_call|>"
    )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="ssh_exec",
            description="run ssh command",
            schema=build_tool_schema(
                properties={"command": {"type": "string"}, "host": {"type": "string"}},
                required=["command", "host"],
            ),
            handler=lambda **kwargs: kwargs,
        )
    )

    result = _parse(text, registry=registry)

    assert [c.tool_name for c in result.pending_tool_calls] == ["ssh_exec"]
    assert result.pending_tool_calls[0].args["command"] == (
        "docker pull vikunja/vikunja && docker run -d --name vikunja -p 80:80 vikunja/vikunja"
    )
    assert result.pending_tool_calls[0].args["host"] == "192.168.1.89"
    assert result.pending_tool_calls[0].args["password"] == "secret"
    assert result.pending_tool_calls[0].args["target"] == "root@192.168.1.89"
    assert result.pending_tool_calls[0].args["user"] == "root"


def test_gemma_brace_tool_call_without_call_prefix() -> None:
    """The brace format may omit the `call:` prefix."""
    q = '<|"|>'
    text = f"<|tool_call>ssh_exec{{command:{q}ls{q},host:{q}1.2.3.4{q}}}<tool_call|>"

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="ssh_exec",
            description="run ssh command",
            schema=build_tool_schema(
                properties={"command": {"type": "string"}, "host": {"type": "string"}},
                required=["command", "host"],
            ),
            handler=lambda **kwargs: kwargs,
        )
    )

    result = _parse(text, registry=registry)

    assert [c.tool_name for c in result.pending_tool_calls] == ["ssh_exec"]
    assert result.pending_tool_calls[0].args == {"command": "ls", "host": "1.2.3.4"}


def test_gemma_brace_task_complete_with_quote_tokens() -> None:
    """The brace format also applies to terminal tools like task_complete."""
    q = '<|"|>'
    text = f"<tool_call>call:task_complete{{message:{q}Done{q}}}<tool_call|>"

    result = _parse(text)

    assert [c.tool_name for c in result.pending_tool_calls] == ["task_complete"]
    assert result.pending_tool_calls[0].args == {"message": "Done"}


def test_gemma_quote_token_fragment_in_assistant_text_is_stripped() -> None:
    """Gemma-4-e2b-it may emit the tool call in reasoning and only a quote-token
    fragment such as '|>root<|' in the assistant content stream. The fragment
    must be stripped so it does not pollute conversation history."""
    q = '<|"|>'
    thinking_text = (
        "<think>I will run the command on the remote host.</think>\n"
        f"<|tool_call>call:ssh_exec{{command:{q}docker ps -a{q},"
        f"host:{q}192.168.1.89{q},password:{q}secret{q},user:{q}root{q}}}<tool_call|>"
    )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="ssh_exec",
            description="run ssh command",
            schema=build_tool_schema(
                properties={"command": {"type": "string"}, "host": {"type": "string"}},
                required=["command", "host"],
            ),
            handler=lambda **kwargs: kwargs,
        )
    )
    harness = _Harness(registry=registry)
    stream = SimpleNamespace(
        assistant_text="|>root<|", thinking_text=thinking_text, tool_calls=[]
    )
    deps = SimpleNamespace(harness=harness)
    graph_state = SimpleNamespace(run_mode="loop")
    result = parse_tool_calls(
        stream, [], graph_state, deps, model_name="google/gemma-4-e2b-it"
    )

    assert [c.tool_name for c in result.pending_tool_calls] == ["ssh_exec"]
    assert result.pending_tool_calls[0].args["command"] == "docker ps -a"
    assert result.pending_tool_calls[0].args["host"] == "192.168.1.89"
    assert result.pending_tool_calls[0].args["user"] == "root"
    assert "|>root<|" not in result.final_assistant_text
    assert "<think>" not in result.final_thinking_text
    assert "</think>" not in result.final_thinking_text
    assert "I will run the command on the remote host." in result.final_thinking_text


def test_gemma_brace_tool_call_without_closing_tag() -> None:
    """Small models sometimes emit `<|tool_call>call:tool{...}` without a closing tag."""
    q = '<|"|>'
    text = f"<|tool_call>call:dir_list{{path:{q}.{q}}}"

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="dir_list",
            description="list directory",
            schema=build_tool_schema(
                properties={"path": {"type": "string"}},
                required=["path"],
            ),
            handler=lambda **kwargs: kwargs,
        )
    )

    result = _parse(text, registry=registry)

    assert [c.tool_name for c in result.pending_tool_calls] == ["dir_list"]
    assert result.pending_tool_calls[0].args == {"path": "."}


def test_standard_tool_call_still_works_after_gemma_fix() -> None:
    """The normal `<tool_call>tool_name(...)</tool_call>` path must remain intact."""
    text = (
        "<tool_call>ssh_exec(command='docker pull vikunja/vikunja',host='192.168.1.89')"
        "</tool_call>"
    )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="ssh_exec",
            description="run ssh command",
            schema=build_tool_schema(
                properties={"command": {"type": "string"}, "host": {"type": "string"}},
                required=["command", "host"],
            ),
            handler=lambda **kwargs: kwargs,
        )
    )

    result = _parse(text, registry=registry)

    assert [c.tool_name for c in result.pending_tool_calls] == ["ssh_exec"]
    assert result.pending_tool_calls[0].args == {
        "command": "docker pull vikunja/vikunja",
        "host": "192.168.1.89",
    }


def test_case_insensitive_json_fence_is_parsed() -> None:
    """Gemma variants sometimes emit uppercase or mixed-case fence labels."""
    text = '```JSON\n{"name": "task_complete", "arguments": {"message": "ok"}}\n```'

    result = _parse(text)

    assert [c.tool_name for c in result.pending_tool_calls] == ["task_complete"]
    assert result.pending_tool_calls[0].args == {"message": "ok"}


def test_unlabeled_json_fence_is_parsed() -> None:
    """Small models may emit a plain triple-backtick fence around a tool JSON blob."""
    text = '```\n{"name": "task_complete", "arguments": {"message": "ok"}}\n```'

    result = _parse(text)

    assert [c.tool_name for c in result.pending_tool_calls] == ["task_complete"]
    assert result.pending_tool_calls[0].args == {"message": "ok"}


def test_unlabeled_json_array_fence_parses_multiple_calls() -> None:
    text = (
        "```\n"
        '[{"name": "ssh_exec", "arguments": {"command": "first"}},'
        ' {"name": "ssh_exec", "arguments": {"command": "second"}}]\n'
        "```"
    )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="ssh_exec",
            description="run ssh command",
            schema=build_tool_schema(
                properties={"command": {"type": "string"}},
                required=["command"],
            ),
            handler=lambda **kwargs: kwargs,
        )
    )

    result = _parse(text, registry=registry)

    assert [c.tool_name for c in result.pending_tool_calls] == ["ssh_exec", "ssh_exec"]
    assert result.pending_tool_calls[0].args == {"command": "first"}
    assert result.pending_tool_calls[1].args == {"command": "second"}


def test_plain_text_fence_is_not_parsed_as_tool_call() -> None:
    """Ordinary code blocks without JSON tool shapes must stay in assistant text."""
    text = "```\nhello world\n```"

    result = _parse(text)

    assert result.pending_tool_calls == []
    assert "hello world" in result.final_assistant_text


def _gemma_chunks(
    reasoning_text: str,
    *,
    backend_model_name: str = "C:\\\\Users\\\\svaye\\\\.lmstudio\\\\models\\\\unsloth\\\\gemma-4-E2B-it-GGUF\\\\gemma-4-E2B-it-IQ4_XS.gguf",
) -> list[dict[str, Any]]:
    """Build a minimal chunk list with reasoning delivered in `reasoning_content`."""
    return [
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": reasoning_text,
                    }
                }
            ],
            "model": backend_model_name,
        },
        {"choices": [], "model": backend_model_name},
    ]


def test_gemma_reasoning_field_preserves_tool_call_wrappers() -> None:
    """The stream collector must not strip `<|tool_call>...<tool_call|>` from Gemma
    reasoning fields, and the parser must recover the call from inside the
    `<think>` block."""
    q = '<|"|>'
    reasoning = (
        "<think>Run dockerd directly to see the fatal error.</think>\n"
        f"<|tool_call>call:ssh_exec{{command:{q}timeout 15 dockerd --debug{q},"
        f"host:{q}192.168.1.89{q},password:{q}secret{q},user:{q}root{q}}}"
        f"<tool_call|>"
    )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="ssh_exec",
            description="run ssh command",
            schema=build_tool_schema(
                properties={"command": {"type": "string"}, "host": {"type": "string"}},
                required=["command", "host"],
            ),
            handler=lambda **kwargs: kwargs,
        )
    )

    stream = OpenAICompatClient.collect_stream(
        _gemma_chunks(reasoning),
        reasoning_mode="auto",
        thinking_start_tag="<think>",
        thinking_end_tag="</think>",
    )
    harness = _Harness(registry=registry)
    deps = SimpleNamespace(harness=harness)
    graph_state = SimpleNamespace(run_mode="loop")
    result = parse_tool_calls(stream, [], graph_state, deps, model_name="Gemma 4 e2b")

    assert [c.tool_name for c in result.pending_tool_calls] == ["ssh_exec"]
    assert result.pending_tool_calls[0].args["command"] == "timeout 15 dockerd --debug"
    assert result.pending_tool_calls[0].args["host"] == "192.168.1.89"
    assert "<|tool_call>" not in result.final_thinking_text
    assert "<tool_call|>" not in result.final_thinking_text


def test_non_gemma_reasoning_field_strips_tool_call_wrappers() -> None:
    """Non-Gemma models still have hallucinated tool-call XML tokens removed from
    the reasoning field so they cannot be misinterpreted as real calls."""
    reasoning = (
        "<think>I will check the logs.</think>\n"
        "<tool_call>call:ssh_exec{command:'docker logs foo',host:'1.2.3.4'}</tool_call>"
    )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="ssh_exec",
            description="run ssh command",
            schema=build_tool_schema(
                properties={"command": {"type": "string"}, "host": {"type": "string"}},
                required=["command", "host"],
            ),
            handler=lambda **kwargs: kwargs,
        )
    )

    stream = OpenAICompatClient.collect_stream(
        _gemma_chunks(reasoning, backend_model_name="qwen/qwen-2.5-7b-instruct"),
        reasoning_mode="auto",
        thinking_start_tag="<think>",
        thinking_end_tag="</think>",
    )
    harness = _Harness(registry=registry)
    deps = SimpleNamespace(harness=harness)
    graph_state = SimpleNamespace(run_mode="loop")
    result = parse_tool_calls(
        stream, [], graph_state, deps, model_name="qwen/qwen-2.5-7b-instruct"
    )

    # The hallucinated wrapper is removed; the bare brace payload is not in a
    # format the parser recognizes as a real call, so no tool calls are emitted.
    assert result.pending_tool_calls == []
    assert "<tool_call>" not in stream.thinking_text
    assert "</tool_call>" not in stream.thinking_text


def test_flat_inline_json_preserves_argument_key_matching_tool_name() -> None:
    text = '{"tool_call": "file_read", "path": "README.md", "file_read": "literal argument value"}'

    result = _parse(text)

    assert [c.tool_name for c in result.pending_tool_calls] == ["file_read"]
    assert result.pending_tool_calls[0].args == {
        "path": "README.md",
        "file_read": "literal argument value",
    }
