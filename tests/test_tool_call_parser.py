from __future__ import annotations

from types import SimpleNamespace

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


def _parse(assistant_text: str, *, model_name: str = "google/gemma-4-e2b-it", registry: ToolRegistry | None = None) -> ToolCallParseResult:
    stream = SimpleNamespace(assistant_text=assistant_text, thinking_text="", tool_calls=[])
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
    text = f"{answer}\n\n{raw_call}1. **File Path Created:** `./temp/catch-the-stars.html`"

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
        "task_complete(message='Successfully created the single-file HTML game \"Catch the Stars\" "
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
    text = (
        f"<|tool_call>ssh_exec{{command:{q}ls{q},host:{q}1.2.3.4{q}}}"
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
    assert result.pending_tool_calls[0].args == {"command": "docker pull vikunja/vikunja", "host": "192.168.1.89"}
