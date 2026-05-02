from __future__ import annotations

from smallctl.client import get_provider_adapter
from smallctl.client.provider_adapters import sanitize_messages_for_lmstudio
from smallctl.client.provider_adapters import sanitize_messages_for_openrouter


def test_adapter_registry_returns_profile_specific_adapter() -> None:
    assert get_provider_adapter("lmstudio").name == "lmstudio"
    assert get_provider_adapter("openrouter").name == "openrouter"


def test_adapter_registry_falls_back_to_generic() -> None:
    adapter = get_provider_adapter("unknown-provider")
    assert adapter.name == "generic"
    assert adapter.stream_policy.supports_stream_options is True


def test_provider_policy_disables_stream_options_for_lmstudio_and_openrouter() -> None:
    assert get_provider_adapter("lmstudio").stream_policy.supports_stream_options is False
    assert get_provider_adapter("openrouter").stream_policy.supports_stream_options is False


def test_openrouter_sanitizer_rewrites_orphan_tool_messages() -> None:
    messages = [
        {"role": "tool", "name": "shell_exec", "tool_call_id": "call_1", "content": "ok"},
    ]
    sanitized = sanitize_messages_for_openrouter(messages)
    assert sanitized[0]["role"] == "user"
    assert "Orphan tool result" in str(sanitized[0]["content"])


def test_openrouter_sanitizer_drops_empty_unresolved_assistant_tool_call() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell_exec", "arguments": "{\"command\":\"whoami\"}"},
                }
            ],
        },
        {
            "role": "system",
            "content": "This is a remote task. Use `ssh_exec`, not local `shell_exec`.",
        },
    ]

    sanitized = sanitize_messages_for_openrouter(messages)

    assert sanitized == [
        {
            "role": "system",
            "content": "This is a remote task. Use `ssh_exec`, not local `shell_exec`.",
        }
    ]


def test_openrouter_sanitizer_merges_multiple_system_messages_and_preserves_tool_pairs() -> None:
    messages = [
        {"role": "system", "content": "Base prompt."},
        {"role": "user", "content": "continue"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell_exec", "arguments": "{\"command\":\"pytest\"}"},
                }
            ],
        },
        {"role": "system", "content": "Recovery nudge."},
        {"role": "tool", "name": "shell_exec", "tool_call_id": "call_1", "content": "ok"},
    ]

    sanitized = sanitize_messages_for_openrouter(messages)

    assert [message["role"] for message in sanitized] == ["system", "user", "assistant", "tool"]
    assert sanitized[0]["content"] == "Base prompt.\n\nRecovery nudge."
    assert sanitized[2]["tool_calls"][0]["id"] == "call_1"
    assert sanitized[3]["tool_call_id"] == "call_1"


def test_openrouter_sanitizer_strips_assistant_content_when_tool_calls_present() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "Let me inspect the file before patching it.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "file_read", "arguments": "{\"path\":\"portainer_cli.py\"}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "file_read",
            "tool_call_id": "call_1",
            "content": "file contents",
        },
    ]

    sanitized = sanitize_messages_for_openrouter(messages)

    assert sanitized == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "file_read", "arguments": "{\"path\":\"portainer_cli.py\"}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "file_read",
            "tool_call_id": "call_1",
            "content": "file contents",
        },
    ]


def test_openrouter_sanitizer_rewrites_unavailable_tool_history_as_user_context() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "task_complete", "arguments": "{\"message\":\"done\"}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "task_complete",
            "tool_call_id": "call_1",
            "content": "task complete blocked",
        },
        {"role": "user", "content": "keep going"},
    ]

    sanitized = sanitize_messages_for_openrouter(
        messages,
        available_tool_names={"shell_exec"},
    )

    assert sanitized == [
        {
            "role": "user",
            "content": "[OpenRouter compatibility] Orphan tool result from task_complete (tool_call_id=call_1):\ntask complete blocked",
        },
        {"role": "user", "content": "keep going"},
    ]


def test_openrouter_mutate_payload_drops_history_for_unavailable_tools() -> None:
    adapter = get_provider_adapter("openrouter")
    payload = {
        "model": "test-model",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "task_complete", "arguments": "{\"message\":\"done\"}"},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "task_complete",
                "tool_call_id": "call_1",
                "content": "task complete blocked",
            },
            {"role": "user", "content": "keep going"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "shell_exec", "description": "Run shell", "parameters": {"type": "object"}},
            }
        ],
    }

    mutated = adapter.mutate_payload(payload)

    assert mutated["messages"] == [
        {
            "role": "user",
            "content": "[OpenRouter compatibility] Orphan tool result from task_complete (tool_call_id=call_1):\ntask complete blocked",
        },
        {"role": "user", "content": "keep going"},
    ]


def test_lmstudio_sanitizer_rewrites_completed_task_complete_pair_before_followup_user() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "task_complete", "arguments": "{\"message\":\"done\"}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "task_complete",
            "tool_call_id": "call_1",
            "content": "Patched the compose file and verified the port mapping.",
        },
        {"role": "user", "content": "Check compose too and confirm the published port."},
    ]

    sanitized = sanitize_messages_for_lmstudio(messages)

    assert sanitized == [
        {
            "role": "assistant",
            "content": "Previous task summary: Patched the compose file and verified the port mapping.",
        },
        {"role": "user", "content": "Check compose too and confirm the published port."},
    ]


def test_lmstudio_sanitizer_keeps_completed_task_complete_pair_without_followup_user() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "task_complete", "arguments": "{\"message\":\"done\"}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "task_complete",
            "tool_call_id": "call_1",
            "content": "Patched the compose file and verified the port mapping.",
        },
    ]

    sanitized = sanitize_messages_for_lmstudio(messages)

    assert sanitized == messages


def test_lmstudio_sanitizer_does_not_rewrite_non_terminal_tool_pairs() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell_exec", "arguments": "{\"command\":\"docker compose ps\"}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "shell_exec",
            "tool_call_id": "call_1",
            "content": "compose status output",
        },
        {"role": "user", "content": "Check compose too and confirm the published port."},
    ]

    sanitized = sanitize_messages_for_lmstudio(messages)

    assert sanitized == messages


def test_lmstudio_mutate_payload_flattens_oneof_tool_schema() -> None:
    adapter = get_provider_adapter("lmstudio")
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "search the web"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch a web result",
                    "parameters": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string"},
                                    "max_chars": {"type": "integer"},
                                },
                                "required": ["url"],
                                "additionalProperties": False,
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "result_id": {"type": "string"},
                                    "max_chars": {"type": "integer"},
                                },
                                "required": ["result_id"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                },
            }
        ],
    }

    mutated = adapter.mutate_payload(payload)
    parameters = mutated["tools"][0]["function"]["parameters"]

    assert "oneOf" not in parameters
    assert parameters["type"] == "object"
    assert set(parameters["properties"]) == {"url", "result_id", "max_chars"}
    assert "required" not in parameters
    assert parameters["additionalProperties"] is False


class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _FakeExc:
    def __init__(self, status_code: int, text: str) -> None:
        self.response = _FakeResponse(status_code, text)


def test_retry_hint_for_stream_options_error() -> None:
    adapter = get_provider_adapter("generic")
    exc = _FakeExc(400, "unknown field: stream_options.include_usage")
    assert adapter.should_retry_without_stream_options(exc) is True


def test_no_retry_hint_for_unrelated_400_error() -> None:
    adapter = get_provider_adapter("generic")
    exc = _FakeExc(400, "invalid model name")
    assert adapter.should_retry_without_stream_options(exc) is False
