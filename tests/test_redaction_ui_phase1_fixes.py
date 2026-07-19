from __future__ import annotations

import json

import pytest

from smallctl.graph.model_stream_resolution_support import (
    _chunk_error_failure_message,
    _chunk_error_failure_type,
)
from smallctl.logging_utils import RunLogger
from smallctl.redaction import (
    redact_sensitive_data,
    redact_sensitive_messages,
    redact_sensitive_text,
)
from smallctl.ui.display import (
    _CRITICAL_EVENTS,
    format_recovery_banner,
    format_run_log_row,
)


_C2_SAMPLES = (
    ("OPENROUTER_API_KEY=sk-or-v1-abc123def456", "sk-or-v1-abc123def456"),
    ("export OPENAI_API_KEY=sk-proj-xyz789", "sk-proj-xyz789"),
    ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
    ("SECRET_KEY=django-super-secret-value", "django-super-secret-value"),
    ("GH_TOKEN=ghp_abcdefghijklmnop012345", "ghp_abcdefghijklmnop012345"),
    ("SSH_PASSWORD=hunter2", "hunter2"),
    ("sshpass -p hunter2 ssh root@192.0.2.10", "hunter2"),
    ("postgres://user:hunter2@db/app", "hunter2"),
)


@pytest.mark.parametrize("text,secret", _C2_SAMPLES)
def test_c2_common_secret_formats_are_redacted(text: str, secret: str) -> None:
    redacted = redact_sensitive_text(text)
    assert secret not in redacted
    assert "REDACTED" in redacted


def test_c2_bearer_jwt_token_is_fully_redacted() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    redacted = redact_sensitive_text(f"Authorization: Bearer {jwt}")
    assert jwt not in redacted
    assert "eyJzdWIiOiIxMjM0In0" not in redacted
    assert "REDACTED" in redacted


@pytest.mark.parametrize(
    "text",
    (
        "token_secret: str",
        "TOKEN_SECRET: str",
        "password: str",
        "password:\n    raise ValueError('missing password')",
        "TOKEN_SECRET:\n    raise ValueError('missing token')",
        "token = parse_token(source)",
        '"token": token_value',
        "token: TokenType",
        "token: IPv4Address",
    ),
)
def test_c2_code_constructs_are_not_redacted(text: str) -> None:
    assert redact_sensitive_text(text) == text


@pytest.mark.parametrize(
    ("text", "secret"),
    (
        ("provider error: api_key=sk-testsecret123", "sk-testsecret123"),
        ("api_key: sk-live-secret", "sk-live-secret"),
        ("token: swordfish", "swordfish"),
        ("token: abc123", "abc123"),
        ("Authorization: Token api-token-secret", "api-token-secret"),
    ),
)
def test_c2_unquoted_secret_assignments_remain_redacted(text: str, secret: str) -> None:
    redacted = redact_sensitive_text(text)
    assert secret not in redacted
    assert "REDACTED" in redacted


def test_c2_provider_bound_code_annotations_are_not_poisoned() -> None:
    code = """class BookStackClient:
    token_id: str
    token_secret: str

    def validate(self) -> None:
        if not self.token_id or not self.token_secret:
            raise ValueError("missing credentials")
"""
    messages = [{"role": "tool", "content": code}]

    assert redact_sensitive_messages(messages)[0]["content"] == code


def test_c2_dict_path_redacts_prefixed_sensitive_keys() -> None:
    payload = {
        "OPENAI_API_KEY": "sk-proj-xyz789",
        "SECRET_KEY": "django-super-secret-value",
        "GH_TOKEN": "ghp_abcdefghijklmnop012345",
        "nested": {"AWS_SECRET_ACCESS_KEY": "aws-secret-value"},
        "safe": "hello",
    }
    redacted = redact_sensitive_data(payload)
    assert redacted["OPENAI_API_KEY"] != "sk-proj-xyz789"
    assert redacted["SECRET_KEY"] != "django-super-secret-value"
    assert redacted["GH_TOKEN"] != "ghp_abcdefghijklmnop012345"
    assert redacted["nested"]["AWS_SECRET_ACCESS_KEY"] != "aws-secret-value"
    assert redacted["safe"] == "hello"
    assert "sk-proj-xyz789" not in json.dumps(redacted)


def test_c2_messages_redact_list_content_parts() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "use OPENAI_API_KEY=sk-proj-xyz789 please"},
                {"type": "text", "text": "nothing sensitive here"},
            ],
        }
    ]
    redacted = redact_sensitive_messages(messages)
    parts = redacted[0]["content"]
    assert "sk-proj-xyz789" not in parts[0]["text"]
    assert "REDACTED" in parts[0]["text"]
    assert parts[1]["text"] == "nothing sensitive here"


def test_c2_messages_redact_tool_call_arguments() -> None:
    arguments = json.dumps({"command": "export OPENAI_API_KEY=sk-proj-xyz789 && env"})
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell_exec", "arguments": arguments},
                }
            ],
        }
    ]
    redacted = redact_sensitive_messages(messages)
    redacted_arguments = redacted[0]["tool_calls"][0]["function"]["arguments"]
    assert "sk-proj-xyz789" not in redacted_arguments
    assert "REDACTED" in redacted_arguments


def test_c2_messages_redact_non_json_tool_call_arguments() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell_exec", "arguments": "sshpass -p hunter2 ssh db"},
                }
            ],
        }
    ]
    redacted = redact_sensitive_messages(messages)
    assert "hunter2" not in redacted[0]["tool_calls"][0]["function"]["arguments"]


def test_c2_messages_without_secrets_are_unchanged() -> None:
    arguments = '{"command": "ls -la", "timeout": 30}'
    messages = [
        {"role": "user", "content": "list files"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "listing files now"}],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell_exec", "arguments": arguments},
                }
            ],
        },
    ]
    redacted = redact_sensitive_messages(messages)
    assert redacted == messages
    assert redacted[1]["tool_calls"][0]["function"]["arguments"] == arguments


def test_m14_content_policy_chunk_error_message() -> None:
    details = {
        "type": "content_policy_violation",
        "reason": "provider_content_policy_block",
        "provider_profile": "openrouter",
        "provider_error": "request blocked",
        "status_code": 403,
    }
    message = _chunk_error_failure_message(details)
    assert "content policy" in message.lower()
    assert _chunk_error_failure_type(details) == "content_policy_violation"


def test_m14_bare_403_chunk_error_message() -> None:
    message = _chunk_error_failure_message({"status_code": 403})
    assert "content policy" not in message.lower()
    assert "403" in message
    assert _chunk_error_failure_type({"status_code": 403}) == "provider"


def test_m14_chunk_error_falls_back_to_details_message() -> None:
    message = _chunk_error_failure_message({"message": "upstream exploded"})
    assert "upstream exploded" in message
    assert _chunk_error_failure_message({}) == "Upstream chunk error after retries"


def test_m14_fama_circuit_breaker_is_critical_and_formats() -> None:
    assert "fama_ssh_transport_circuit_breaker" in _CRITICAL_EVENTS
    row = {
        "channel": "harness",
        "event": "fama_ssh_transport_circuit_breaker",
        "message": "SSH transport failure requires remediation",
        "data": {
            "failure_kind": "transport",
            "previous_task_mode": "local_execute",
            "next_task_mode": "remote_execute",
            "required_tool": "ask_human",
        },
    }
    formatted = format_run_log_row(row)
    assert "circuit breaker" in formatted.lower()
    assert "transport" in formatted
    banner = format_recovery_banner("fama_ssh_transport_circuit_breaker", row["data"])
    assert "circuit breaker" in banner.lower()


def test_m15_raising_listener_does_not_propagate(tmp_path) -> None:
    logger = RunLogger(tmp_path / "run")

    def _boom(row: dict) -> None:
        raise RuntimeError("listener exploded")

    logger.set_listener(_boom)
    logger.log("harness", "listener_guard_test", "row survives listener failure")

    row = json.loads((logger.run_dir / "harness.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["event"] == "listener_guard_test"
