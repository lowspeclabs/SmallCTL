from __future__ import annotations

from types import SimpleNamespace

from smallctl.context.artifact_visibility import (
    is_prompt_visible_artifact,
    is_superseded_artifact,
)
from smallctl.graph.shell_outcomes import (
    _clear_shell_human_retry_state,
    _remember_shell_human_retry_state,
    _shell_human_retry_hint,
    _shell_ssh_retry_hint,
    _shell_workspace_relative_retry_hint,
)
from smallctl.harness.shell_attempts import (
    file_read_cache_key,
    shell_attempt_family_key,
    shell_attempt_is_diagnostic,
    shell_command_root,
    shell_tokens,
)
from smallctl.shell_utils import leading_command_tokens
from smallctl.shell_utils import is_read_only_shell_evidence_action, split_shell_segments
from smallctl.graph.model_stream_fallback_support import (
    _fallback_task_text,
    _looks_like_harness_recovery_message,
    _fallback_response_ready_for_early_exit,
)
from smallctl.graph.model_stream_fallback_recovery import (
    _build_text_write_fallback_prompt,
    _is_sub4b_write_timeout,
)
from smallctl.harness import Harness
from smallctl.harness.task_intent import (
    derive_task_contract,
    extract_intent_state,
    next_action_for_task,
)
from smallctl.harness.tool_dispatch import (
    attempt_tool_sanitization,
    maybe_reuse_file_read,
)
from smallctl.harness.task_classifier import (
    needs_contextual_loop_escalation,
    is_smalltalk,
    looks_like_action_request,
    looks_like_execution_followup,
    looks_like_readonly_chat_request,
    looks_like_shell_request,
    needs_loop_for_content_lookup,
    needs_memory_persistence,
)


def test_task_classifier_helpers_cover_core_run_mode_signals() -> None:
    assert is_smalltalk("hello")
    assert needs_loop_for_content_lookup("show me the lines 12-18 in the log")
    assert looks_like_execution_followup("please use the command")
    assert looks_like_action_request("run git status")
    assert needs_memory_persistence("remember this")
    assert looks_like_shell_request("run a shell command")
    assert looks_like_readonly_chat_request("what files are in this repo")


def test_task_classifier_contextual_loop_escalation_uses_recent_shell_context() -> None:
    messages = [
        SimpleNamespace(role="assistant", content="Try this command:\n```bash\npytest -q\n```"),
        SimpleNamespace(role="user", content="please use the command"),
    ]

    assert needs_contextual_loop_escalation(messages, "please use the command") is True
    assert needs_contextual_loop_escalation([], "please use the command") is False


def test_shell_attempt_helpers_normalize_wrapper_commands(tmp_path) -> None:
    command = "bash -lc 'pytest -h'"
    assert shell_tokens(command)[0] == "bash"
    assert shell_command_root(command) == "pytest"
    assert shell_attempt_family_key(command) == "shell_exec:pytest"
    assert shell_attempt_is_diagnostic(command) is True

    env_command = "env FOO=bar bash -lc 'pytest -q'"
    assert leading_command_tokens(env_command) == ["pytest", "-q"]
    assert shell_command_root(env_command) == "pytest"

    cache_key = file_read_cache_key(str(tmp_path), {"path": "notes/todo.txt", "start_line": 3, "end_line": 7})
    assert cache_key is not None
    assert "notes" in cache_key


def test_shell_utils_identify_read_only_shell_evidence_actions() -> None:
    readonly = "journalctl -u ssh --no-pager | tail -100"
    mutating = "cat /var/log/dpkg.log && rm -f /tmp/example"

    assert split_shell_segments(readonly) == ["journalctl -u ssh --no-pager", "tail -100"]
    assert is_read_only_shell_evidence_action(readonly) is True
    assert is_read_only_shell_evidence_action(mutating) is False


def test_model_stream_fallback_helpers_skip_recovery_prompts() -> None:
    harness = SimpleNamespace(state=SimpleNamespace(run_brief=SimpleNamespace(original_task="")))
    messages = [
        {"role": "system", "content": "please regenerate the full tool call from scratch"},
        {"role": "user", "content": "build the helper"},
    ]

    assert _looks_like_harness_recovery_message(messages[0]["content"]) is True
    assert _fallback_task_text(harness, messages) == "build the helper"
    assert _fallback_response_ready_for_early_exit("```python\nprint('hi')\n```", target_path="notes.py", path_confidence="high")


def test_model_stream_fallback_recovery_helpers_build_prompts_and_timeout_gates(monkeypatch) -> None:
    session = SimpleNamespace(
        write_target_path="notes.py",
        write_session_id="ws-1",
        write_session_intent="replace_file",
        write_current_section="imports",
        suggested_sections=["imports", "body"],
    )
    harness = SimpleNamespace(
        client=SimpleNamespace(model="qwen-4b"),
        state=SimpleNamespace(
            cwd="/tmp",
            run_brief=SimpleNamespace(original_task="write notes"),
        ),
    )
    monkeypatch.setattr(
        "smallctl.graph.model_stream_fallback_recovery.should_enable_complex_write_chat_draft",
        lambda *args, **kwargs: True,
    )

    prompt = _build_text_write_fallback_prompt(
        session=session,
        current_section="imports",
        remaining_sections=["body"],
        task_text="write notes",
    )

    assert "write session id" in prompt.lower()
    assert _is_sub4b_write_timeout(
        harness,
        error_text="stream timed out",
        error_details={"message": "deadline exceeded"},
    ) is True


def test_shell_outcomes_helpers_share_retry_state() -> None:
    harness = SimpleNamespace(state=SimpleNamespace(scratchpad={}))
    record = SimpleNamespace(
        args={"command": "bash -lc 'echo hi'"},
        result=SimpleNamespace(metadata={"reason": "unsupported_shell_syntax", "question": "Use bash?"}),
        tool_call_id="shell-1",
    )
    pending = SimpleNamespace(tool_name="shell_exec", args={"command": "bash -lc 'echo hi'"})

    _remember_shell_human_retry_state(harness, record)
    assert _shell_human_retry_hint(harness, pending) is not None

    ssh_hint = _shell_ssh_retry_hint(harness, SimpleNamespace(tool_name="shell_exec", args={"command": "ssh host"}))
    assert ssh_hint is not None

    workspace_hint = _shell_workspace_relative_retry_hint(
        harness,
        SimpleNamespace(tool_name="shell_exec", args={"command": "cat /temp/output.txt"}),
    )
    assert workspace_hint is not None

    _clear_shell_human_retry_state(harness)
    assert _shell_human_retry_hint(harness, pending) is None


def test_artifact_visibility_helpers_read_metadata_flags() -> None:
    visible = SimpleNamespace(metadata={"model_visible": True})
    hidden = SimpleNamespace(metadata={"model_visible": False})
    superseded = SimpleNamespace(metadata={"superseded_by": "artifact-2"})

    assert is_prompt_visible_artifact(visible) is True
    assert is_prompt_visible_artifact(hidden) is False
    assert is_superseded_artifact(superseded) is True


def test_tool_dispatch_helpers_reuse_cached_file_reads(tmp_path) -> None:
    artifact_id = "artifact-1"
    cached_artifact = SimpleNamespace(
        source=str(tmp_path / "notes.txt"),
        summary="cached summary",
    )
    harness = SimpleNamespace(
        state=SimpleNamespace(
            cwd=str(tmp_path),
            scratchpad={
                "file_read_cache": {
                    file_read_cache_key(str(tmp_path), {"path": "notes.txt"}): artifact_id,
                },
            },
            artifacts={artifact_id: cached_artifact},
        ),
        _runlog=lambda *args, **kwargs: None,
    )

    result = maybe_reuse_file_read(harness, tool_name="file_read", args={"path": "notes.txt"})

    assert result is not None
    assert result.success is True
    assert result.metadata["cache_hit"] is True
    assert result.metadata["artifact_id"] == artifact_id


def test_tool_dispatch_helpers_sanitize_hallucinated_tool_names() -> None:
    harness = SimpleNamespace(registry=SimpleNamespace(names=lambda: {"write_file", "shell_exec"}))

    assert attempt_tool_sanitization(harness, "write_fileshell_exec") == "write_file"
    assert attempt_tool_sanitization(harness, "shell_exec") is None


def test_harness_core_facade_restores_small_model_name_helper() -> None:
    harness = Harness.__new__(Harness)

    assert harness._is_small_model_name("wrench-9b") is True
    assert harness._is_small_model_name("gpt-oss-120b") is False


def test_harness_initialization_preserves_small_model_flag(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="explore",
        api_key="test-key",
    )

    assert harness.state.scratchpad["_model_is_small"] is True


def test_task_intent_helpers_cover_memory_and_contract_signals() -> None:
    harness = SimpleNamespace(
        provider_profile="generic",
        state=SimpleNamespace(
            current_phase="explore",
            cwd="/tmp/scripts",
            working_memory=SimpleNamespace(failures=[], next_actions=[]),
        ),
        _looks_like_shell_request=lambda task: "shell" in task,
    )

    primary, secondary, tags = extract_intent_state(harness, "remember this exact fact")

    assert primary == "requested_memory_update"
    assert "complete_validation_task" in secondary
    assert "memory_update" in tags
    assert next_action_for_task(harness, "remember this exact fact").startswith("Call `memory_update(")
    assert derive_task_contract("make a plan") == "high_fidelity"
