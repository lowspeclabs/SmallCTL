from __future__ import annotations

import asyncio
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
from smallctl.client import OpenAICompatClient, StreamResult
from smallctl.graph.model_stream_loop_recovery import handle_model_stream_chunk_error
from smallctl.harness import Harness
from smallctl.harness.task_intent import (
    completion_next_action,
    derive_task_contract,
    extract_intent_state,
    next_action_for_task,
    infer_requested_tool_name,
)
from smallctl.harness.tool_dispatch import (
    attempt_tool_sanitization,
    chat_mode_requires_tools,
    maybe_reuse_file_read,
)
from smallctl.harness.run_mode import ModeDecisionService, normalize_mode_decision
from smallctl.harness.task_classifier import (
    classify_runtime_intent,
    needs_contextual_loop_escalation,
    is_smalltalk,
    looks_like_capability_query,
    looks_like_action_request,
    looks_like_execution_followup,
    looks_like_readonly_chat_request,
    looks_like_shell_request,
    looks_like_write_patch_request,
    needs_loop_for_content_lookup,
    needs_memory_persistence,
    runtime_policy_for_intent,
)
from smallctl.models.conversation import ConversationMessage
from smallctl.state import LoopState


def test_task_classifier_helpers_cover_core_run_mode_signals() -> None:
    assert is_smalltalk("hello")
    assert needs_loop_for_content_lookup("show me the lines 12-18 in the log")
    assert looks_like_execution_followup("please use the command")
    assert looks_like_action_request("run git status")
    assert looks_like_action_request("install caddy server on the remote host 192.168.1.63")
    assert needs_memory_persistence("remember this")
    assert looks_like_shell_request("run a shell command")
    assert looks_like_shell_request("install nginx on 10.0.0.5")
    assert looks_like_readonly_chat_request("what files are in this repo")


def test_write_patch_requests_outrank_shell_script_markers() -> None:
    cases = [
        "start with 1) patch the script",
        "patch the script and test it",
        "fix this python file",
        "update the script to use streaming hashes",
    ]
    for task in cases:
        assert looks_like_shell_request(task) is False
        assert looks_like_write_patch_request(task) is True
        intent = classify_runtime_intent(task, recent_messages=[])
        assert intent.label == "execute"
        assert intent.task_mode == "local_execute"


def test_task_intent_classifies_patch_script_as_file_patch() -> None:
    harness = SimpleNamespace(
        state=SimpleNamespace(current_phase="execute", working_memory=SimpleNamespace(failures=[], next_actions=[])),
        _looks_like_shell_request=looks_like_shell_request,
    )

    assert infer_requested_tool_name(harness, "patch the script and test it") == "file_patch"
    assert infer_requested_tool_name(harness, "fix this python file") == "file_patch"
    assert infer_requested_tool_name(harness, "patch temp/file_deduper.py") == "file_patch"
    assert infer_requested_tool_name(harness, "update ./scripts/cleanup.sh") == "file_patch"
    assert infer_requested_tool_name(harness, "implement a new script") == "write_file"


def test_runtime_intent_classifies_capability_queries_without_overfiring() -> None:
    assert (
        classify_runtime_intent(
            "what tools do you have access to?",
            recent_messages=[],
        ).label
        == "capability_query"
    )
    assert (
        classify_runtime_intent(
            "which tools are available right now?",
            recent_messages=[],
        ).label
        == "capability_query"
    )
    assert (
        classify_runtime_intent(
            "what can you do in this harness?",
            recent_messages=[],
        ).label
        == "capability_query"
    )
    assert (
        classify_runtime_intent(
            "are file tools enabled right now?",
            recent_messages=[],
        ).label
        == "capability_query"
    )
    assert (
        classify_runtime_intent(
            "can you inspect the environment?",
            recent_messages=[],
        ).label
        == "capability_query"
    )
    assert (
        classify_runtime_intent(
            "what mode are you in?",
            recent_messages=[],
        ).label
        == "capability_query"
    )
    assert looks_like_capability_query("what tools would you use for X?", recent_messages=[]) is False
    assert looks_like_capability_query("how do your tools work?", recent_messages=[]) is False
    assert looks_like_capability_query("tooling seems broken", recent_messages=[]) is False
    assert looks_like_capability_query("tooling failure in chat mode", recent_messages=[]) is False
    assert looks_like_capability_query("what can you do to fix this bug?", recent_messages=[]) is False
    assert runtime_policy_for_intent(
        classify_runtime_intent("what tools do you have access to?", recent_messages=[])
    ).route_mode == "loop"
    assert (
        classify_runtime_intent(
            "tooling failure in chat mode",
            recent_messages=[],
        ).label
        == "chat_only"
    )


def test_runtime_intent_uses_recent_capability_context_for_short_followups() -> None:
    recent_messages = [
        ConversationMessage(
            role="assistant",
            content="I can inspect the current harness environment, available tools, and active mode.",
        ),
    ]

    assert (
        classify_runtime_intent(
            "what about file tools?",
            recent_messages=recent_messages,
        ).label
        == "capability_query"
    )


def test_action_like_remote_install_tasks_require_loop_tools() -> None:
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=SimpleNamespace(cwd="/tmp"),
        _current_user_task=lambda: "install caddy server on the remote host 192.168.1.63",
        _runlog=lambda *args, **kwargs: None,
        registry=SimpleNamespace(export_openai_tools=lambda **kwargs: []),
    )

    assert chat_mode_requires_tools(harness, harness._current_user_task()) is True


async def _decide_mode_for(task: str) -> str:
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it", stream_chat=None),
        state=SimpleNamespace(
            planning_mode_enabled=False,
            active_plan=None,
            recent_messages=[],
        ),
        _emit=lambda *args, **kwargs: None,
        _runlog=lambda *args, **kwargs: None,
    )
    return await ModeDecisionService(harness).decide(task)


def test_mode_decision_routes_remote_install_task_to_loop() -> None:
    import asyncio

    mode = asyncio.run(_decide_mode_for("install caddy server on the remote host 192.168.1.63"))

    assert mode == "loop"


def test_mode_decision_routes_plain_greeting_to_chat() -> None:
    import asyncio

    mode = asyncio.run(_decide_mode_for("hello"))

    assert mode == "chat"


def test_mode_decision_routes_capability_query_to_loop_without_model_fallback() -> None:
    async def _unexpected_stream_chat(*, messages, tools):
        del messages, tools
        raise AssertionError("mode decision model should not be consulted for capability queries")
        if False:
            yield {}

    runlog: list[tuple[str, dict[str, object]]] = []
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it", stream_chat=_unexpected_stream_chat),
        state=SimpleNamespace(
            planning_mode_enabled=False,
            active_plan=None,
            recent_messages=[],
        ),
        thinking_start_tag="<think>",
        thinking_end_tag="</think>",
        _emit=lambda *args, **kwargs: None,
        _runlog=lambda event, _message, **data: runlog.append((event, data)),
    )

    mode = asyncio.run(ModeDecisionService(harness).decide("what tools do you have access to?"))

    assert mode == "loop"
    assert any(data.get("intent") == "capability_query" for event, data in runlog if event == "mode_decision")


def test_normalize_mode_decision_accepts_allowlisted_variants() -> None:
    assert normalize_mode_decision("loop") == "loop"
    assert normalize_mode_decision("Loop") == "loop"
    assert normalize_mode_decision("mode l") == "loop"
    assert normalize_mode_decision("l") == "loop"
    assert normalize_mode_decision("loop mode") == "loop"
    assert normalize_mode_decision("chat") == "chat"
    assert normalize_mode_decision("c") == "chat"
    assert normalize_mode_decision("mode chat") == "chat"
    assert normalize_mode_decision("I think loop") is None


def test_mode_decision_invalid_model_output_defaults_to_loop() -> None:
    async def _fake_stream_chat(*, messages, tools):
        del messages, tools
        yield {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": "I think loop",
                        }
                    }
                ]
            },
        }

    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it", stream_chat=_fake_stream_chat),
        state=SimpleNamespace(
            planning_mode_enabled=False,
            active_plan=None,
            recent_messages=[],
        ),
        thinking_start_tag="<think>",
        thinking_end_tag="</think>",
        _emit=lambda *args, **kwargs: None,
        _runlog=lambda *args, **kwargs: None,
    )

    mode = asyncio.run(ModeDecisionService(harness).decide("tell me something interesting"))

    assert mode == "loop"


def test_runtime_intent_policy_contract_aligns_mode_decision_and_chat_tool_requirement() -> None:
    tasks = {
        "what tools do you have access to?": ("capability_query", "loop", True),
        "hello": ("smalltalk", "chat", False),
    }

    for task, (expected_intent, expected_mode, expected_chat_tools) in tasks.items():
        runlog: list[tuple[str, dict[str, object]]] = []
        harness = SimpleNamespace(
            client=SimpleNamespace(model="gemma-4-e2b-it", stream_chat=None),
            state=SimpleNamespace(
                planning_mode_enabled=False,
                active_plan=None,
                recent_messages=[],
            ),
            thinking_start_tag="<think>",
            thinking_end_tag="</think>",
            _emit=lambda *args, **kwargs: None,
            _runlog=lambda event, _message, **data: runlog.append((event, data)),
        )

        intent = classify_runtime_intent(task, recent_messages=[])
        policy = runtime_policy_for_intent(intent)
        mode = asyncio.run(ModeDecisionService(harness).decide(task))

        assert intent.label == expected_intent
        assert policy.chat_requires_tools is expected_chat_tools
        assert mode == expected_mode
        assert ModeDecisionService(harness).chat_mode_requires_tools(task) is expected_chat_tools


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
    readonly_with_sink = "apt list --upgradable 2>/dev/null | grep '/' | wc -l"
    mutating = "cat /var/log/dpkg.log && rm -f /tmp/example"

    assert split_shell_segments(readonly) == ["journalctl -u ssh --no-pager", "tail -100"]
    assert split_shell_segments(readonly_with_sink) == ["apt list --upgradable", "grep '/'", "wc -l"]
    assert is_read_only_shell_evidence_action(readonly) is True
    assert is_read_only_shell_evidence_action(readonly_with_sink) is True
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


def test_stream_chunk_recovery_uses_sub4b_timeout_signature(monkeypatch) -> None:
    state = LoopState(cwd="/tmp")
    harness = SimpleNamespace(
        state=state,
        client=SimpleNamespace(model="qwen-4b"),
        reasoning_mode="off",
        thinking_start_tag="<think>",
        thinking_end_tag="</think>",
        _runlog=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "smallctl.graph.model_stream_fallback_recovery.should_enable_complex_write_chat_draft",
        lambda *args, **kwargs: True,
    )

    result = asyncio.run(
        handle_model_stream_chunk_error(
            harness=harness,
            deps=SimpleNamespace(event_handler=None),
            graph_state=SimpleNamespace(),
            messages=[],
            chunks=[],
            err_msg="tool call continuation timed out",
            details={"reason": "tool_call_continuation_timeout", "message": "deadline exceeded"},
            model_attempt=2,
            chunk_error_max_retries=2,
            timeout_recovery_nudges=0,
            trigger_early_4b_fallback=False,
            salvage_partial_stream=None,
        )
    )

    assert result["trigger_early_4b_fallback"] is True


def test_provider_input_validation_chunk_error_does_not_add_system_nudge() -> None:
    state = LoopState(cwd="/tmp")
    harness = SimpleNamespace(
        state=state,
        client=SimpleNamespace(model="qwen-9b"),
        reasoning_mode="off",
        thinking_start_tag="<think>",
        thinking_end_tag="</think>",
        _runlog=lambda *args, **kwargs: None,
    )
    messages: list[dict[str, object]] = []

    result = asyncio.run(
        handle_model_stream_chunk_error(
            harness=harness,
            deps=SimpleNamespace(event_handler=None),
            graph_state=SimpleNamespace(),
            messages=messages,
            chunks=[],
            err_msg="openrouter/Together input validation failed after retries",
            details={"type": "provider_input_validation", "status_code": 400},
            model_attempt=0,
            chunk_error_max_retries=0,
            timeout_recovery_nudges=0,
            trigger_early_4b_fallback=False,
            salvage_partial_stream=None,
        )
    )

    assert result["retrying"] is False
    assert messages == []
    assert state.recent_messages == []


def test_timeout_recovery_payload_records_tool_call_diagnostics() -> None:
    state = LoopState(cwd="/tmp")

    class _Registry:
        @staticmethod
        def get(tool_name: str):
            if tool_name == "ssh_file_write":
                return SimpleNamespace(schema={"required": ["path", "content"]})
            return None

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        client=SimpleNamespace(model="qwen-9b"),
        reasoning_mode="off",
        thinking_start_tag="<think>",
        thinking_end_tag="</think>",
        _runlog=lambda *args, **kwargs: None,
    )

    partial_stream = StreamResult(
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "ssh_file_write",
                    "arguments": "{\"path\":\"/var/www/html/index.html\"",
                },
            }
        ]
    )
    original_collect_stream = OpenAICompatClient.collect_stream
    OpenAICompatClient.collect_stream = staticmethod(lambda *args, **kwargs: partial_stream)

    try:
        asyncio.run(
            handle_model_stream_chunk_error(
                harness=harness,
                deps=SimpleNamespace(event_handler=None),
                graph_state=SimpleNamespace(),
                messages=[],
                chunks=[],
                err_msg="tool call continuation timed out",
                details={"reason": "tool_call_continuation_timeout", "provider_profile": "lmstudio", "message": "deadline exceeded"},
                model_attempt=0,
                chunk_error_max_retries=0,
                timeout_recovery_nudges=0,
                trigger_early_4b_fallback=False,
                salvage_partial_stream=None,
            )
        )
    finally:
        OpenAICompatClient.collect_stream = original_collect_stream

    payload = state.scratchpad["_last_incomplete_tool_call"]
    diagnostics = payload["tool_call_diagnostics"]
    assert diagnostics[0]["tool_name"] == "ssh_file_write"
    assert diagnostics[0]["present_fields"] == ["path"]
    assert diagnostics[0]["missing_required_fields"] == ["content"]
    assert diagnostics[0]["raw_arguments_preview"].startswith("{\"path\"")


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


def test_task_intent_smalltalk_uses_completion_next_action() -> None:
    harness = SimpleNamespace(
        state=SimpleNamespace(current_phase="explore"),
        _looks_like_shell_request=lambda _task: False,
    )

    assert next_action_for_task(harness, "hello") == completion_next_action()
