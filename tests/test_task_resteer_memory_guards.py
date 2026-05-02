from __future__ import annotations

from types import SimpleNamespace

from smallctl.context.assembler import PromptAssembler
from smallctl.context.retrieval import LexicalRetriever
from smallctl.harness.memory import MemoryService
from smallctl.harness.task_intent import infer_requested_tool_name
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ExperienceMemory, LoopState


def test_remote_followup_prefers_ssh_exec_intent() -> None:
    harness = SimpleNamespace(_looks_like_shell_request=lambda task: True)

    tool_name = infer_requested_tool_name(
        harness,
        "run apt-get remove -y nginx on the remote host 192.168.1.63 without sudo",
    )

    assert tool_name == "ssh_exec"


def test_terminal_experience_redacts_password_like_text() -> None:
    state = LoopState(cwd="/tmp")
    state.thread_id = "thread-1"
    state.run_brief.original_task = "ssh into the remote host"
    harness = SimpleNamespace(
        state=state,
        provider_profile="lmstudio",
        context_policy=SimpleNamespace(memory_staleness_step_limit=10, soft_prompt_token_limit=2048),
        _current_user_task=lambda: state.run_brief.original_task,
        cold_memory_store=SimpleNamespace(upsert=lambda memory: memory),
        warm_memory_store=SimpleNamespace(upsert=lambda memory: memory),
    )
    service = MemoryService(harness)

    memory = service.record_experience(
        tool_name="task_complete",
        result=ToolEnvelope(success=True, output={"status": "completed"}, metadata={"status": "completed"}),
        notes='Task complete: SSH to root@192.168.1.63 with password "@S02v1735" succeeded.',
    )

    assert "[REDACTED]" in memory.notes
    assert "@S02v1735" not in memory.notes


def test_retrieval_and_rendering_deemphasize_redacted_task_complete_memories() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.active_intent = "requested_ssh_exec"
    state.intent_tags = ["ssh_exec", "lmstudio"]
    state.run_brief.original_task = "run apt-get remove -y nginx on the remote host 192.168.1.63"
    state.working_memory.current_goal = state.run_brief.original_task

    task_complete = ExperienceMemory(
        memory_id="mem-task-complete",
        tool_name="task_complete",
        intent="requested_ssh_exec",
        intent_tags=["ssh_exec", "lmstudio"],
        environment_tags=["lmstudio"],
        outcome="success",
        confidence=0.95,
        notes='Task complete: SSH to root@192.168.1.63 with password "@S02v1735" succeeded.',
    )
    ssh_exec = ExperienceMemory(
        memory_id="mem-ssh-exec",
        tool_name="ssh_exec",
        intent="requested_ssh_exec",
        intent_tags=["ssh_exec", "lmstudio"],
        environment_tags=["lmstudio"],
        outcome="success",
        confidence=0.85,
        notes="Successfully called ssh_exec. Key pattern: ['host', 'user', 'auth', 'command'].",
    )
    state.warm_experiences = [task_complete, ssh_exec]

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query=state.run_brief.original_task,
        include_experiences=True,
    )

    assert bundle.experiences
    assert bundle.experiences[0].memory_id == "mem-ssh-exec"

    rendered = PromptAssembler()._render_warm_item(task_complete)
    assert "[REDACTED]" in rendered
    assert "@S02v1735" not in rendered


def test_chat_task_mode_filters_remote_execution_memories() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.task_mode = "chat"
    state.active_intent = "general_task"
    state.intent_tags = ["phase_execute"]
    state.run_brief.original_task = "hello"
    state.working_memory.current_goal = "hello"

    ssh_exec = ExperienceMemory(
        memory_id="mem-ssh-exec",
        tool_name="ssh_exec",
        intent="requested_ssh_exec",
        intent_tags=["ssh_exec"],
        outcome="success",
        confidence=0.95,
        notes="Successfully called ssh_exec. Key pattern: ['host', 'user', 'auth', 'command'].",
    )
    general = ExperienceMemory(
        memory_id="mem-general",
        tool_name="memory_update",
        intent="general_task",
        intent_tags=["task_greeting"],
        outcome="success",
        confidence=0.7,
        notes="Friendly chat requests should stay conversational.",
    )
    state.warm_experiences = [ssh_exec, general]

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query="hello",
        include_experiences=True,
    )

    assert [memory.memory_id for memory in bundle.experiences] == ["mem-general"]


def test_memory_service_canonicalizes_successful_ssh_exec_pattern_notes() -> None:
    state = LoopState(cwd="/tmp")
    state.thread_id = "thread-1"
    state.current_phase = "execute"
    state.active_intent = "requested_ssh_exec"
    state.intent_tags = ["ssh_exec", "lmstudio"]
    state.run_brief.original_task = "run whoami on the remote host"
    harness = SimpleNamespace(
        state=state,
        provider_profile="lmstudio",
        context_policy=SimpleNamespace(memory_staleness_step_limit=10, soft_prompt_token_limit=2048),
        _current_user_task=lambda: state.run_brief.original_task,
        cold_memory_store=SimpleNamespace(upsert=lambda memory: memory),
        warm_memory_store=SimpleNamespace(upsert=lambda memory: memory),
    )
    service = MemoryService(harness)

    memory = service.record_experience(
        tool_name="ssh_exec",
        result=ToolEnvelope(
            success=True,
            output={"stdout": "root", "stderr": "", "exit_code": 0},
            metadata={
                "arguments": {
                    "host": "192.168.1.63",
                    "user": "root",
                    "password": "@S02v1735",
                    "command": "whoami",
                }
            },
        ),
    )

    assert memory.notes == "Successfully called ssh_exec. Key pattern: ['host', 'user', 'auth', 'command']."
    assert memory.namespace == "ssh_remote"
