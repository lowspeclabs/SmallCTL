from __future__ import annotations

from types import SimpleNamespace

from smallctl.context.retrieval import LexicalRetriever
from smallctl.harness.memory import MemoryService
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ExperienceMemory, LoopState


def _memory_service_for_state(state: LoopState) -> MemoryService:
    harness = SimpleNamespace(
        state=state,
        provider_profile="lmstudio",
        context_policy=SimpleNamespace(memory_staleness_step_limit=10, soft_prompt_token_limit=2048),
        _current_user_task=lambda: state.run_brief.original_task,
        cold_memory_store=SimpleNamespace(upsert=lambda memory: memory),
        warm_memory_store=SimpleNamespace(upsert=lambda memory: memory),
    )
    return MemoryService(harness)


def test_memory_service_assigns_local_shell_namespace_for_shell_exec() -> None:
    state = LoopState(cwd="/tmp")
    state.thread_id = "thread-local"
    state.current_phase = "execute"
    state.task_mode = "local_execute"
    state.active_intent = "requested_shell_exec"
    state.intent_tags = ["shell_exec", "phase_execute"]
    state.run_brief.original_task = "run pytest locally"

    memory = _memory_service_for_state(state).record_experience(
        tool_name="shell_exec",
        result=ToolEnvelope(
            success=True,
            output={"stdout": "ok", "stderr": "", "exit_code": 0},
            metadata={"arguments": {"command": "pytest -q"}},
        ),
    )

    assert memory.namespace == "local_shell"


def test_memory_service_assigns_planning_namespace_to_plan_only_terminal_memory() -> None:
    state = LoopState(cwd="/tmp")
    state.thread_id = "thread-plan"
    state.current_phase = "execute"
    state.task_mode = "plan_only"
    state.active_intent = "plan_execution"
    state.intent_tags = ["phase_execute"]
    state.run_brief.original_task = "make a plan first"

    memory = _memory_service_for_state(state).record_experience(
        tool_name="task_complete",
        result=ToolEnvelope(success=True, output={"status": "completed"}, metadata={"status": "completed"}),
        notes="Plan drafted and ready for approval.",
    )

    assert memory.namespace == "planning"


def test_memory_service_assigns_incident_namespace_for_remote_outage_terminal_memory() -> None:
    state = LoopState(cwd="/tmp")
    state.thread_id = "thread-incident"
    state.current_phase = "execute"
    state.task_mode = "remote_execute"
    state.active_intent = "requested_ssh_exec"
    state.intent_tags = ["ssh_exec", "phase_execute"]
    state.run_brief.original_task = "restore service on unreachable remote host after failed deploy"

    memory = _memory_service_for_state(state).record_experience(
        tool_name="task_fail",
        result=ToolEnvelope(success=False, error="service down", metadata={"status": "failed"}),
        notes="Failed deploy left the service down on the remote host; restart and recover the daemon.",
    )

    assert memory.namespace == "incidents"


def test_analysis_retrieval_prefers_coding_and_debugging_namespaces() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.task_mode = "analysis"
    state.active_intent = "general_task"
    state.intent_tags = ["phase_execute"]
    state.run_brief.original_task = "read src/app.py and explain the bug"
    state.working_memory.current_goal = state.run_brief.original_task
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-shell",
            namespace="local_shell",
            tool_name="shell_exec",
            intent="general_task",
            outcome="success",
            confidence=0.9,
            notes="Ran grep on src/app.py to inspect the bug.",
        ),
        ExperienceMemory(
            memory_id="mem-debug",
            namespace="debugging",
            tool_name="file_read",
            intent="general_task",
            outcome="success",
            confidence=0.8,
            notes="Read src/app.py and traced the failing branch.",
        ),
    ]

    bundle = LexicalRetriever().retrieve_bundle(state=state, query=state.run_brief.original_task, include_experiences=True)

    assert bundle.experiences
    assert bundle.experiences[0].memory_id == "mem-debug"


def test_remote_execute_retrieval_prefers_ssh_remote_over_local_shell() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.task_mode = "remote_execute"
    state.active_intent = "general_task"
    state.intent_tags = ["phase_execute"]
    state.run_brief.original_task = "run whoami on the remote host"
    state.working_memory.current_goal = state.run_brief.original_task
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-local",
            namespace="local_shell",
            tool_name="shell_exec",
            intent="general_task",
            outcome="success",
            confidence=0.95,
            notes="Run whoami command and inspect stdout.",
        ),
        ExperienceMemory(
            memory_id="mem-remote",
            namespace="ssh_remote",
            tool_name="memory_update",
            intent="general_task",
            outcome="success",
            confidence=0.75,
            notes="Run whoami command on the remote host and confirm the SSH result.",
        ),
    ]

    bundle = LexicalRetriever().retrieve_bundle(state=state, query=state.run_brief.original_task, include_experiences=True)

    assert bundle.experiences
    assert bundle.experiences[0].memory_id == "mem-remote"


def test_plan_only_retrieval_blocks_execution_namespaces_when_planning_memory_exists() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.task_mode = "plan_only"
    state.active_intent = "plan_execution"
    state.intent_tags = ["phase_execute"]
    state.run_brief.original_task = "make a plan first for the remote repair"
    state.working_memory.current_goal = state.run_brief.original_task
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-plan",
            namespace="planning",
            tool_name="task_complete",
            intent="plan_execution",
            outcome="success",
            confidence=0.8,
            notes="Plan the remote repair in clear steps before execution.",
        ),
        ExperienceMemory(
            memory_id="mem-ssh",
            namespace="ssh_remote",
            tool_name="ssh_exec",
            intent="requested_ssh_exec",
            outcome="success",
            confidence=0.95,
            notes="Run systemctl restart on the remote host.",
        ),
    ]

    bundle = LexicalRetriever().retrieve_bundle(state=state, query=state.run_brief.original_task, include_experiences=True)

    assert [memory.memory_id for memory in bundle.experiences] == ["mem-plan"]
