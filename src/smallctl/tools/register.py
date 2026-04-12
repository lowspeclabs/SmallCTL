from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from . import artifact, control, data, fs, http, indexer, indexer_query, memory, network, planning, search, shell
from .base import ToolMode, ToolRisk, ToolSpec, ToolTier, build_tool_schema
from .profiles import (
    CORE_PROFILE,
    DATA_PROFILE,
    INDEXER_PROFILE,
    MUTATE_PROFILE,
    NETWORK_PROFILE,
    OPS_PROFILE,
    SUPPORT_PROFILE,
)
from .registry import ToolRegistry

Handler = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolRegistration:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Handler
    category: str
    risk: ToolRisk
    allowed_phases: set[str] | None = None
    allowed_modes: set[ToolMode] | None = None
    profiles: set[str] | None = None
    tier: ToolTier = "tier1"


def build_registry(
    state_provider: Any,
    *,
    include_ansible: bool = True,
    registry_profiles: set[str] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    state_provider.log.info("build_registry: starting registration")

    def _inject_cwd(func: Callable[..., Awaitable[dict[str, Any]]]) -> Handler:
        return lambda **kwargs: func(cwd=state_provider.state.cwd, **kwargs)

    def _inject_state_and_cwd(func: Callable[..., Awaitable[dict[str, Any]]]) -> Handler:
        return lambda **kwargs: func(cwd=state_provider.state.cwd, state=state_provider.state, **kwargs)

    def _inject_state(func: Callable[..., Awaitable[dict[str, Any]]]) -> Handler:
        return lambda **kwargs: func(state=state_provider.state, **kwargs)

    def _inject_state_and_harness(func: Callable[..., Awaitable[dict[str, Any]]]) -> Handler:
        return lambda **kwargs: func(state=state_provider.state, harness=state_provider, **kwargs)

    def _inject_harness(func: Callable[..., Awaitable[dict[str, Any]]]) -> Handler:
        return lambda **kwargs: func(harness=state_provider, **kwargs)

    def _register(tools: list[ToolRegistration]) -> None:
        for tool in tools:
            spec = ToolSpec(
                name=tool.name,
                description=tool.description,
                schema=tool.schema,
                handler=tool.handler,
                tier=tool.tier,
                category=tool.category,
                risk=tool.risk,
                allowed_phases=tool.allowed_phases,
                allowed_modes=tool.allowed_modes,
                profiles=tool.profiles,
            )
            if registry_profiles and tool.profiles and not (tool.profiles & registry_profiles):
                continue
            registry.register(spec)

    _register([
        # Filesystem
        ToolRegistration(
            name="file_read",
            description=(
                "Read a text file with optional line slicing. Paths resolve relative to the current cwd. "
                "For large files, read in chunks to avoid context overflow. If a file was recently written in 'chunked_build' mode, "
                "verify the final output after the last section is written."
            ),
            schema=build_tool_schema(
                required=["path"],
                properties={
                    "path": {"type": "string", "description": "Path to file."},
                    "start_line": {"type": "integer", "description": "Start line (1-indexed)."},
                    "end_line": {"type": "integer", "description": "End line (inclusive)."},
                    "max_bytes": {"type": "integer", "description": "Max bytes to read."},
                },
            ),
            handler=_inject_state_and_cwd(fs.file_read),
            category="filesystem",
            risk="low",
            allowed_modes={"chat", "loop", "indexer", "planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="dir_list",
            description="List directory entries. Paths resolve relative to the current cwd; a leading slash or backslash is treated as an absolute path.",
            schema=build_tool_schema(properties={"path": {"type": "string"}}),
            handler=_inject_cwd(fs.dir_list),
            category="filesystem",
            risk="low",
            allowed_modes={"chat", "loop", "indexer", "planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="file_write",
            description=(
                "Write content to a file. For large files or complex implementations, use chunked mode by providing "
                "`write_session_id`, section metadata, and an optional `replace_strategy`. Overwrites existing files unless in an active session. "
                "During an active write session, always pass the target file path as `path`; staged copy paths under `.smallctl/write_sessions/` are for read/verify only."
            ),
            schema=build_tool_schema(
                required=["path", "content"],
                properties={
                    "path": {"type": "string", "description": "Path to file."},
                    "content": {"type": "string", "description": "Content to write."},
                    "write_session_id": {"type": "string", "description": "ID of the active write session (if any)."},
                    "section_name": {"type": "string", "description": "Brief name for this logical block (e.g. 'imports', 'class_def')."},
                    "section_id": {"type": "string", "description": "Optional stable section identifier for chunk-mode writes."},
                    "section_role": {"type": "string", "description": "Optional section role such as 'imports', 'helpers', or 'core_logic'."},
                    "next_section_name": {"type": "string", "description": "Name of the logical block you will write next. Omit for the last chunk."},
                    "replace_strategy": {"type": "string", "description": "Optional write mode override: 'append' or 'overwrite'. Use 'overwrite' for local repair."},
                    "expected_followup_verifier": {"type": "string", "description": "Optional verifier hint such as 'python -m py_compile'."},
                },
            ),
            handler=_inject_state_and_cwd(fs.file_write),
            category="filesystem",
            risk="high",
            allowed_modes={"loop"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="file_append",
            description="Append content to a file. In chunk mode, this also accepts write-session metadata and defaults to append semantics.",
            schema=build_tool_schema(
                required=["path", "content"],
                properties={
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "write_session_id": {"type": "string"},
                    "section_name": {"type": "string"},
                    "section_id": {"type": "string"},
                    "section_role": {"type": "string"},
                    "next_section_name": {"type": "string"},
                    "replace_strategy": {"type": "string"},
                    "expected_followup_verifier": {"type": "string"},
                },
            ),
            handler=_inject_state_and_cwd(fs.file_append),
            category="filesystem",
            risk="high",
            allowed_modes={"loop"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="file_patch",
            description=(
                "Patch a file by replacing exact target text with replacement text. "
                "Use this for small, precise edits to an existing file or active staged write session. "
                "During an active write session, always patch the target file path as `path`; staged copy paths under `.smallctl/write_sessions/` are for read/verify only."
            ),
            schema=build_tool_schema(
                required=["path", "target_text", "replacement_text"],
                properties={
                    "path": {"type": "string", "description": "Path to file."},
                    "target_text": {"type": "string", "description": "Exact text to replace, including whitespace."},
                    "replacement_text": {"type": "string", "description": "Text to insert in place of the target text."},
                    "expected_occurrences": {"type": "integer", "description": "Expected number of exact matches. Defaults to 1."},
                    "write_session_id": {"type": "string", "description": "ID of the active write session (if any)."},
                    "expected_followup_verifier": {"type": "string", "description": "Optional verifier hint such as 'python -m py_compile'."},
                },
            ),
            handler=_inject_state_and_cwd(fs.file_patch),
            category="filesystem",
            risk="high",
            allowed_modes={"loop"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="file_delete",
            description="Delete a file.",
            schema=build_tool_schema(required=["path"], properties={"path": {"type": "string"}}),
            handler=_inject_state_and_cwd(fs.file_delete),
            category="filesystem",
            risk="high",
            allowed_modes={"loop"},
            profiles={MUTATE_PROFILE},
        ),
        ToolRegistration(
            name="dir_tree",
            description="Show a recursive directory tree.",
            schema=build_tool_schema(
                properties={
                    "path": {"type": "string"},
                    "max_depth": {"type": "integer"},
                    "max_entries": {"type": "integer"},
                }
            ),
            handler=_inject_cwd(fs.dir_tree),
            category="filesystem",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={SUPPORT_PROFILE},
        ),
        # Control
        ToolRegistration(
            name="task_complete",
            description="Mark task complete and end loop.",
            schema=build_tool_schema(
                required=["message"], properties={"message": {"type": "string"}}
            ),
            handler=_inject_state(control.task_complete),
            category="control",
            risk="medium",
            allowed_phases={"explore", "plan", "execute", "verify"},
            allowed_modes={"loop", "planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="task_fail",
            description="Mark task failed and end loop.",
            schema=build_tool_schema(
                required=["message"], properties={"message": {"type": "string"}}
            ),
            handler=_inject_state(control.task_fail),
            category="control",
            risk="medium",
            allowed_phases={"explore", "plan", "execute", "verify"},
            allowed_modes={"loop", "planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="ask_human",
            description="Request human input and pause loop.",
            schema=build_tool_schema(
                required=["question"], properties={"question": {"type": "string"}}
            ),
            handler=_inject_state(control.ask_human),
            category="control",
            risk="medium",
            allowed_phases={"explore", "plan", "execute", "verify"},
            allowed_modes={"loop", "planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="loop_status",
            description="Return current loop state summary.",
            schema=build_tool_schema(),
            handler=_inject_state(control.loop_status),
            category="control",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={CORE_PROFILE},
        ),
        # Planning
        ToolRegistration(
            name="plan_set",
            description="Create or replace the current draft execution plan. Provide the full spec contract: goal, inputs, outputs, constraints, acceptance criteria, implementation plan, and steps. Export fields are only for plan documents, never implementation files.",
            schema=build_tool_schema(
                required=["goal", "inputs", "outputs", "constraints", "acceptance_criteria", "implementation_plan", "steps"],
                properties={
                    "goal": {
                        "type": "string",
                        "description": "High-level objective the plan is trying to achieve.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Optional short context summary for the plan.",
                    },
                    "inputs": {
                        "type": "array",
                        "description": "Facts, files, or inputs the implementation will rely on.",
                    },
                    "outputs": {
                        "type": "array",
                        "description": "Artifacts, files, or outcomes the plan should produce.",
                    },
                    "constraints": {
                        "type": "array",
                        "description": "Rules or limits the implementation must respect.",
                    },
                    "acceptance_criteria": {
                        "type": "array",
                        "description": "Checklist items that must be satisfied before task completion.",
                    },
                    "implementation_plan": {
                        "type": "array",
                        "description": "Short implementation stages or ordered authoring plan items.",
                    },
                    "steps": {
                        "type": "array",
                        "description": "Ordered plan steps. Prefer concise step objects with titles over prose paragraphs.",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "object"},
                            ]
                        },
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional plan document export target. Use only .md, .txt, or .text; never pass implementation paths like .py.",
                    },
                    "plan_output_path": {
                        "type": "string",
                        "description": "Alias for output_path. Reserved for plan document exports only.",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["markdown", "md", "text", "txt"],
                        "description": "Optional plan document format. Use markdown/md or text/txt only.",
                    },
                    "plan_output_format": {
                        "type": "string",
                        "enum": ["markdown", "md", "text", "txt"],
                        "description": "Alias for output_format.",
                    },
                },
            ),
            handler=_inject_state_and_harness(planning.plan_set),
            category="planning",
            risk="low",
            allowed_modes={"planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="plan_step_update",
            description="Update a plan step status during planning or execution.",
            schema=build_tool_schema(
                required=["step_id", "status"],
                properties={
                    "step_id": {"type": "string"},
                    "status": {"type": "string"},
                    "note": {"type": "string"},
                },
            ),
            handler=_inject_state_and_harness(planning.plan_step_update),
            category="planning",
            risk="low",
            allowed_modes={"planning", "loop"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="plan_request_execution",
            description="Pause planning and ask the user to approve execution.",
            schema=build_tool_schema(
                required=["question"],
                properties={"question": {"type": "string"}},
            ),
            handler=_inject_state_and_harness(planning.plan_request_execution),
            category="planning",
            risk="low",
            allowed_modes={"planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="plan_export",
            description="Write the current plan to disk using the canonical exporter.",
            schema=build_tool_schema(
                required=["path"],
                properties={
                    "path": {"type": "string"},
                    "format": {"type": "string"},
                },
            ),
            handler=_inject_state_and_harness(planning.plan_export),
            category="planning",
            risk="low",
            allowed_modes={"planning", "loop"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="plan_subtask",
            description="Run a bounded planning child run.",
            schema=build_tool_schema(
                required=["brief"],
                properties={
                    "brief": {"type": "string"},
                    "phase": {"type": "string"},
                    "constraints": {"type": "array"},
                    "acceptance_criteria": {"type": "array"},
                },
            ),
            handler=_inject_state_and_harness(planning.plan_subtask),
            category="planning",
            risk="medium",
            allowed_modes={"planning"},
            profiles={CORE_PROFILE},
        ),
        # Network
        ToolRegistration(
            name="ssh_exec",
            description=(
                "Execute a command on a remote host via SSH with live streaming support. "
                "Prefer `target='user@host'` when a username is known, for example "
                "`target='root@192.168.1.63'`, rather than splitting identity across separate fields."
            ),
            schema=build_tool_schema(
                required=["command"],
                properties={
                    "target": {"type": "string", "description": "Preferred SSH target in `user@host` or `host` form."},
                    "host": {"type": "string", "description": "Target hostname or IP."},
                    "command": {"type": "string", "description": "Command to run remotely."},
                    "user": {"type": "string", "description": "SSH username."},
                    "username": {"type": "string", "description": "Alias for `user`; useful when the task says 'username root'."},
                    "port": {"type": "integer", "default": 22},
                    "identity_file": {"type": "string", "description": "Path to SSH private key."},
                    "password": {"type": "string", "description": "Optional SSH password. Uses `sshpass` when provided."},
                    "timeout_sec": {"type": "integer", "default": 60},
                },
            ),
            handler=_inject_state_and_harness(network.ssh_exec),
            category="network",
            risk="high",
            allowed_modes={"chat", "loop", "planning"},
            profiles={NETWORK_PROFILE},
        ),
        # Shell
        ToolRegistration(
            name="shell_exec",
            description="Execute a shell command after user approval, launch it in background, or poll a background job with job_id.",
            schema=build_tool_schema(
                properties={
                    "command": {"type": "string"},
                    "job_id": {"type": "string"},
                    "background": {"type": "boolean"},
                    "timeout_sec": {"type": "integer"},
                },
            ),
            handler=_inject_state_and_harness(shell.shell_exec),
            category="shell",
            risk="high",
            allowed_modes={"loop"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="process_kill",
            description="Kill a tracked background process.",
            schema=build_tool_schema(
                required=["job_id"], properties={"job_id": {"type": "string"}}
            ),
            handler=_inject_state(shell.process_kill),
            category="shell",
            risk="high",
            allowed_modes={"loop"},
            profiles={SUPPORT_PROFILE},
        ),
        ToolRegistration(
            name="env_get",
            description="Read an environment variable.",
            schema=build_tool_schema(
                required=["name"], properties={"name": {"type": "string"}}
            ),
            handler=shell.env_get,
            category="environment",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={SUPPORT_PROFILE},
        ),
        ToolRegistration(
            name="env_set",
            description="Set an environment variable.",
            schema=build_tool_schema(
                required=["name", "value"],
                properties={"name": {"type": "string"}, "value": {"type": "string"}},
            ),
            handler=shell.env_set,
            category="environment",
            risk="medium",
            allowed_modes={"loop"},
            profiles={SUPPORT_PROFILE},
        ),
        ToolRegistration(
            name="cwd_get",
            description="Get current working directory.",
            schema=build_tool_schema(),
            handler=_inject_state(shell.cwd_get),
            category="environment",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={SUPPORT_PROFILE},
        ),
        ToolRegistration(
            name="cwd_set",
            description="Set current working directory.",
            schema=build_tool_schema(
                required=["path"], properties={"path": {"type": "string"}}
            ),
            handler=_inject_state(shell.cwd_set),
            category="environment",
            risk="medium",
            allowed_modes={"loop"},
            profiles={SUPPORT_PROFILE},
        ),
        # HTTP
        ToolRegistration(
            name="http_get",
            description="Run an HTTP GET request.",
            schema=build_tool_schema(
                required=["url"],
                properties={
                    "url": {"type": "string"},
                    "headers": {"type": "object"},
                    "timeout_sec": {"type": "integer"},
                },
            ),
            handler=http.http_get,
            category="http",
            risk="medium",
            allowed_modes={"loop"},
            profiles={NETWORK_PROFILE},
        ),
        ToolRegistration(
            name="http_post",
            description="Run an HTTP POST request.",
            schema=build_tool_schema(
                required=["url"],
                properties={
                    "url": {"type": "string"},
                    "json_body": {"type": "object"},
                    "headers": {"type": "object"},
                    "timeout_sec": {"type": "integer"},
                },
            ),
            handler=http.http_post,
            category="http",
            risk="high",
            allowed_modes={"loop"},
            profiles={NETWORK_PROFILE},
        ),
        ToolRegistration(
            name="file_download",
            description="Download a file from URL.",
            schema=build_tool_schema(
                required=["url", "output_path"],
                properties={
                    "url": {"type": "string"},
                    "output_path": {"type": "string"},
                    "headers": {"type": "object"},
                    "timeout_sec": {"type": "integer"},
                },
            ),
            handler=http.file_download,
            category="http",
            risk="high",
            allowed_modes={"loop"},
            profiles={NETWORK_PROFILE},
        ),
        # Memory
        ToolRegistration(
            name="scratch_set",
            description="Set a scratchpad value.",
            schema=build_tool_schema(
                required=["key", "value"],
                properties={
                    "key": {"type": "string"},
                    "value": {},
                    "persist": {"type": "boolean"},
                },
            ),
            handler=_inject_state(memory.scratch_set),
            category="memory",
            risk="low",
            allowed_modes={"loop", "planning"},
            profiles={SUPPORT_PROFILE},
        ),
        ToolRegistration(
            name="scratch_get",
            description="Get scratchpad value.",
            schema=build_tool_schema(
                required=["key"], properties={"key": {"type": "string"}}
            ),
            handler=_inject_state(memory.scratch_get),
            category="memory",
            risk="low",
            allowed_modes={"loop", "planning"},
            profiles={SUPPORT_PROFILE},
        ),
        ToolRegistration(
            name="scratch_list",
            description="List scratchpad keys.",
            schema=build_tool_schema(),
            handler=_inject_state(memory.scratch_list),
            category="memory",
            risk="low",
            allowed_modes={"loop", "planning"},
            profiles={SUPPORT_PROFILE},
        ),
        ToolRegistration(
            name="scratch_delete",
            description="Delete scratchpad key.",
            schema=build_tool_schema(
                required=["key"], properties={"key": {"type": "string"}}
            ),
            handler=_inject_state(memory.scratch_delete),
            category="memory",
            risk="low",
            allowed_modes={"loop", "planning"},
            profiles={SUPPORT_PROFILE},
        ),
        ToolRegistration(
            name="memory_update",
            description="Update the pinned Working Memory (plan, decisions, known_facts, next_actions). Use this to persist critical facts (like numerical values or key findings) found in large artifacts so they aren't lost when history is truncated.",
            schema=build_tool_schema(
                required=["section", "content"],
                properties={
                    "section": {
                        "type": "string",
                        "enum": ["plan", "decisions", "open_questions", "known_facts", "failures", "next_actions"],
                    },
                    "content": {"type": "string"},
                    "action": {"type": "string", "enum": ["add", "remove"]},
                },
            ),
            handler=_inject_state(memory.memory_update),
            category="memory",
            risk="low",
            allowed_modes={"loop", "planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="log_note",
            description="Append a concise note to the session notepad. Notes persist across task-boundary resets within this session.",
            schema=build_tool_schema(
                required=["content"],
                properties={
                    "content": {"type": "string"},
                    "tag": {"type": "string"},
                },
            ),
            handler=_inject_state(memory.log_note),
            category="memory",
            risk="low",
            allowed_modes={"loop", "planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="checkpoint",
            description="Persist current loop state.",
            schema=build_tool_schema(
                properties={"label": {"type": "string"}, "output_path": {"type": "string"}}
            ),
            handler=_inject_state(memory.checkpoint),
            category="memory",
            risk="medium",
            allowed_modes={"loop"},
            profiles={SUPPORT_PROFILE},
        ),
        # Search
        ToolRegistration(
            name="grep",
            description="Search file contents recursively.",
            schema=build_tool_schema(
                required=["pattern"],
                properties={
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "regex": {"type": "boolean"},
                    "case_sensitive": {"type": "boolean"},
                    "max_results": {"type": "integer"},
                },
            ),
            handler=search.grep,
            category="search",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="find_files",
            description="Find files recursively.",
            schema=build_tool_schema(
                required=["pattern"],
                properties={
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "regex": {"type": "boolean"},
                    "max_results": {"type": "integer"},
                },
            ),
            handler=search.find_files,
            category="search",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={CORE_PROFILE},
        ),
        # Data
        ToolRegistration(
            name="json_query",
            description="Run JMESPath query on JSON data.",
            schema=build_tool_schema(
                required=["data", "expression"],
                properties={"data": {}, "expression": {"type": "string"}},
            ),
            handler=data.json_query,
            category="data",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={DATA_PROFILE},
        ),
        ToolRegistration(
            name="yaml_read",
            description="Read YAML file into structured data.",
            schema=build_tool_schema(
                required=["path"], properties={"path": {"type": "string"}}
            ),
            handler=data.yaml_read,
            category="data",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={DATA_PROFILE},
        ),
        ToolRegistration(
            name="diff",
            description="Generate unified diff for two strings.",
            schema=build_tool_schema(
                required=["before", "after"],
                properties={
                    "before": {"type": "string"},
                    "after": {"type": "string"},
                    "context": {"type": "integer"},
                },
            ),
            handler=data.diff,
            category="data",
            risk="low",
            allowed_modes={"loop"},
            profiles={DATA_PROFILE},
        ),
        # Artifact
        ToolRegistration(
            name="artifact_read",
            description="Read stored artifact content by ID. Large artifacts are returned in bounded line chunks; use start_line/end_line for follow-up paging.",
            schema=build_tool_schema(
                required=["artifact_id"],
                properties={
                    "artifact_id": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "max_chars": {"type": "integer"},
                },
            ),
            handler=_inject_state(artifact.artifact_read),
            category="data",
            risk="low",
            allowed_modes={"chat", "loop", "indexer", "planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="artifact_grep",
            description="Search for a pattern within an artifact and return matching lines with line numbers. Use this to find specific information within large artifacts WITHOUT reading the entire content.",
            schema=build_tool_schema(
                required=["artifact_id", "query"],
                properties={
                    "artifact_id": {"type": "string"},
                    "query": {"type": "string"},
                    "case_insensitive": {"type": "boolean"},
                    "max_results": {"type": "integer"},
                },
            ),
            handler=_inject_state(artifact.artifact_grep),
            category="data",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="artifact_recall",
            description="Retrieve the full text content of an artifact and bring it into your context window for analysis. Use for small to medium artifacts that need close inspection.",
            schema=build_tool_schema(
                required=["artifact_id"], properties={"artifact_id": {"type": "string"}}
            ),
            handler=_inject_state(artifact.artifact_recall),
            category="data",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="artifact_print",
            description="Print out the full contents of an artifact directly to the user's console/UI without including it in your context window. Use this for extremely large outputs (like full logs) that the user should see, but you only need confirmation of.",
            schema=build_tool_schema(
                required=["artifact_id"], properties={"artifact_id": {"type": "string"}}
            ),
            handler=_inject_harness(artifact.artifact_print),
            category="data",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={CORE_PROFILE},
        ),
        ToolRegistration(
            name="show_artifact",
            description="Alias for artifact_print. Print out the full contents of an artifact directly to the user's console/UI without including it in your context window.",
            schema=build_tool_schema(
                required=["artifact_id"], properties={"artifact_id": {"type": "string"}}
            ),
            handler=_inject_harness(artifact.artifact_print),
            category="data",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={CORE_PROFILE},
        ),
    ])

    if include_ansible:
        async def _tier2_placeholder(**_: Any) -> dict[str, Any]:
            return {"success": False, "error": "tier2 adapter not configured", "output": None, "metadata": {}}

        _register([
            ToolRegistration(
                name="ansible_inventory",
                description="Manage runtime inventory: list, add_host, add_group, remove_host.",
                schema=build_tool_schema(
                    properties={
                        "action": {"type": "string"},
                        "host": {"type": "string"},
                        "group": {"type": "string"},
                        "variables": {"type": "object"},
                    },
                    required=["action"],
                ),
                handler=_tier2_placeholder,  # type: ignore[arg-type]
                category="ansible",
                risk="high",
                allowed_phases={"explore", "plan", "execute", "verify"},
                allowed_modes={"loop"},
                profiles={OPS_PROFILE},
                tier="tier2",
            ),
            ToolRegistration(
                name="ansible_task",
                description="Run one Ansible module task across hosts.",
                schema=build_tool_schema(
                    properties={
                        "module": {"type": "string"},
                        "args": {"type": "object"},
                        "hosts": {"type": "string"},
                        "become": {"type": "boolean"},
                        "check": {"type": "boolean"},
                        "timeout": {"type": "integer"},
                    },
                    required=["module"],
                ),
                handler=_tier2_placeholder,  # type: ignore[arg-type]
                category="ansible",
                risk="high",
                allowed_phases={"plan", "execute", "verify"},
                allowed_modes={"loop"},
                profiles={OPS_PROFILE},
                tier="tier2",
            ),
            ToolRegistration(
                name="ansible_playbook",
                description="Run an Ansible playbook file or inline tasks.",
                schema=build_tool_schema(
                    properties={
                        "playbook": {"type": "string"},
                        "tasks": {"type": "array"},
                        "hosts": {"type": "string"},
                        "vars": {"type": "object"},
                        "become": {"type": "boolean"},
                        "check": {"type": "boolean"},
                        "tags": {"type": "string"},
                        "limit": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                ),
                handler=_tier2_placeholder,  # type: ignore[arg-type]
                category="ansible",
                risk="high",
                allowed_phases={"plan", "execute", "verify"},
                allowed_modes={"loop"},
                profiles={OPS_PROFILE},
                tier="tier2",
            ),
        ])

    _register([
        # Indexer
        ToolRegistration(
            name="index_write_symbol",
            description="Record a source code symbol (class, function) in the project index.",
            schema=build_tool_schema(
                required=["name", "kind", "file", "start_line", "end_line"],
                properties={
                    "name": {"type": "string"},
                    "kind": {"type": "string"},
                    "file": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "signature": {"type": "string"},
                    "docstring_short": {"type": "string"},
                    "parent": {"type": "string"},
                },
            ),
            handler=indexer.index_write_symbol,
            category="indexer",
            risk="low",
            allowed_modes={"indexer"},
            profiles={INDEXER_PROFILE},
        ),
        ToolRegistration(
            name="index_write_reference",
            description="Record a cross-reference between symbols in the project index.",
            schema=build_tool_schema(
                required=["to_symbol", "call_site_file"],
                properties={
                    "to_symbol": {"type": "string"},
                    "call_site_file": {"type": "string"},
                    "from_symbol": {"type": "string"},
                    "call_site_line": {"type": "integer"},
                },
            ),
            handler=indexer.index_write_reference,
            category="indexer",
            risk="low",
            allowed_modes={"indexer"},
            profiles={INDEXER_PROFILE},
        ),
        ToolRegistration(
            name="index_write_import",
            description="Record a module import dependency in the project index.",
            schema=build_tool_schema(
                required=["file", "imported_from"],
                properties={
                    "file": {"type": "string"},
                    "imported_from": {"type": "string"},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                    "is_external": {"type": "boolean"},
                },
            ),
            handler=indexer.index_write_import,
            category="indexer",
            risk="low",
            allowed_modes={"indexer"},
            profiles={INDEXER_PROFILE},
        ),
        ToolRegistration(
            name="index_finalize",
            description="Finalize the indexing process and generate a project manifest.",
            schema=build_tool_schema(),
            handler=indexer.index_finalize,
            category="indexer",
            risk="low",
            allowed_modes={"indexer"},
            profiles={INDEXER_PROFILE},
        ),
        ToolRegistration(
            name="index_batch_write",
            description="Write multiple indexing records (symbols, references, imports) in a single batch.",
            schema=build_tool_schema(
                properties={
                    "symbols": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "kind": {"type": "string"},
                                "file": {"type": "string"},
                                "start_line": {"type": "integer"},
                                "end_line": {"type": "integer"},
                                "signature": {"type": "string"},
                                "docstring_short": {"type": "string"},
                                "parent": {"type": "string"},
                            },
                            "required": ["name", "kind", "file"],
                        },
                    },
                    "references": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "to_symbol": {"type": "string"},
                                "call_site_file": {"type": "string"},
                                "from_symbol": {"type": "string"},
                                "call_site_line": {"type": "integer"},
                            },
                            "required": ["to_symbol", "call_site_file"],
                        },
                    },
                    "imports": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {"type": "string"},
                                "imported_from": {"type": "string"},
                                "symbols": {"type": "array", "items": {"type": "string"}},
                                "is_external": {"type": "boolean"},
                            },
                            "required": ["file", "imported_from"],
                        },
                    },
                }
            ),
            handler=indexer.index_batch_write,
            category="indexer",
            risk="low",
            allowed_modes={"indexer"},
            profiles={INDEXER_PROFILE},
        ),
        # Index query
        ToolRegistration(
            name="index_query_symbol",
            description="Search for symbols in the code index by name or fuzzy pattern.",
            schema=build_tool_schema(
                required=["query"],
                properties={"query": {"type": "string"}},
            ),
            handler=indexer_query.index_query_symbol,
            category="indexer",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={CORE_PROFILE, INDEXER_PROFILE},
        ),
        ToolRegistration(
            name="index_get_references",
            description="Retrieve all locations where a specific symbol is referenced.",
            schema=build_tool_schema(
                required=["symbol_name"],
                properties={"symbol_name": {"type": "string"}},
            ),
            handler=indexer_query.index_get_references,
            category="indexer",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={CORE_PROFILE, INDEXER_PROFILE},
        ),
        ToolRegistration(
            name="index_get_definition",
            description="Retrieve the file path and line numbers where a symbol is defined.",
            schema=build_tool_schema(
                required=["symbol_name"],
                properties={"symbol_name": {"type": "string"}},
            ),
            handler=indexer_query.index_get_definition,
            category="indexer",
            risk="low",
            allowed_modes={"chat", "loop", "planning"},
            profiles={CORE_PROFILE, INDEXER_PROFILE},
        ),
    ])

    return registry


def register_mock_tool(
    registry: ToolRegistry,
    name: str,
    description: str,
    handler: Handler,
    required: list[str] | None = None,
    properties: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Helper for registering mock tools with common patterns.

    This function consolidates the schema building patterns used in AHO,
    reducing code duplication and ensuring consistency with core tool registration.
    """
    registry.register(
        ToolSpec(
            name=name,
            description=description,
            schema=build_tool_schema(required=required or [], properties=properties or {}),
            handler=handler,
            tier="tier1",
            category="mock",
            risk="low",
            allowed_phases={"explore", "plan", "execute", "verify"},
            allowed_modes={"loop"},
            profiles={CORE_PROFILE},
            **kwargs,
        )
    )
