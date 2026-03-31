from __future__ import annotations

from typing import Any, Awaitable, Callable

from ..state import LoopState
from . import artifact, control, data, fs, http, indexer, indexer_query, memory, network, planning, search, shell
from .base import ToolMode, ToolRisk, ToolSpec, build_tool_schema
from .profiles import (
    CORE_PROFILE,
    DATA_PROFILE,
    MUTATE_PROFILE,
    NETWORK_PROFILE,
    OPS_PROFILE,
    SUPPORT_PROFILE,
)
from .registry import ToolRegistry

INDEXER_PROFILE = "indexer"

Handler = Callable[..., Awaitable[dict[str, Any]]]


def build_registry(
    state_provider: Any,
    *,
    include_ansible: bool = True,
    registry_profiles: set[str] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    state_provider.log.info("build_registry: starting registration")

    def reg(
        name: str,
        description: str,
        schema: dict[str, Any],
        handler: Handler,
        category: str,
        risk: ToolRisk,
        allowed_phases: set[str] | None = None,
        allowed_modes: set[ToolMode] | None = None,
        profiles: set[str] | None = None,
    ) -> None:
        spec = ToolSpec(
            name=name,
            description=description,
            schema=schema,
            handler=handler,
            tier="tier1",
            category=category,
            risk=risk,
            allowed_phases=allowed_phases,
            allowed_modes=allowed_modes,
            profiles=profiles,
        )
        if registry_profiles and profiles and not (profiles & registry_profiles):
            return
        registry.register(spec)

    reg(
        "file_read",
        "Read a text file with optional line slicing. Paths resolve relative to the current cwd; a leading slash or backslash is treated as an absolute path.",
        build_tool_schema(
            required=["path"],
            properties={
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
                "max_bytes": {"type": "integer"},
            },
        ),
        lambda **kwargs: fs.file_read(cwd=state_provider.state.cwd, **kwargs),
        category="filesystem",
        risk="low",
        allowed_modes={"chat", "loop", "indexer", "planning"},
        profiles={CORE_PROFILE},
    )
    reg(
        "dir_list",
        "List directory entries. Paths resolve relative to the current cwd; a leading slash or backslash is treated as an absolute path.",
        build_tool_schema(properties={"path": {"type": "string"}}),
        lambda **kwargs: fs.dir_list(cwd=state_provider.state.cwd, **kwargs),
        category="filesystem",
        risk="low",
        allowed_modes={"chat", "loop", "indexer", "planning"},
        profiles={CORE_PROFILE},
    )
    reg(
        "task_complete",
        "Mark task complete and end loop.",
        build_tool_schema(required=["message"], properties={"message": {"type": "string"}}),
        lambda **kwargs: control.task_complete(state=state_provider.state, **kwargs),
        category="control",
        risk="medium",
        allowed_phases={"explore", "plan", "execute", "verify"},
        allowed_modes={"loop", "planning"},
        profiles={CORE_PROFILE},
    )
    reg(
        "task_fail",
        "Mark task failed and end loop.",
        build_tool_schema(required=["message"], properties={"message": {"type": "string"}}),
        lambda **kwargs: control.task_fail(state=state_provider.state, **kwargs),
        category="control",
        risk="medium",
        allowed_phases={"explore", "plan", "execute", "verify"},
        allowed_modes={"loop", "planning"},
        profiles={CORE_PROFILE},
    )
    reg(
        "ask_human",
        "Request human input and pause loop.",
        build_tool_schema(required=["question"], properties={"question": {"type": "string"}}),
        lambda **kwargs: control.ask_human(state=state_provider.state, **kwargs),
        category="control",
        risk="medium",
        allowed_phases={"explore", "plan", "execute", "verify"},
        allowed_modes={"loop", "planning"},
        profiles={CORE_PROFILE},
    )
    reg(
        "loop_status",
        "Return current loop state summary.",
        build_tool_schema(),
        lambda **kwargs: control.loop_status(state=state_provider.state, **kwargs),
        category="control",
        risk="low",
        allowed_modes={"chat", "loop", "planning"},
        profiles={CORE_PROFILE},
    )

    reg(
        "plan_set",
        "Create or replace the current draft execution plan. Export fields are only for plan documents, never implementation files.",
        build_tool_schema(
            required=["goal", "steps"],
            properties={
                "goal": {
                    "type": "string",
                    "description": "High-level objective the plan is trying to achieve.",
                },
                "summary": {
                    "type": "string",
                    "description": "Optional short context summary for the plan.",
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
        lambda **kwargs: planning.plan_set(state=state_provider.state, **kwargs),
        category="planning",
        risk="low",
        allowed_modes={"planning"},
        profiles={CORE_PROFILE},
    )
    reg(
        "plan_step_update",
        "Update a plan step status during planning or execution.",
        build_tool_schema(
            required=["step_id", "status"],
            properties={
                "step_id": {"type": "string"},
                "status": {"type": "string"},
                "note": {"type": "string"},
            },
        ),
        lambda **kwargs: planning.plan_step_update(state=state_provider.state, **kwargs),
        category="planning",
        risk="low",
        allowed_modes={"planning", "loop"},
        profiles={CORE_PROFILE},
    )
    reg(
        "plan_request_execution",
        "Pause planning and ask the user to approve execution.",
        build_tool_schema(
            required=["question"],
            properties={"question": {"type": "string"}},
        ),
        lambda **kwargs: planning.plan_request_execution(state=state_provider.state, **kwargs),
        category="planning",
        risk="low",
        allowed_modes={"planning"},
        profiles={CORE_PROFILE},
    )
    reg(
        "plan_export",
        "Write the current plan to disk using the canonical exporter.",
        build_tool_schema(
            required=["path"],
            properties={
                "path": {"type": "string"},
                "format": {"type": "string"},
            },
        ),
        lambda **kwargs: planning.plan_export(state=state_provider.state, **kwargs),
        category="planning",
        risk="low",
        allowed_modes={"planning", "loop"},
        profiles={CORE_PROFILE},
    )
    reg(
        "plan_subtask",
        "Run a bounded planning child run.",
        build_tool_schema(
            required=["brief"],
            properties={
                "brief": {"type": "string"},
                "phase": {"type": "string"},
                "constraints": {"type": "array"},
                "acceptance_criteria": {"type": "array"},
            },
        ),
        lambda **kwargs: planning.plan_subtask(harness=state_provider, state=state_provider.state, **kwargs),
        category="planning",
        risk="medium",
        allowed_modes={"planning"},
        profiles={CORE_PROFILE},
    )

    # Remaining Tier 1 tools from Phase 3
    reg("file_write", "Write content to a file.", build_tool_schema(required=["path", "content"], properties={"path": {"type": "string"}, "content": {"type": "string"}}), lambda **kwargs: fs.file_write(cwd=state_provider.state.cwd, **kwargs), category="filesystem", risk="high", allowed_modes={"loop"}, profiles={CORE_PROFILE})
    reg("file_append", "Append content to a file.", build_tool_schema(required=["path", "content"], properties={"path": {"type": "string"}, "content": {"type": "string"}}), lambda **kwargs: fs.file_append(cwd=state_provider.state.cwd, **kwargs), category="filesystem", risk="high", allowed_modes={"loop"}, profiles={CORE_PROFILE})
    reg("file_delete", "Delete a file.", build_tool_schema(required=["path"], properties={"path": {"type": "string"}}), lambda **kwargs: fs.file_delete(cwd=state_provider.state.cwd, **kwargs), category="filesystem", risk="high", allowed_modes={"loop"}, profiles={MUTATE_PROFILE})
    reg("dir_tree", "Show a recursive directory tree.", build_tool_schema(properties={"path": {"type": "string"}, "max_depth": {"type": "integer"}, "max_entries": {"type": "integer"}}), lambda **kwargs: fs.dir_tree(cwd=state_provider.state.cwd, **kwargs), category="filesystem", risk="low", allowed_modes={"chat", "loop", "planning"}, profiles={SUPPORT_PROFILE})
    reg(
        "ssh_exec",
        "Execute a command on a remote host via SSH with live streaming support.",
        build_tool_schema(
            required=["host", "command"],
            properties={
                "host": {"type": "string", "description": "Target hostname or IP."},
                "command": {"type": "string", "description": "Command to run remotely."},
                "user": {"type": "string", "description": "SSH username."},
                "port": {"type": "integer", "default": 22},
                "identity_file": {"type": "string", "description": "Path to SSH private key."},
                "timeout_sec": {"type": "integer", "default": 60},
            },
        ),
        lambda **kwargs: network.ssh_exec(
            state=state_provider.state, 
            harness=state_provider, 
            **kwargs
        ),
        category="network",
        risk="high",
        allowed_modes={"chat", "loop", "planning"},
        profiles={NETWORK_PROFILE},
    )
    reg("shell_exec", "Execute a shell command after user approval.", build_tool_schema(required=["command"], properties={"command": {"type": "string"}, "timeout_sec": {"type": "integer"}}), lambda **kwargs: shell.shell_exec(state=state_provider.state, harness=state_provider, **kwargs), category="shell", risk="high", allowed_modes={"loop"}, profiles={CORE_PROFILE})
    reg("shell_background", "Run shell command in background.", build_tool_schema(required=["command"], properties={"command": {"type": "string"}}), lambda **kwargs: shell.shell_background(state=state_provider.state, harness=state_provider, **kwargs), category="shell", risk="high", allowed_modes={"loop"}, profiles={SUPPORT_PROFILE})
    reg("process_kill", "Kill a tracked background process.", build_tool_schema(required=["job_id"], properties={"job_id": {"type": "string"}}), lambda **kwargs: shell.process_kill(state=state_provider.state, **kwargs), category="shell", risk="high", allowed_modes={"loop"}, profiles={SUPPORT_PROFILE})
    reg("env_get", "Read an environment variable.", build_tool_schema(required=["name"], properties={"name": {"type": "string"}}), shell.env_get, category="environment", risk="low", allowed_modes={"chat", "loop", "planning"}, profiles={SUPPORT_PROFILE})
    reg("env_set", "Set an environment variable.", build_tool_schema(required=["name", "value"], properties={"name": {"type": "string"}, "value": {"type": "string"}}), shell.env_set, category="environment", risk="medium", allowed_modes={"loop"}, profiles={SUPPORT_PROFILE})
    reg("cwd_get", "Get current working directory.", build_tool_schema(), lambda **kwargs: shell.cwd_get(state=state_provider.state, **kwargs), category="environment", risk="low", allowed_modes={"chat", "loop", "planning"}, profiles={SUPPORT_PROFILE})
    reg("cwd_set", "Set current working directory.", build_tool_schema(required=["path"], properties={"path": {"type": "string"}}), lambda **kwargs: shell.cwd_set(state=state_provider.state, **kwargs), category="environment", risk="medium", allowed_modes={"loop"}, profiles={SUPPORT_PROFILE})
    reg("http_get", "Run an HTTP GET request.", build_tool_schema(required=["url"], properties={"url": {"type": "string"}, "headers": {"type": "object"}, "timeout_sec": {"type": "integer"}}), http.http_get, category="http", risk="medium", allowed_modes={"loop"}, profiles={NETWORK_PROFILE})
    reg("http_post", "Run an HTTP POST request.", build_tool_schema(required=["url"], properties={"url": {"type": "string"}, "json_body": {"type": "object"}, "headers": {"type": "object"}, "timeout_sec": {"type": "integer"}}), http.http_post, category="http", risk="high", allowed_modes={"loop"}, profiles={NETWORK_PROFILE})
    reg("file_download", "Download a file from URL.", build_tool_schema(required=["url", "output_path"], properties={"url": {"type": "string"}, "output_path": {"type": "string"}, "headers": {"type": "object"}, "timeout_sec": {"type": "integer"}}), http.file_download, category="http", risk="high", allowed_modes={"loop"}, profiles={NETWORK_PROFILE})
    reg("scratch_set", "Set a scratchpad value.", build_tool_schema(required=["key", "value"], properties={"key": {"type": "string"}, "value": {}, "persist": {"type": "boolean"}}), lambda **kwargs: memory.scratch_set(state=state_provider.state, **kwargs), category="memory", risk="low", allowed_modes={"loop", "planning"}, profiles={SUPPORT_PROFILE})
    reg("scratch_get", "Get scratchpad value.", build_tool_schema(required=["key"], properties={"key": {"type": "string"}}), lambda **kwargs: memory.scratch_get(state=state_provider.state, **kwargs), category="memory", risk="low", allowed_modes={"loop", "planning"}, profiles={SUPPORT_PROFILE})
    reg("scratch_list", "List scratchpad keys.", build_tool_schema(), lambda **kwargs: memory.scratch_list(state=state_provider.state, **kwargs), category="memory", risk="low", allowed_modes={"loop", "planning"}, profiles={SUPPORT_PROFILE})
    reg("scratch_delete", "Delete scratchpad key.", build_tool_schema(required=["key"], properties={"key": {"type": "string"}}), lambda **kwargs: memory.scratch_delete(state=state_provider.state, **kwargs), category="memory", risk="low", allowed_modes={"loop", "planning"}, profiles={SUPPORT_PROFILE})
    reg(
        "memory_update",
        "Update the pinned Working Memory (plan, decisions, known_facts, next_actions). Use this to persist critical facts (like numerical values or key findings) found in large artifacts so they aren't lost when history is truncated.",
        build_tool_schema(
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
        lambda **kwargs: memory.memory_update(state=state_provider.state, **kwargs),
        category="memory",
        risk="low",
        allowed_modes={"loop", "planning"},
        profiles={CORE_PROFILE},
    )
    reg("checkpoint", "Persist current loop state.", build_tool_schema(properties={"label": {"type": "string"}, "output_path": {"type": "string"}}), lambda **kwargs: memory.checkpoint(state=state_provider.state, **kwargs), category="memory", risk="medium", allowed_modes={"loop"}, profiles={SUPPORT_PROFILE})
    reg("grep", "Search file contents recursively.", build_tool_schema(required=["pattern"], properties={"pattern": {"type": "string"}, "path": {"type": "string"}, "regex": {"type": "boolean"}, "case_sensitive": {"type": "boolean"}, "max_results": {"type": "integer"}}), search.grep, category="search", risk="low", allowed_modes={"chat", "loop", "planning"}, profiles={CORE_PROFILE})
    reg("find_files", "Find files recursively.", build_tool_schema(required=["pattern"], properties={"pattern": {"type": "string"}, "path": {"type": "string"}, "regex": {"type": "boolean"}, "max_results": {"type": "integer"}}), search.find_files, category="search", risk="low", allowed_modes={"chat", "loop", "planning"}, profiles={CORE_PROFILE})
    reg("json_query", "Run JMESPath query on JSON data.", build_tool_schema(required=["data", "expression"], properties={"data": {}, "expression": {"type": "string"}}), data.json_query, category="data", risk="low", allowed_modes={"chat", "loop", "planning"}, profiles={DATA_PROFILE})
    reg("yaml_read", "Read YAML file into structured data.", build_tool_schema(required=["path"], properties={"path": {"type": "string"}}), data.yaml_read, category="data", risk="low", allowed_modes={"chat", "loop", "planning"}, profiles={DATA_PROFILE})
    reg("diff", "Generate unified diff for two strings.", build_tool_schema(required=["before", "after"], properties={"before": {"type": "string"}, "after": {"type": "string"}, "context": {"type": "integer"}}), data.diff, category="data", risk="low", allowed_modes={"loop"}, profiles={DATA_PROFILE})
    reg(
        "artifact_read",
        "Read stored artifact content by ID. Large artifacts are returned in bounded line chunks; use start_line/end_line for follow-up paging.",
        build_tool_schema(
            required=["artifact_id"],
            properties={
                "artifact_id": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
                "max_chars": {"type": "integer"},
            },
        ),
        lambda **kwargs: artifact.artifact_read(state=state_provider.state, **kwargs),
        category="data",
        risk="low",
        allowed_modes={"chat", "loop", "indexer", "planning"},
        profiles={CORE_PROFILE},
    )
    reg(
        "artifact_grep",
        "Search for a pattern within an artifact and return matching lines with line numbers. Use this to find specific information within large artifacts WITHOUT reading the entire content.",
        build_tool_schema(
            required=["artifact_id", "query"],
            properties={
                "artifact_id": {"type": "string"},
                "query": {"type": "string"},
                "case_insensitive": {"type": "boolean"},
                "max_results": {"type": "integer"},
            },
        ),
        lambda **kwargs: artifact.artifact_grep(state=state_provider.state, **kwargs),
        category="data",
        risk="low",
        allowed_modes={"chat", "loop", "planning"},
        profiles={CORE_PROFILE},
    )
    reg(
        "artifact_recall",
        "Retrieve the full text content of an artifact and bring it into your context window for analysis. Use for small to medium artifacts that need close inspection.",
        build_tool_schema(required=["artifact_id"], properties={"artifact_id": {"type": "string"}}),
        lambda **kwargs: artifact.artifact_recall(state=state_provider.state, **kwargs),
        category="data",
        risk="low",
        allowed_modes={"chat", "loop", "planning"},
        profiles={CORE_PROFILE},
    )
    reg(
        "artifact_print",
        "Print out the full contents of an artifact directly to the user's console/UI without including it in your context window. Use this for extremely large outputs (like full logs) that the user should see, but you only need confirmation of.",
        build_tool_schema(required=["artifact_id"], properties={"artifact_id": {"type": "string"}}),
        lambda **kwargs: artifact.artifact_print(harness=state_provider, **kwargs),
        category="data",
        risk="low",
        allowed_modes={"chat", "loop", "planning"},
        profiles={CORE_PROFILE},
    )

    if include_ansible:
        async def _tier2_placeholder(**_: Any) -> dict[str, Any]:
            return {"success": False, "error": "tier2 adapter not configured", "output": None, "metadata": {}}

        registry.register(
            ToolSpec(
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
                handler=_tier2_placeholder,
                tier="tier2",
                category="ansible",
                risk="high",
                allowed_phases={"explore", "plan", "execute", "verify"},
                allowed_modes={"loop"},
                profiles={OPS_PROFILE},
            )
        )
        registry.register(
            ToolSpec(
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
                handler=_tier2_placeholder,
                tier="tier2",
                category="ansible",
                risk="high",
                allowed_phases={"plan", "execute", "verify"},
                allowed_modes={"loop"},
                profiles={OPS_PROFILE},
            )
        )
        registry.register(
            ToolSpec(
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
                handler=_tier2_placeholder,
                tier="tier2",
                category="ansible",
                risk="high",
                allowed_phases={"plan", "execute", "verify"},
                allowed_modes={"loop"},
                profiles={OPS_PROFILE},
            )
        )
    
    # Indexing tools
    registry.register(
        ToolSpec(
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
        )
    )
    registry.register(
        ToolSpec(
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
        )
    )
    registry.register(
        ToolSpec(
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
        )
    )
    registry.register(
        ToolSpec(
            name="index_finalize",
            description="Finalize the indexing process and generate a project manifest.",
            schema=build_tool_schema(),
            handler=indexer.index_finalize,
            category="indexer",
            risk="low",
            allowed_modes={"indexer"},
            profiles={INDEXER_PROFILE},
        )
    )
    registry.register(
        ToolSpec(
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
                                "parent": {"type": "string"}
                            },
                            "required": ["name", "kind", "file"]
                        }
                    },
                    "references": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "to_symbol": {"type": "string"},
                                "call_site_file": {"type": "string"},
                                "from_symbol": {"type": "string"},
                                "call_site_line": {"type": "integer"}
                            },
                            "required": ["to_symbol", "call_site_file"]
                        }
                    },
                    "imports": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {"type": "string"},
                                "imported_from": {"type": "string"},
                                "symbols": {"type": "array", "items": {"type": "string"}},
                                "is_external": {"type": "boolean"}
                            },
                            "required": ["file", "imported_from"]
                        }
                    }
                }
            ),
            handler=indexer.index_batch_write,
            category="indexer",
            risk="low",
            allowed_modes={"indexer"},
            profiles={INDEXER_PROFILE},
        )
    )

    # Index query tools
    registry.register(
        ToolSpec(
            name="index_query_symbol",
            description="Search for symbols in the code index by name or fuzzy pattern.",
            schema=build_tool_schema(
                required=["query"],
                properties={"query": {"type": "string"}}
            ),
            handler=indexer_query.index_query_symbol,
            category="indexer",
            risk="low",
        allowed_modes={"chat", "loop", "planning"},
            profiles={CORE_PROFILE, INDEXER_PROFILE},
        )
    )
    registry.register(
        ToolSpec(
            name="index_get_references",
            description="Retrieve all locations where a specific symbol is referenced.",
            schema=build_tool_schema(
                required=["symbol_name"],
                properties={"symbol_name": {"type": "string"}}
            ),
            handler=indexer_query.index_get_references,
            category="indexer",
            risk="low",
        allowed_modes={"chat", "loop", "planning"},
            profiles={CORE_PROFILE, INDEXER_PROFILE},
        )
    )
    registry.register(
        ToolSpec(
            name="index_get_definition",
            description="Retrieve the file path and line numbers where a symbol is defined.",
            schema=build_tool_schema(
                required=["symbol_name"],
                properties={"symbol_name": {"type": "string"}}
            ),
            handler=indexer_query.index_get_definition,
            category="indexer",
            risk="low",
        allowed_modes={"chat", "loop", "planning"},
            profiles={CORE_PROFILE, INDEXER_PROFILE},
        )
    )

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
