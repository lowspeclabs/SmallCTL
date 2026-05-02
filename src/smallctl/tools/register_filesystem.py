from __future__ import annotations

from typing import Any, Awaitable, Callable

from . import fs


def register_filesystem_tools(
    *,
    register: Callable[[list[Any]], None],
    make_registration: Callable[..., Any],
    inject_cwd: Callable[[Callable[..., Awaitable[dict[str, Any]]]], Callable[..., Awaitable[dict[str, Any]]]],
    inject_state_and_cwd: Callable[[Callable[..., Awaitable[dict[str, Any]]]], Callable[..., Awaitable[dict[str, Any]]]],
    core_profile: str,
    mutate_profile: str,
    support_profile: str,
) -> None:
    register(
        [
            make_registration(
                name="file_read",
                description=(
                    "Read a text file with optional line slicing. Paths resolve relative to the current cwd. "
                    "For large files, read in chunks to avoid context overflow. If a file was recently written in 'chunked_build' mode, "
                    "verify the final output after the last section is written."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file."},
                        "start_line": {"type": "integer", "description": "Start line (1-indexed)."},
                        "end_line": {"type": "integer", "description": "End line (inclusive)."},
                        "max_bytes": {"type": "integer", "description": "Max bytes to read."},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_cwd(fs.file_read),
                category="filesystem",
                risk="low",
                allowed_modes={"chat", "loop", "indexer", "planning"},
                profiles={core_profile},
            ),
            make_registration(
                name="dir_list",
                description="List directory entries. Paths resolve relative to the current cwd; a leading slash or backslash is treated as an absolute path.",
                schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": [],
                    "additionalProperties": False,
                },
                handler=inject_cwd(fs.dir_list),
                category="filesystem",
                risk="low",
                allowed_modes={"chat", "loop", "indexer", "planning"},
                profiles={core_profile},
            ),
            make_registration(
                name="file_write",
                description=(
                    "Write content to a file. For large files or complex implementations, use chunked mode by providing "
                    "`write_session_id`, section metadata, and an optional `replace_strategy`. Overwrites existing files unless in an active session. "
                    "During an active write session, always pass the target file path as `path`; staged copy paths under `.smallctl/write_sessions/` are for read/verify only."
                ),
                schema={
                    "type": "object",
                    "properties": {
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
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_cwd(fs.file_write),
                category="filesystem",
                risk="high",
                allowed_modes={"chat", "loop"},
                profiles={core_profile},
            ),
            make_registration(
                name="file_patch",
                description=(
                    "Patch a file by replacing exact target text with replacement text. "
                    "Use this for small, precise edits to an existing file or active staged write session. "
                    "During an active write session, always patch the target file path as `path`; staged copy paths under `.smallctl/write_sessions/` are for read/verify only."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file."},
                        "target_text": {"type": "string", "description": "Exact text to replace, including whitespace."},
                        "replacement_text": {"type": "string", "description": "Text to insert in place of the target text."},
                        "expected_occurrences": {"type": "integer", "description": "Expected number of exact matches. Defaults to 1."},
                        "write_session_id": {"type": "string", "description": "ID of the active write session (if any)."},
                        "expected_followup_verifier": {"type": "string", "description": "Optional verifier hint such as 'python -m py_compile'."},
                    },
                    "required": ["path", "target_text", "replacement_text"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_cwd(fs.file_patch),
                category="filesystem",
                risk="high",
                allowed_modes={"chat", "loop"},
                profiles={core_profile},
            ),
            make_registration(
                name="ast_patch",
                description=(
                    "Patch a Python file by targeting structural anchors such as imports, functions, calls, or class fields. "
                    "Use this when an edit is easier to describe by structure than by exact text. "
                    "During an active write session, always patch the target file path as `path`; staged copy paths under "
                    "`.smallctl/write_sessions/` are for read/verify only."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file."},
                        "language": {"type": "string", "description": "Language for the structural edit. v1 supports `python`."},
                        "operation": {"type": "string", "description": "Structural edit operation such as `add_import` or `replace_function`."},
                        "target": {"type": "object", "description": "Operation-specific locator data."},
                        "payload": {"type": "object", "description": "Operation-specific replacement or insertion data."},
                        "write_session_id": {"type": "string", "description": "ID of the active write session (if any)."},
                        "expected_followup_verifier": {"type": "string", "description": "Optional verifier hint such as `python -m py_compile`."},
                        "dry_run": {"type": "boolean", "description": "Preview the structural edit without writing it."},
                    },
                    "required": ["path", "language", "operation", "target"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_cwd(fs.ast_patch),
                category="filesystem",
                risk="high",
                allowed_modes={"chat", "loop"},
                profiles={core_profile},
            ),
            make_registration(
                name="file_delete",
                description="Delete a file.",
                schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_cwd(fs.file_delete),
                category="filesystem",
                risk="high",
                allowed_modes={"loop"},
                profiles={mutate_profile},
            ),
            make_registration(
                name="dir_tree",
                description="Show a recursive directory tree.",
                schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_depth": {"type": "integer"},
                        "max_entries": {"type": "integer"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                handler=inject_cwd(fs.dir_tree),
                category="filesystem",
                risk="low",
                allowed_modes={"chat", "loop", "planning"},
                profiles={support_profile},
            ),
        ]
    )
