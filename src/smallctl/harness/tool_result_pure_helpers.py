from __future__ import annotations

from ..models.tool_result import ToolEnvelope
from ..tools.fs import is_file_mutating_tool
from ..tools.ssh_files import SSH_FILE_MUTATING_TOOLS


def should_auto_record_known_fact(tool_name: str, result: ToolEnvelope) -> bool:
    if tool_name == "shell_exec":
        return False
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if metadata.get("skip_auto_fact_record"):
        return False
    if not result.success and (is_file_mutating_tool(tool_name) or tool_name in SSH_FILE_MUTATING_TOOLS):
        return False
    return True


def is_dry_run_invariant_violation(tool_name: str, result: ToolEnvelope) -> bool:
    if tool_name not in {"ssh_file_patch", "ssh_file_replace_between", "ssh_file_write"}:
        return False
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    dry_run = bool(metadata.get("dry_run"))
    changed = metadata.get("changed")
    return dry_run and bool(changed)
