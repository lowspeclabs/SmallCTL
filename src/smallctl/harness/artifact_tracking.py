from __future__ import annotations

from typing import Any

from .. import shell_utils
from ..models.tool_result import ToolEnvelope
from ..state import LoopState

file_read_cache_key = shell_utils.file_read_cache_key
ssh_file_read_cache_key = shell_utils.ssh_file_read_cache_key
shell_tokens = shell_utils.shell_tokens
looks_like_env_assignment = shell_utils.looks_like_env_assignment
shell_unwrap_command = shell_utils.shell_unwrap_command
shell_command_root = shell_utils.shell_command_root
shell_attempt_family_key = shell_utils.shell_attempt_family_key
shell_attempt_is_diagnostic = shell_utils.shell_attempt_is_diagnostic


def mark_artifact_superseded(
    *,
    state: LoopState,
    artifact_id: str,
    superseded_by: str,
    family_key: str,
    reason: str,
) -> None:
    shell_utils.mark_artifact_superseded(
        state=state,
        artifact_id=artifact_id,
        superseded_by=superseded_by,
        family_key=family_key,
        reason=reason,
    )


def consolidate_shell_attempt_family(
    *,
    state: LoopState,
    artifact_id: str,
    result: ToolEnvelope,
) -> None:
    shell_utils.consolidate_shell_attempt_family(state=state, artifact_id=artifact_id, result=result)
