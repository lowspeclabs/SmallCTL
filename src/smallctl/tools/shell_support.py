from __future__ import annotations

import shlex
import re
from pathlib import Path
from typing import Any

from ..state import LoopState
from .common import fail
from .fs_sessions import _same_target_path, _write_session_can_finalize
from .fs_write_session_policy import _write_session_resume_metadata
from .shell_parsing import (
    _simple_shell_command_segments,
    _split_shell_command_segments,
    _split_shell_words,
    _strip_environment_and_wrappers,
)
from .shell_path_utils import (
    _is_within_path,
    _path_alias_mentioned,
    _safe_resolve_path,
    _target_path_aliases,
    _token_path_candidates,
)
from .shell_support_constants import (
    _ARGPARSE_REQUIRED_ARGS_PATTERN,
    _DEB822_FIELDS,
    _DETACHED_COMMAND_MARKERS,
    _DISPOSABLE_PATH_NAMES,
    _DISPOSABLE_PATH_SUFFIXES,
    _FOLLOW_FLAGS,
    _FOREGROUND_BINARIES,
    _FOREGROUND_SUBCOMMANDS,
    _INSPECTION_FLAGS,
    _INVALID_INPUT_MARKERS,
    _PACKAGE_RUNNERS,
    _REMOTE_INSTALLER_PREFLIGHT_KEY,
    _SERVICE_MANAGER_COMMANDS,
    _SHELL_CONTROL_TOKENS,
    _SINGLE_ANSWER_PIPE_PATTERN,
    _SOURCE_OR_TEST_DIR_NAMES,
    _SOURCE_OR_TEST_SUFFIXES,
    _YES_PIPE_PATTERN,
)
from .shell_support_delete_guards import (
    _explicit_delete_requested,
    _extract_shell_delete_targets,
    _is_disposable_delete_target,
    _looks_like_source_or_test_artifact,
    _protected_working_set_paths,
    _shell_workspace_destructive_delete_guard,
    _target_contains_protected_path,
)
from .shell_support_installer_guards import (
    _interactive_installer_yes_pipe_guard,
    _installer_command_suggested_timeout,
    _remote_installer_preflight_guard,
    _looks_like_remote_installer_mutation,
    _remote_installer_cwd_and_script,
    _remote_installer_preflight_checks,
    _mark_remote_installer_preflight_clean,
    _mark_remote_installer_preflight_clean_from_write,
    _remote_installer_preflight_has_verified_write,
    _expose_interactive_session_tools,
    _looks_like_interactive_installer_target,
    _looks_like_interactive_installer_word,
)
from .shell_support_foreground_guards import (
    _foreground_command_guard,
    _likely_long_running_foreground_reason,
    _has_detached_or_bounded_marker,
    _likely_long_running_simple_command_reason,
)
from .shell_support_write_session_guards import (
    _shell_write_session_target_path_guard,
    _shell_write_session_artifact_delete_guard,
    _shell_execution_authoring_guard,
    _write_session_delete_targets,
    _protected_write_session_paths,
    _targets_write_session_artifact,
    _command_targets_path,
)
from .shell_support_invalid_input import InvalidInputLoopDetector
from .shell_support_argparse import (
    _extract_missing_argparse_arguments,
    _build_argparse_missing_args_question,
    _detect_unsupported_shell_syntax,
)
from .shell_support_misc import (
    _shell_workspace_relative_hint,
    _shell_status_update_interval,
    _build_shell_status_update,
)
from .shell_support_apt_and_outcome import (
    _apt_deb822_preflight_guard,
    _apt_sources_list_d_guard,
    _is_deb822_preflight_clean,
    _looks_like_deb822_validator,
    _mark_deb822_preflight_clean,
    classify_shell_outcome,
    record_apt_update_result,
    record_sources_list_d_modification,
    validate_sources_file,
)


def guard_fail(
    message: str,
    *,
    reason: str,
    command: str,
    error_kind: str | None = None,
    next_required_tool: dict[str, Any] | None = None,
    next_required_action: dict[str, Any] | str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a consistent guard failure result."""
    metadata: dict[str, Any] = {
        "reason": reason,
        "command": command,
    }
    if error_kind is not None:
        metadata["error_kind"] = error_kind
    if next_required_tool is not None:
        metadata["next_required_tool"] = next_required_tool
    if next_required_action is not None:
        metadata["next_required_action"] = next_required_action
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    return fail(message, metadata=metadata)



