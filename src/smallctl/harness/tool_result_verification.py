from __future__ import annotations

import hashlib
import re
import shlex
from typing import Any

from ..docker_retry_normalization import (
    classify_docker_failure,
    docker_failure_is_registry_resolution,
    docker_retry_family,
    docker_retry_key,
    extract_docker_command_target,
)
from .directory_empty_checks import (
    match_directory_empty_check,
    parse_directory_empty_checks,
)
from ..models.tool_result import ToolEnvelope
from ..challenge_progress import record_verifier_result
from ..shell_utils import strip_benign_shell_redirections as _strip_benign_shell_redirections
from .tool_result_verification_constants import (
    _BINARY_PROBE_RE,
    _CURL_VERIFIER_FAILURE_RE,
    _FOG_RESOURCE_RE,
    _INTERACTIVE_PROMPT_RE,
    _LONG_RUNNING_REMOTE_INSTALLER_COMMAND_RE,
    _LONG_RUNNING_REMOTE_INSTALLER_OUTPUT_RE,
    _LS_NO_SUCH_FILE_RE,
    _NGINX_VERIFIER_COMMAND_RE,
    _NGINX_VERIFIER_FAILURE_RE,
    _NOT_FOUND_MARKERS,
    _RAW_SSH_COMMAND_RE,
    _REMOTE_APPLICATION_BLOCKERS,
    _REMOTE_FILE_PRESENCE_PROBE_RE,
    _REMOTE_MUTATING_COMMAND_RE,
    _REMOTE_READBACK_COMMANDS,
    _REMOVAL_ABSENCE_PIPE_RE,
    _REMOVAL_ABSENCE_PROBE_RE,
    _REMOVAL_TASK_KEYWORDS,
    _SSH_AUTH_RECOVERY_KEY,
    _TEST_FAILURE_COUNT_RE,
    _TEST_FAILURE_OUTPUT_RE,
    _TEST_FAILURE_SUMMARY_RE,
    _ZERO_TESTS_RAN_RE,
)
from .tool_result_verification_helpers import (
    _VERIFIER_KIND_STRENGTH,
    classify_execution_failure,
    command_has_write_or_heredoc_shape,
    command_is_binary_probe,
    exit_code_matches,
    looks_like_infinite_loop,
    output_confirms_not_found,
    snip_text,
    verifier_kind_for_command,
    verifier_strength,
)
from .tool_result_verification_assess import assess_remote_mutation_verification
from .tool_result_verification_ssh_recovery import _update_ssh_auth_recovery_state
from .tool_result_verification_artifact import _annotate_verifier_artifact
from .tool_result_verification_removal import (
    _classify_removal_absence_probe,
    _command_is_removal_absence_probe,
    _command_mentions_removal_subject,
    _removal_task_subject_terms,
    _removal_task_text,
    _absence_probe_found_resources,
    _task_has_removal_intent,
)
from .tool_result_verification_semantic import (
    _semantic_verifier_failure,
    _task_or_history_requires_runtime_verifier,
    _prior_failed_verifier_command,
    _passing_verifier_is_weaker_than_prior_failure,
    _insufficient_verifier_message,
)
from .tool_result_verification_timeout import _is_long_running_remote_command_timeout
from .tool_result_verification_blocker import (
    _extract_latest_execution_blocker,
    _store_latest_execution_blocker,
)
from .tool_result_verification_readback import _simple_remote_readback_path
from .tool_result_verification_repair import (
    _record_docker_retry_state,
    _update_acceptance_ledger,
    _update_repair_cycle_state,
)
from .tool_result_verification_audit import _is_audit_task
from .tool_result_verification_store import _store_verifier_verdict



